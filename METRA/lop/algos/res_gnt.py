from torch.nn import Conv2d, Linear, BatchNorm2d
from torch import where, topk, long, empty, zeros, no_grad
from math import sqrt
import torch
import sys
from torch.nn.init import calculate_gain


def get_layer_bound(layer, init, gain):
    if isinstance(layer, Conv2d):
        return sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


def get_layer_std(layer, gain):
    if isinstance(layer, Conv2d):
        return gain * sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, Linear):
        return gain * sqrt(1 / layer.in_features)


class ResGnT(object):
    """
    Generate-and-Test algorithm for a simple resnet, assuming only one fully connected layer at the top.

    CReLU support:
    - the network's activations are doubled after each conv/bn when using CReLU
    - but CBP should still track/replace the underlying conv filters, not the doubled channels
    - so we merge paired CReLU activation channels back into base channels before computing utility
    """
    def __init__(self, net, hidden_activation, decay_rate=0.99, replacement_rate=1e-4, util_type='weight',
                 maturity_threshold=1000, device=torch.device("cpu")):
        super(ResGnT, self).__init__()

        self.net = net
        self.hidden_activation = hidden_activation.lower()
        self.is_crelu = self.hidden_activation == "crelu"

        self.bn_layers = []
        self.weight_layers = []
        self.get_weight_layers(nn_module=self.net)
        self.num_hidden_layers = len(self.weight_layers) - 1
        self.device = device

        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        self.util, self.ages, self.mean_feature_mag = [], [], []

        # Track underlying feature units, not post-CReLU doubled channels.
        for i in range(self.num_hidden_layers):
            num_units = self._base_num_units(i)
            self.util.append(zeros(num_units, dtype=torch.float32, device=self.device))
            self.ages.append(zeros(num_units, dtype=torch.float32, device=self.device))
            self.mean_feature_mag.append(zeros(num_units, dtype=torch.float32, device=self.device))

        self.accumulated_num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        self.m = torch.nn.Softmax(dim=1)

        # Use ReLU gain for CReLU base filters.
        gain_activation = "relu" if self.is_crelu else self.hidden_activation
        self.stds = self.compute_std(hidden_activation=gain_activation)

        self.num_new_features_to_replace = []
        for i in range(self.num_hidden_layers):
            with no_grad():
                self.num_new_features_to_replace.append(self.replacement_rate * self._base_num_units(i))

    def _base_num_units(self, layer_idx):
        layer = self.weight_layers[layer_idx]
        if isinstance(layer, Conv2d):
            return layer.out_channels
        if isinstance(layer, Linear):
            return layer.out_features
        raise TypeError(f"Unsupported layer type: {type(layer)}")

    def _merge_crelu_features(self, feat):
        """
        Merge doubled CReLU activation channels back to the underlying base channels.

        For conv activations:
            [N, 2C, H, W] -> [N, C, H, W]
        For fc activations:
            [N, 2H] -> [N, H]

        We use abs-max pairing:
            merged = max(pos_half, neg_half)
        """
        if not self.is_crelu:
            return feat

        if feat.dim() == 4:
            c2 = feat.shape[1]
            if c2 % 2 != 0:
                raise ValueError(f"CReLU conv feature channels must be even, got {c2}")
            c = c2 // 2
            pos = feat[:, :c, :, :]
            neg = feat[:, c:, :, :]
            return torch.maximum(pos.abs(), neg.abs())

        if feat.dim() == 2:
            h2 = feat.shape[1]
            if h2 % 2 != 0:
                raise ValueError(f"CReLU fc feature width must be even, got {h2}")
            h = h2 // 2
            pos = feat[:, :h]
            neg = feat[:, h:]
            return torch.maximum(pos.abs(), neg.abs())

        return feat

    def _next_layer_outgoing_magnitude(self, layer_idx):
        """
        Return outgoing weight magnitude aligned to the base feature units of layer_idx.
        """
        next_layer = self.weight_layers[layer_idx + 1]

        if isinstance(next_layer, Linear):
            # weight shape [out_features, in_features]
            mag = next_layer.weight.data.abs().mean(dim=0)
        elif isinstance(next_layer, Conv2d):
            # weight shape [out_channels, in_channels, kH, kW]
            mag = next_layer.weight.data.abs().mean(dim=(0, 2, 3))
        else:
            raise TypeError(f"Unsupported next layer type: {type(next_layer)}")

        if self.is_crelu:
            if mag.numel() % 2 != 0:
                raise ValueError(
                    f"CReLU expected even outgoing magnitude width, got {mag.numel()} at layer {layer_idx}"
                )
            half = mag.numel() // 2
            mag = torch.maximum(mag[:half], mag[half:])

        return mag

    def _zero_outgoing_weights_for_replaced_features(self, next_layer, feature_indices):
        """
        Zero outgoing weights corresponding to replaced base features.
        For CReLU, each base feature maps to two adjacent halves in the next layer input.
        """
        if not self.is_crelu:
            next_layer.weight.data[:, feature_indices] = 0
            return

        total_in = next_layer.weight.data.shape[1]
        if total_in % 2 != 0:
            raise ValueError(f"CReLU expected even next-layer input width, got {total_in}")

        half = total_in // 2
        next_layer.weight.data[:, feature_indices] = 0
        next_layer.weight.data[:, feature_indices + half] = 0

    def get_weight_layers(self, nn_module: torch.nn.Module):
        if isinstance(nn_module, Conv2d) or isinstance(nn_module, Linear):
            self.weight_layers.append(nn_module)
        elif isinstance(nn_module, BatchNorm2d):
            self.bn_layers.append(nn_module)
        else:
            for m in nn_module.children():
                if hasattr(nn_module, 'downsample'):
                    if nn_module.downsample == m:
                        continue
                self.get_weight_layers(nn_module=m)

    def compute_std(self, hidden_activation):
        stds = []
        gain = calculate_gain(nonlinearity=hidden_activation)
        for i in range(self.num_hidden_layers):
            stds.append(get_layer_std(layer=self.weight_layers[i], gain=gain))
        stds.append(get_layer_std(layer=self.weight_layers[-1], gain=1))
        return stds

    def test_features(self, features):
        """
        Args:
            features: activation values collected from the network
        Returns:
            features_to_replace, num_features_to_replace
        """
        features_to_replace = [empty(0, dtype=long, device=self.device) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]

        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace

        for i in range(self.num_hidden_layers):
            self.ages[i] += 1

            with torch.no_grad():
                feat = self._merge_crelu_features(features[i])

                if feat.dim() == 2:
                    self.mean_feature_mag[i] *= self.decay_rate
                    self.mean_feature_mag[i] += (1 - self.decay_rate) * feat.abs().mean(dim=0)
                elif feat.dim() == 4:
                    self.mean_feature_mag[i] *= self.decay_rate
                    self.mean_feature_mag[i] += (1 - self.decay_rate) * feat.abs().mean(dim=(0, 2, 3))
                else:
                    raise ValueError(f"Unsupported feature shape: {tuple(feat.shape)}")

            eligible_feature_indices = where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue

            self.accumulated_num_features_to_replace[i] += self.num_new_features_to_replace[i]

            num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
            self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace

            if num_new_features_to_replace == 0:
                continue

            with torch.no_grad():
                output_weight_mag = self._next_layer_outgoing_magnitude(i)

                if self.util_type == 'weight':
                    self.util[i] = output_weight_mag
                elif self.util_type == 'contribution':
                    self.util[i] = output_weight_mag * self.mean_feature_mag[i]
                else:
                    raise ValueError(f"Unsupported util_type: {self.util_type}")

            new_features_to_replace = topk(
                -self.util[i][eligible_feature_indices],
                num_new_features_to_replace
            )[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            self.util[i][new_features_to_replace] = 0

            num_features_to_replace[i] = num_new_features_to_replace
            features_to_replace[i] = new_features_to_replace

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        """
        Reset low-utility underlying filters/features.
        """
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue

                current_layer = self.weight_layers[i]
                next_layer = self.weight_layers[i + 1]
                idx = features_to_replace[i]

                current_layer.weight.data[idx, :] *= 0.0
                current_layer.weight.data[idx, :] += empty(
                    [num_features_to_replace[i]] + list(current_layer.weight.shape[1:]),
                    device=self.device
                ).normal_(std=self.stds[i])

                if current_layer.bias is not None:
                    current_layer.bias.data[idx] *= 0.0

                self._zero_outgoing_weights_for_replaced_features(next_layer, idx)

                self.ages[i][idx] = 0
                self.mean_feature_mag[i][idx] = 0
                self.util[i][idx] = 0

                # BatchNorm tracks underlying conv channels, so base indices are correct.
                self.bn_layers[i].bias.data[idx] *= 0.0
                self.bn_layers[i].weight.data[idx] *= 0.0
                self.bn_layers[i].weight.data[idx] += 1.0
                self.bn_layers[i].running_mean.data[idx] *= 0.0
                self.bn_layers[i].running_var.data[idx] *= 0.0
                self.bn_layers[i].running_var.data[idx] += 1.0

    def gen_and_test(self, features):
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()

        features_to_replace, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace, num_features_to_replace)