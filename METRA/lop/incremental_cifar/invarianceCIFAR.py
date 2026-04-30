import torch
import numpy as np
import torch.nn.functional as F

def apply_semantic_transforms_cifar(x):
    """
    x: [N, C, H, W] float tensor, expected CIFAR-style tensors (e.g. [N, 3, 32, 32])
    Returns transformed x' with one label-preserving transform sampled per image.

    Assumes tensors are already normalized. Spatial transforms remain valid.
    """
    assert x.dim() == 4, f"Expected [N, C, H, W], got {tuple(x.shape)}"
    n, c, h, w = x.shape
    out = []

    for img in x:
        choice = np.random.choice(["shift_x", "shift_y", "flip", "rot90", "crop", "identity"])

        if choice == "shift_x":
            k = np.random.randint(-2, 3)
            t = torch.roll(img, shifts=k, dims=2)   # width axis after [C,H,W]

        elif choice == "shift_y":
            k = np.random.randint(-2, 3)
            t = torch.roll(img, shifts=k, dims=1)   # height axis

        elif choice == "flip":
            t = torch.flip(img, dims=[2])           # horizontal flip

        elif choice == "rot90":
            k = np.random.choice([0, 1, 3])         # 0, 90, 270 degrees
            t = torch.rot90(img, k=int(k), dims=(1, 2))

        elif choice == "crop":
            # reflect-pad by 4 then random crop back to 32x32
            padded = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            top = np.random.randint(0, 9)
            left = np.random.randint(0, 9)
            t = padded[:, top:top + h, left:left + w]

        else:
            t = img

        out.append(t)

    return torch.stack(out, dim=0)


@torch.no_grad()
def get_layer_activations_cifar(net, x):
    """
    Uses the ResNet API already present in your CIFAR code:
    net.forward(x, features_list)
    Returns a list of intermediate activations.
    """
    reps = []
    _ = net.forward(x, reps)
    return reps


def _positive_part(a, use_abs=False):
    return a.abs() if use_abs else a.clamp(min=0)


def compute_unit_thresholds_cifar(layer_act, firing_rate=0.01, use_abs=False):
    """
    layer_act can be:
      - [N, C, H, W] for conv feature maps
      - [N, H] for fully connected features

    Returns tau:
      - [C] for conv features
      - [H] for fc features
    """
    a = _positive_part(layer_act, use_abs=use_abs)

    if a.dim() == 4:
        # flatten over batch and spatial locations -> threshold per channel
        # [N, C, H, W] -> [C, N*H*W]
        flat = a.permute(1, 0, 2, 3).contiguous().view(a.shape[1], -1)
        q = 1.0 - firing_rate
        tau = torch.quantile(flat, q=q, dim=1)
        return tau

    if a.dim() == 2:
        q = 1.0 - firing_rate
        tau = torch.quantile(a, q=q, dim=0)
        return tau

    raise ValueError(f"Unsupported activation shape: {tuple(layer_act.shape)}")


def invariance_score_for_layer_cifar(act_orig, act_trans, tau, use_abs=False):
    """
    Returns:
      per-unit invariance score
      mean layer invariance scalar

    For conv layers, a channel is considered 'firing' for a sample if any spatial
    position exceeds the threshold.
    """
    a0 = _positive_part(act_orig, use_abs=use_abs)
    a1 = _positive_part(act_trans, use_abs=use_abs)

    if a0.dim() == 4:
        # [N, C, H, W], tau [C]
        fire0 = (a0 > tau.view(1, -1, 1, 1)).any(dim=3).any(dim=2)   # [N, C]
        fire1 = (a1 > tau.view(1, -1, 1, 1)).any(dim=3).any(dim=2)   # [N, C]

    elif a0.dim() == 2:
        # [N, H], tau [H]
        fire0 = a0 > tau.unsqueeze(0)
        fire1 = a1 > tau.unsqueeze(0)

    else:
        raise ValueError(f"Unsupported activation shape: {tuple(a0.shape)}")

    p_fire1 = fire1.float().mean(dim=0).clamp(min=1e-8)
    denom = fire0.float().sum(dim=0).clamp(min=1.0)
    p_cond = (fire0 & fire1).float().sum(dim=0) / denom

    score = p_cond / p_fire1
    return score, score.mean().item()


@torch.no_grad()
def compute_invariance_scores_cifar(net, x_eval, use_abs=False, firing_rate=0.01):
    """
    x_eval: [N, 3, 32, 32]
    returns:
      layer_scores: list of mean invariance scores, one per collected layer
      per_unit_scores: list of tensors, one per collected layer
    """
    x_t = apply_semantic_transforms_cifar(x_eval)

    reps_orig = get_layer_activations_cifar(net, x_eval)
    reps_trans = get_layer_activations_cifar(net, x_t)

    layer_scores = []
    per_unit_scores = []

    for act0, act1 in zip(reps_orig, reps_trans):
        tau = compute_unit_thresholds_cifar(act0, firing_rate=firing_rate, use_abs=use_abs)
        s_unit, s_layer = invariance_score_for_layer_cifar(act0, act1, tau, use_abs=use_abs)
        per_unit_scores.append(s_unit.cpu())
        layer_scores.append(s_layer)

    return layer_scores, per_unit_scores