import torch
import numpy as np

def apply_semantic_transforms_mnist(x):
    """
    x: [N, 784] float tensor in flattened MNIST format
    Returns transformed x': [N, 784]
    """
    x2 = x.view(-1, 1, 28, 28)

    # pick one label-preserving transform per sample
    out = []
    for img in x2:
        choice = np.random.choice(["shift_x", "shift_y", "rot", "identity"])
        if choice == "shift_x":
            k = np.random.randint(-2, 3)
            t = torch.roll(img, shifts=k, dims=2)
        elif choice == "shift_y":
            k = np.random.randint(-2, 3)
            t = torch.roll(img, shifts=k, dims=1)
        elif choice == "rot":
            # cheap 90-degree option; replace with affine rotate if you want milder transforms
            k = np.random.choice([0, 1, 3])  # 0, 90, 270
            t = torch.rot90(img, k=int(k), dims=(1, 2))
        else:
            t = img
        out.append(t)

    return torch.stack(out, dim=0).view(-1, 784)


def get_layer_activations(net, x):
    """
    Uses your existing API: net.predict(x)[1] returns per-layer reps.
    """
    with torch.no_grad():
        _, reps = net.predict(x)
    return reps  # list of [N, num_units]


def compute_unit_thresholds(layer_act, firing_rate=0.01, use_abs=False):
    """
    layer_act: [N, H]
    returns tau: [H]
    """
    a = layer_act.abs() if use_abs else layer_act.clamp(min=0)
    # threshold so each unit fires ~1% of the time
    q = 1.0 - firing_rate
    tau = torch.quantile(a, q=q, dim=0)
    return tau


def invariance_score_for_layer(act_orig, act_trans, tau, use_abs=False):
    """
    act_orig, act_trans: [N, H]
    tau: [H]
    returns:
      per-unit invariance score [H]
      mean layer invariance scalar
    """
    a0 = act_orig.abs() if use_abs else act_orig.clamp(min=0)
    a1 = act_trans.abs() if use_abs else act_trans.clamp(min=0)

    fire0 = a0 > tau.unsqueeze(0)   # [N, H]
    fire1 = a1 > tau.unsqueeze(0)   # [N, H]

    # P(fire on transformed)
    p_fire1 = fire1.float().mean(dim=0).clamp(min=1e-8)

    # P(fire on transformed | fire on original)
    denom = fire0.float().sum(dim=0).clamp(min=1.0)
    p_cond = (fire0 & fire1).float().sum(dim=0) / denom

    # Goodfellow-style normalized score
    score = p_cond / p_fire1
    return score, score.mean().item()


def compute_invariance_scores(net, x_eval, use_abs=False, firing_rate=0.01):
    """
    x_eval: [N, 784]
    returns list of mean invariance scores, one per hidden layer
    """
    x_t = apply_semantic_transforms_mnist(x_eval)

    reps_orig = get_layer_activations(net, x_eval)
    reps_trans = get_layer_activations(net, x_t)

    layer_scores = []
    per_unit_scores = []

    for act0, act1 in zip(reps_orig, reps_trans):
        tau = compute_unit_thresholds(act0, firing_rate=firing_rate, use_abs=use_abs)
        s_unit, s_layer = invariance_score_for_layer(act0, act1, tau, use_abs=use_abs)
        per_unit_scores.append(s_unit.cpu())
        layer_scores.append(s_layer)

    return layer_scores, per_unit_scores