import torch.nn as nn
from lop.algos.cbp_linear import CBPLinear 

def _patch_sequential(seq: nn.Sequential, cbp_kwargs) -> nn.Sequential:
    modules = list(seq.children())
    new_modules = []
    last_linear = None
    for m in modules:
        if isinstance(m, nn.Linear):
            if last_linear is not None:
                new_modules.append(CBPLinear(in_layer=last_linear, out_layer=m, **cbp_kwargs))
            new_modules.append(m)
            last_linear = m
        else:
            new_modules.append(m)
    return nn.Sequential(*new_modules)

def add_cbp_to_model(model: nn.Module, **cbp_kwargs) -> int:
    patched = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Sequential):
                setattr(parent, name, _patch_sequential(child, cbp_kwargs))
                patched += 1
    return patched