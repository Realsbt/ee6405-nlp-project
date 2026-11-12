"""Separate residual-scale controls; leave the running fusion v1 unchanged."""

import torch
from torch import nn

from experiments.edu_fusion import EDUFusionModel, fusion_config
from experiments.same_window_controls import IndexedSameWindowDataset


SCALE_CONDITIONS = ("scale_fixed", "scale_global", "scale_adaptive")
INITIAL_LOGIT = -2.0


class InputIndependentGate(nn.Module):
    """Return a shared logit; EDUFusion applies the sigmoid afterwards."""

    def __init__(self, learnable):
        super().__init__()
        value = torch.tensor(INITIAL_LOGIT)
        if learnable:
            self.logit = nn.Parameter(value)
        else:
            self.register_buffer("logit", value)

    def forward(self, features):
        return self.logit.expand(features.shape[0], 1)


class ScaleControlModel(EDUFusionModel):
    def __init__(self, config):
        condition = config["condition"]
        if condition not in SCALE_CONDITIONS:
            raise ValueError(f"Unknown scale condition: {condition}")
        # Build the identical encoder, head and attention before replacing gate.
        super().__init__({**config, "condition": "fusion_gated"})
        if condition != "scale_adaptive":
            self.fusion.gate = InputIndependentGate(condition == "scale_global")


def scale_config(condition, seed=3407):
    if condition not in SCALE_CONDITIONS:
        raise ValueError(f"Unknown scale condition: {condition}")
    config = fusion_config("fusion_gated", seed)
    config.update(condition=condition, recipe_id="edu_residual_scale_controls_v1")
    config["fusion"].update(
        scale_mode=condition,
        gate_initial_bias=INITIAL_LOGIT,
        initial_scale=float(torch.sigmoid(torch.tensor(INITIAL_LOGIT))),
    )
    return config


def scale_dataset(cache, split, condition):
    if condition not in SCALE_CONDITIONS:
        raise ValueError(f"Unknown scale condition: {condition}")
    return IndexedSameWindowDataset(cache, split, "text")
