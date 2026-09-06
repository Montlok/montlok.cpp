# -*- coding: utf-8 -*-

"""Minimal RDT inference package for the Montlok CPU runtime.

Modified from the upstream package entrypoint: training, OCR, vision, and
language-model exports are intentionally absent from this deployment subset.
"""

from Model.config import (
    OMVTConfig,
    RDTConfig,
    TrainingConfig,
    base_config,
    pretrain_config,
    small_config,
    tiny_config,
)
__all__ = [
    "OMVTConfig",
    "RDTConfig",
    "TrainingConfig",
    "base_config",
    "pretrain_config",
    "small_config",
    "tiny_config",
]
