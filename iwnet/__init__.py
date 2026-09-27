"""IWNET Grape Leaf Disease Classifier.

A research pipeline (dataset build, leakage audit, training, evaluation) plus a
separate inference/API application layer, both built around the same IWNET
architecture: EfficientNet-B3 multi-level features, Channel Attention, Spatial
Attention, IW Attention, adaptive scale weighting and multi-scale fusion.
"""

__version__ = "1.0.0"

from iwnet.config import BACKBONE, FINAL_CLASSES, Config, load_config

__all__ = ["__version__", "BACKBONE", "FINAL_CLASSES", "Config", "load_config"]
