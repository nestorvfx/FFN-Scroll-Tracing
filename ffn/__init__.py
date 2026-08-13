"""3D Flood-Filling Network for papyrus-sheet instance separation.

A GPU-native PyTorch reimplementation of the Januszewski et al. FFN core,
adapted for thin laminar sheets (~3-4 vox thick) in Herculaneum scroll CT.

See DESIGN.md for the full rationale and README.md for usage.
"""

from .config import FFNConfig

__all__ = ["FFNConfig"]
