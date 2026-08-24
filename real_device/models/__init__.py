"""Models shipped with the RTSS 2026 physical-testbed artifact.

The original development package exposed several legacy architectures here.
Those modules are not used by the submitted ResNet-8 workflow and are not part
of the artifact.  Keeping imports limited to the submitted implementation
prevents Python from failing before ``models.resnet8_mmdfl`` can be loaded.
"""

from .resnet8_mmdfl import build_resnet8

__all__ = ["build_resnet8"]
