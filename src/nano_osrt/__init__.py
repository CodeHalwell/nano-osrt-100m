"""nano-osrt: recursive Mixtral-style MoE transformer (~363M physical params).

Top-level imports expose the current architecture. Earlier versions (v1/v2/v3
dense recursive, v4 MoE with dense FFN crutch) are preserved under
`archive/` for reference but not importable here.
"""

from nano_osrt.config import NanoOSRTConfig
from nano_osrt.model import (
    ExpertFFN,
    MoELayer,
    NanoOSRTForCausalLM,
    NanoOSRTModel,
    NanoOSRTPreTrainedModel,
    RecursiveBlock,
    orthogonal_expert_init,
)

__all__ = [
    "ExpertFFN",
    "MoELayer",
    "NanoOSRTConfig",
    "NanoOSRTForCausalLM",
    "NanoOSRTModel",
    "NanoOSRTPreTrainedModel",
    "RecursiveBlock",
    "orthogonal_expert_init",
]
__version__ = "0.5.0"
