"""nano-osrt-100m: A nano-scale Open-Set Reasoning Transformer (~100M parameters)."""

from nano_osrt.config import ModelConfig
from nano_osrt.model import NanoOSRT

__all__ = ["ModelConfig", "NanoOSRT"]
__version__ = "0.1.0"

# The recursive-block variant and Modal helpers are importable via their
# fully-qualified module paths:
#   from nano_osrt.modal_config import ModalConfig
#   from nano_osrt.recursive_model import RecursiveNanoOSRT
#   from nano_osrt.rope import apply_rope, compute_rope_freqs
#   from nano_osrt.modal_data import TokenStream, make_loader
#   from nano_osrt.modal_train import run_training
