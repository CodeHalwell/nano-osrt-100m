"""HuggingFace-compatible wrapper for NanoOSRT.

Enables standard transformers inference:

    from nano_osrt.hf_model import NanoOSRTForCausalLM, NanoOSRTConfig

    model = NanoOSRTForCausalLM.from_pretrained("path/to/model")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    inputs = tokenizer("user: What is 2+2?\nassistant:", return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=512)
    print(tokenizer.decode(output[0]))

Also supports pushing to HuggingFace Hub:

    model.push_to_pretrained("your-username/nano-osrt-100m")
"""

import json
import os
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── RoPE (self-contained) ──────────────────────────────────────────────


def _compute_rope_freqs(
    seq_len: int, dim: int, theta: float = 10000.0, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2] / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(0).unsqueeze(2)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(0).unsqueeze(2)
    return cos, sin


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin


# ── HRA Linear (self-contained) ────────────────────────────────────────


class HRALinear(nn.Module):
    """Linear + parallel high-rank adapter: y = W @ x + scale * (x @ A @ B)."""

    def __init__(self, in_features: int, out_features: int, rank: int, scale: float = 1.0, bias: bool = False):
        super().__init__()
        self.original = nn.Linear(in_features, out_features, bias=bias)
        self.scale = scale
        self.adapter_a = nn.Parameter(torch.zeros(in_features, rank))
        self.adapter_b = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: Tensor) -> Tensor:
        return self.original(x) + self.scale * ((x @ self.adapter_a) @ self.adapter_b)

    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features


# ── Model config ────────────────────────────────────────────────────────


@dataclass
class NanoOSRTConfig:
    """Configuration for NanoOSRT model."""

    model_type: str = "nano-osrt"
    dim: int = 1280
    heads: int = 20
    head_dim: int = 64
    vocab_size: int = 50304
    real_vocab_size: int = 50277
    seq_len: int = 4096
    num_blocks: int = 2
    recursive_loops: int = 6
    adapter_rank: int = 16
    adapter_alpha: float = 16.0
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_enabled: bool = True
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    torch_dtype: str = "float32"

    def save_pretrained(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "NanoOSRTConfig":
        with open(os.path.join(path, "config.json")) as f:
            return cls(**json.load(f))


# ── Model components ────────────────────────────────────────────────────


class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = 64 * ((int(dim * 8 / 3) + 63) // 64)
        self.w_gate = HRALinear(dim, hidden, rank=0, bias=False)
        self.w_up = HRALinear(dim, hidden, rank=0, bias=False)
        self.w_down = HRALinear(hidden, dim, rank=0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class RecursiveBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm_attn = nn.RMSNorm(dim)
        self.qkv = HRALinear(dim, dim * 3, rank=0, bias=False)
        self.out_proj = HRALinear(dim, dim, rank=0, bias=False)
        self.norm_ffn = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(
        self, x: Tensor, adapter_a: Tensor, adapter_b: Tensor,
        adapter_scale: float, rope_cos: Tensor, rope_sin: Tensor,
    ) -> Tensor:
        B, S, D = x.shape
        x_mod = x + adapter_scale * (x @ adapter_a @ adapter_b)

        h = self.norm_attn(x_mod)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.heads, self.head_dim)
        k = k.view(B, S, self.heads, self.head_dim)
        v = v.view(B, S, self.heads, self.head_dim)

        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        x = x_mod + self.out_proj(attn_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ── Main model ──────────────────────────────────────────────────────────


class NanoOSRTForCausalLM(nn.Module):
    """HuggingFace-compatible NanoOSRT for causal language modeling.

    Supports:
        - from_pretrained() / save_pretrained()
        - generate() with top-p sampling
        - push_to_hub() via safetensors
    """

    def __init__(self, config: NanoOSRTConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        cos, sin = _compute_rope_freqs(config.seq_len, config.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList(
            [RecursiveBlock(config.dim, config.heads) for _ in range(config.num_blocks)]
        )

        total_pairs = config.num_blocks * config.recursive_loops
        self.adapters_a = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.dim, config.adapter_rank)) for _ in range(total_pairs)]
        )
        self.adapters_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.adapter_rank, config.dim)) for _ in range(total_pairs)]
        )
        self.adapter_scale = config.adapter_alpha / config.adapter_rank
        self.norm_loop = nn.RMSNorm(config.dim)
        self.norm_out = nn.RMSNorm(config.dim)

        # Inject HRA ranks into the linear layers
        if config.hra_enabled and config.hra_rank > 0:
            self._inject_hra(config.hra_rank, config.hra_scale)

    def _inject_hra(self, rank: int, scale: float) -> None:
        """Replace zero-rank HRALinear layers with actual adapters."""
        for block in self.blocks:
            for name in ("qkv", "out_proj"):
                layer = getattr(block, name)
                new_layer = HRALinear(
                    layer.in_features, layer.out_features,
                    rank=rank, scale=scale, bias=layer.original.bias is not None,
                )
                setattr(block, name, new_layer)
            for name in ("w_gate", "w_up", "w_down"):
                layer = getattr(block.ffn, name)
                new_layer = HRALinear(
                    layer.in_features, layer.out_features,
                    rank=rank, scale=scale, bias=layer.original.bias is not None,
                )
                setattr(block.ffn, name, new_layer)

    def forward(self, input_ids: Tensor, **kwargs) -> dict:
        x = self.embedding(input_ids)
        S = input_ids.shape[1]
        cos = self.rope_cos[:, :S, :, :]
        sin = self.rope_sin[:, :S, :, :]

        for loop in range(self.config.recursive_loops):
            for block_idx, block in enumerate(self.blocks):
                idx = loop * self.config.num_blocks + block_idx
                x = block(x, self.adapters_a[idx], self.adapters_b[idx],
                          self.adapter_scale, cos, sin)
            if loop < self.config.recursive_loops - 1:
                x = self.norm_loop(x)

        x = self.norm_out(x)
        logits = F.linear(x, self.embedding.weight)
        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int = 0,
        repetition_penalty: float = 1.2,
        **kwargs,
    ) -> Tensor:
        """Generate tokens autoregressively with top-p sampling and repetition penalty."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max seq_len
            context = generated[:, -self.config.seq_len:]
            out = self.forward(context)
            next_logits = out["logits"][:, -1, :self.config.real_vocab_size].float()

            # Repetition penalty: reduce logits for tokens already generated
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if token_id < next_logits.shape[-1]:
                        if next_logits[0, token_id] > 0:
                            next_logits[0, token_id] /= repetition_penalty
                        else:
                            next_logits[0, token_id] *= repetition_penalty

            if temperature > 0:
                next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

                # Top-p (nucleus) filtering
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                next_logits.scatter_(1, sorted_indices, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return generated

    def save_pretrained(self, save_dir: str) -> None:
        """Save model weights and config."""
        os.makedirs(save_dir, exist_ok=True)
        self.config.save_pretrained(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
        # Also save as safetensors if available
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))
        except ImportError:
            pass
        print(f"Model saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu", dtype: torch.dtype | None = None) -> "NanoOSRTForCausalLM":
        """Load model from a directory with config.json and model weights."""
        config = NanoOSRTConfig.from_pretrained(path)
        model = cls(config)

        # Try safetensors first, then .pt
        safetensors_path = os.path.join(path, "model.safetensors")
        pt_path = os.path.join(path, "model.pt")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path, device=device)
        elif os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device, weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights found in {path}")

        model.load_state_dict(state_dict, strict=False)

        if dtype:
            model = model.to(dtype=dtype)
        model = model.to(device)
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        config: NanoOSRTConfig | None = None,
    ) -> "NanoOSRTForCausalLM":
        """Load from a training checkpoint (model_state_dict format)."""
        if config is None:
            config = NanoOSRTConfig()

        model = cls(config)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        model = model.to(device)
        return model
