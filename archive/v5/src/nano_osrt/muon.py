"""Muon optimizer — momentum orthogonalised by Newton-Schulz.

Reference: Keller Jordan's modded-NanoGPT speedrun + the
"MomentUm Orthogonalized by Newton-Schulz" recipe (2024).

Why Muon over Lion/AdamW for transformer hidden weights:
  - The 2D weight matrices in attention (qkv, out_proj) and MoE
    (router, expert SwiGLU projections) have a strong matrix structure
    that scalar adaptive optimizers ignore. Muon takes the SGD-momentum
    update and projects it onto the nearest semi-orthogonal matrix via
    a quintic Newton-Schulz iteration. That update equalises the
    singular spectrum, so under-represented feature directions get the
    same step size as dominant ones — Adam/Lion shrink rare directions
    by their per-parameter variance scaling.
  - Newton-Schulz runs in five matmuls in bf16 (no SVD, no fp32), so
    the FLOP overhead is < 1 % of forward+backward.
  - For sparse MoE in particular, Muon helps cold experts recover
    because each expert's projection sees an orthogonal update rather
    than one biased toward whatever directions already had large
    magnitude.

Use Muon ONLY for 2D matrix parameters of hidden layers
(`Linear.weight`, `nn.Parameter` of shape `(out_dim, in_dim)`).
Embeddings, RMSNorm scales, biases, and 0-D scalars must be optimised
with AdamW (or Lion). The training loop wires this split via
`build_param_groups` below.

Implementation is intentionally minimal: SGD-Nesterov-momentum on the
gradient, Newton-Schulz on the resulting update, scale by
`max(1, rows/cols)**0.5` so the step magnitude matches Adam-scale
expectations (Keller Jordan's heuristic).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

# Newton-Schulz quintic polynomial coefficients (a, b, c) tuned by
# Keller Jordan to maximise the convergence rate of X → orthogonal(X)
# under the iteration:
#
#     A = X X^T
#     X ← a X + (b A + c A²) X
#
# Five iterations is enough for bf16 gradients in practice (matches the
# modded-NanoGPT speedrun). Increasing to ten gives marginally cleaner
# orthogonality at twice the cost.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)
_DEFAULT_NS_STEPS = 5


def newton_schulz5(g: Tensor, steps: int = _DEFAULT_NS_STEPS) -> Tensor:
    """Approximate the polar factor (orthogonal part) of g via NS iteration.

    Operates in bf16 for speed (the iteration is well-conditioned even
    at low precision once g is normalised). Returns a tensor of g's
    original shape and dtype with approximately orthogonal columns
    (or rows, if g is fat).
    """
    if g.ndim != 2:
        raise ValueError(f"newton_schulz5 expects 2D input, got shape {tuple(g.shape)}")
    a, b, c = _NS_COEFFS
    # Cast to bf16 — NS converges fine and bf16 matmuls are ~2× the
    # throughput of fp32 on H100 tensor cores.
    x = g.to(dtype=torch.bfloat16)
    # Normalise so the spectral norm is close to 1; otherwise NS can
    # diverge in the first iteration.
    x = x / (x.norm() + 1e-7)
    # Operate on the squarer side: if rows > cols, transpose so the
    # X X^T product is the smaller of the two possible Gram matrices.
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D matrix parameters.

    Args:
        params: Iterable of parameters or param groups. EVERY parameter
            in this optimizer must be 2D — pass scalars/embeddings/
            norms to a separate AdamW instance.
        lr: Base learning rate. Muon's effective step is much smaller
            than AdamW's per-parameter scale, so the typical Muon LR is
            about 30-50× the AdamW LR (e.g. 0.02 for transformer hidden
            weights vs 6e-4 for AdamW). Tune with a sanity run.
        momentum: SGD momentum coefficient (default 0.95).
        nesterov: Whether to use Nesterov momentum (default True). The
            Muon recipe uses Nesterov by default.
        ns_steps: Number of Newton-Schulz iterations (default 5).
        weight_decay: Decoupled weight decay applied multiplicatively
            to the parameter before the Muon update lands. Default 0.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = _DEFAULT_NS_STEPS,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        # Validate at construction so a wrong param group fails loudly
        # instead of crashing inside the first step().
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only accepts 2D parameters; got "
                        f"shape {tuple(p.shape)}. Use AdamW for "
                        f"embeddings, norms, biases, and scalars."
                    )

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad)
                    state["momentum_buffer"] = buf

                # SGD momentum buffer update — operate in fp32 so the
                # buffer doesn't accumulate bf16 roundoff over millions
                # of steps.
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf

                # Newton-Schulz orthogonalisation of the update.
                ortho = newton_schulz5(update, steps=ns_steps)

                # Shape-aware scale. For fat matrices (rows < cols, e.g.
                # SwiGLU's w_down) we shrink the update so the per-element
                # variance matches what Adam-scale optimisers produce.
                rows, cols = p.shape
                shape_scale = max(1.0, rows / cols) ** 0.5

                # Decoupled weight decay (AdamW-style — applied to the
                # parameter, not added to the gradient).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * shape_scale)

        return loss


class HybridMuonAdamW:
    """Wrapper exposing the standard `torch.optim.Optimizer` surface
    over a Muon + AdamW pair.

    The training loop calls `.zero_grad()`, `.step()`, iterates
    `.param_groups` for LR scheduling, and saves/loads `.state_dict()`.
    Both inner optimisers see the same call sequence; param_groups are
    concatenated so a single `lr` update touches both.

    Save/load uses a single dict with two sub-keys so checkpoints stay
    one-file. If you swap optimiser type mid-run (Lion → Muon), the
    train.py resume-load already wraps in try/except and starts the
    optimiser fresh on mismatch.
    """

    def __init__(
        self,
        muon: Muon,
        adamw: torch.optim.Optimizer,
    ) -> None:
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self) -> list[dict]:
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):  # noqa: ANN001
        # Run Muon first, then AdamW. The order doesn't matter for
        # correctness because the two optimisers touch disjoint params,
        # but Muon-then-AdamW gives a nicer wall-clock profile (Muon's
        # NS iteration is the longer step, so kick it off first).
        self.muon.step()
        return self.adamw.step(closure)

    def state_dict(self) -> dict:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def build_param_groups(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    weight_decay: float,
) -> tuple[list[torch.nn.Parameter], list[dict]]:
    """Split a model's parameters into (muon_params, adamw_groups).

    Routing rules — kept here so the choice is reviewable in one
    place rather than scattered through train.py:

      - 2D parameters that are NOT embeddings → Muon
        (linear weights of attention, MoE router, experts, HRA adapters)
      - nn.Embedding weights → AdamW
        (sparse-ish updates, Muon's orthogonal projection is the wrong
        operator on lookup tables)
      - 1D parameters → AdamW
        (RMSNorm scales, including QK-Norm)
      - 0-D scalars → AdamW
        (moe_gate)
      - Router and loop_embeddings keep wd=0 (matches Lion path).

    AdamW gets two groups (decay vs no-decay) so weight decay only
    touches matrix-style params, not norms/embeddings.
    """
    # Names are needed both to detect embeddings and to apply the
    # router/loop_embedding wd=0 carve-out, so iterate named_parameters.
    muon_params: list[torch.nn.Parameter] = []
    adamw_decay: list[torch.nn.Parameter] = []
    adamw_no_decay: list[torch.nn.Parameter] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        is_embedding = "embedding" in name.lower()
        is_router_like = ("router" in name) or ("loop_embeddings" in name)
        is_norm_or_scalar = param.ndim < 2

        if is_router_like:
            # Router and loop_embeddings: AdamW with wd=0 — matches
            # the Lion config carve-out so the bias controller and
            # routing logits aren't fighting weight decay.
            adamw_no_decay.append(param)
            continue

        if is_norm_or_scalar or is_embedding:
            # AdamW with weight decay applied only to non-embedding
            # 2D params — embeddings are excluded by convention.
            if is_embedding or is_norm_or_scalar:
                adamw_no_decay.append(param)
            else:
                adamw_decay.append(param)
            continue

        if param.ndim == 2:
            muon_params.append(param)
        else:
            # ndim > 2 (e.g. conv filters). Not present in NanoOSRT
            # but flag if a future change adds them.
            raise ValueError(
                f"build_param_groups: parameter '{name}' has ndim "
                f"{param.ndim} which is neither matrix nor norm-like. "
                f"Decide explicitly whether this should go through Muon."
            )

    adamw_groups: list[dict] = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "weight_decay": weight_decay})
    if adamw_no_decay:
        adamw_groups.append({"params": adamw_no_decay, "weight_decay": 0.0})
    return muon_params, adamw_groups
