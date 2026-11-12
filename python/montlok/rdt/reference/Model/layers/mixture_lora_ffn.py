# -*- coding: utf-8 -*-

"""Step-aware Mixture-of-LoRAs SwiGLU for the recurrent depth core.

This is a drop-in replacement for :class:`Model.layers.swiglu.SwiGLU` that adds
*breadth* to the recurrent loop without paying the depth cost. A single shared
base SwiGLU is kept (the parameter-reuse dividend of recurrence), and ``K``
low-rank LoRA experts modulate the gate/up projection per token. A router
(Manifold-Hyper-Connection style: ``bias + alpha*tanh(proj(rms_norm(x)))``)
mixes the experts per token, optionally conditioned on the **recurrent step
index** -- the signal that only the RDT loop has, so the *same* token is routed
to *different* experts at *different* depths (Mixture-of-Recursions / Mixture-of-
LoRAs, arXiv:2507.10524 / 2512.12880), traversing the expert pool along the time
axis of the recurrence.

Cold-start bit-compatibility (the keystone property): with ``lora_b=0``,
``router_alpha=0`` and ``step_embed=0`` the expert delta is exactly zero and the
module reduces to the plain base SwiGLU. So ``use_mol=True`` is byte-identical to
``use_mol=False`` at init, and existing smoke configs are unaffected.

Two design invariants other code depends on -- do not break them without
re-checking the call sites:

* The base down projection is named ``w_down`` so that
  :meth:`RDTForCausalLM._scale_residual_projections` matches it via the
  ``ffn.w_down`` suffix and scales the residual write exactly as for a plain
  SwiGLU. **Do not rename it.**
* LoRA modulates ``w_in`` (upstream of the nonlinearity), NOT ``w_down``. It is
  stored as ``nn.Parameter`` fused tensors (not ``nn.Linear``), so the
  ``isinstance(module, nn.Linear)`` filter in ``_scale_residual_projections``
  never touches it and no LoRA-awareness is needed there. **Do not convert the
  experts to nn.Linear / move them onto w_down** without revisiting residual
  scaling.

``step_embed``/``router_bias``/``router_alpha`` are plain ``nn.Parameter`` (not
``nn.Embedding``) precisely so the model-level ``apply(_init_weights)`` -- which
re-inits every ``nn.Linear``/``nn.Embedding`` -- leaves their zero init intact.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.layers.rmsnorm import RMSNorm


class MixtureLoRAFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        n_experts: int,
        rank: int,
        step_table: int,
        *,
        step_aware: bool = True,
        router_temp: float = 1.0,
        top_k: int = 0,
        init_std: float = 0.02,
        rmsnorm_eps: float = 1e-5,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if ffn_hidden <= 0:
            raise ValueError("ffn_hidden must be positive")
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if step_table <= 0:
            raise ValueError("step_table must be positive")
        if router_temp <= 0:
            raise ValueError("router_temp must be positive")
        if not (0 <= top_k <= n_experts):
            raise ValueError("top_k must be in [0, n_experts]")

        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.n_experts = n_experts
        self.rank = rank
        self.step_table = step_table
        self.step_aware = step_aware
        self.router_temp = float(router_temp)
        self.top_k = top_k
        self.init_std = init_std

        # Shared base SwiGLU -- identical layout/names to ``SwiGLU`` so residual
        # scaling and weight loading stay compatible.
        self.w_in = nn.Linear(d_model, 2 * ffn_hidden, bias=bias)
        self.w_down = nn.Linear(ffn_hidden, d_model, bias=bias)

        # K LoRA experts on the gate/up projection (w_in), fused for a batched
        # GEMM (no Python loop over experts).
        self.lora_a = nn.Parameter(torch.empty(n_experts, d_model, rank))
        self.lora_b = nn.Parameter(torch.empty(n_experts, rank, 2 * ffn_hidden))

        # Per-token router (causal: no cross-position mixing).
        self.router_norm = RMSNorm(d_model, eps=rmsnorm_eps)
        self.router_proj = nn.Linear(d_model, n_experts, bias=False)
        self.router_bias = nn.Parameter(torch.zeros(n_experts))
        self.router_alpha = nn.Parameter(torch.zeros(()))

        # Recurrent step embedding: the RDT-intrinsic conditioning signal.
        if step_aware:
            self.step_embed = nn.Parameter(torch.zeros(step_table, d_model))
        else:
            self.register_parameter("step_embed", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # LoRA-A nonzero (standard), LoRA-B zero -> expert delta is exactly 0 at
        # init regardless of routing, the cold-start guarantee. router_alpha and
        # step_embed stay zero (constructed as zeros) so the router has no effect
        # at init either. w_in/w_down/router_proj are left for the model-level
        # apply(_init_weights) to initialize like any nn.Linear.
        nn.init.normal_(self.lora_a, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.lora_b)
        # Exempt the routing scalars/table from weight decay. router_bias (1-D)
        # and router_alpha (0-D) are auto-exempt by the optimizer's ndim<=1 rule,
        # but step_embed (2-D) needs the explicit marker; tag all three for
        # clarity. LoRA-A/B intentionally keep weight decay (B starts at 0).
        self.router_bias._no_weight_decay = True
        self.router_alpha._no_weight_decay = True
        if self.step_embed is not None:
            self.step_embed._no_weight_decay = True

    @staticmethod
    def _topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
        # Keep the top-k logits per token, set the rest to -inf so softmax zeroes
        # them. Ties at the boundary keep all tied entries (>= threshold).
        threshold = logits.topk(k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def forward(self, x: torch.Tensor, step: int | None = None) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {x.shape[-1]}")

        # Base pre-activation.
        h_in = self.w_in(x)  # [B, L, 2F]

        # Router logits (per token, optionally step-conditioned).
        ref = self.router_norm(x)
        if self.step_aware and step is not None:
            idx = min(max(int(step), 0), self.step_table - 1)
            ref = ref + self.step_embed[idx]
        logits = self.router_bias + self.router_alpha * torch.tanh(self.router_proj(ref))
        logits = logits / self.router_temp
        if 0 < self.top_k < self.n_experts:
            logits = self._topk_mask(logits, self.top_k)
        weights = torch.softmax(logits, dim=-1)  # [B, L, K]

        # Per-token mixed LoRA delta on the w_in pre-activation:
        # expert e contributes (x @ A_e) @ B_e, then mix by router weights.
        xa = torch.einsum("bld,edr->bler", x, self.lora_a)  # [B, L, K, r]
        delta = torch.einsum("bler,erf->blef", xa, self.lora_b)  # [B, L, K, 2F]
        delta = torch.einsum("ble,blef->blf", weights, delta)  # [B, L, 2F]

        gate, up = (h_in + delta).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


def _check() -> None:
    torch.manual_seed(0)

    from Model.layers.swiglu import SwiGLU

    d, F_hidden, K, r, table = 512, 1536, 4, 8, 4
    x = torch.randn(2, 16, d)

    base = SwiGLU(d, F_hidden)
    mol = MixtureLoRAFFN(d, F_hidden, K, r, table)
    # Copy base weights so the only difference is the (zero) LoRA path.
    mol.w_in.load_state_dict(base.w_in.state_dict())
    mol.w_down.load_state_dict(base.w_down.state_dict())

    # Cold-start: lora_b=0, router_alpha=0, step_embed=0 -> identical to base.
    with torch.no_grad():
        mol.router_alpha.fill_(2.0)  # even with a hot router...
        mol.step_embed.normal_()  # ...and a random step table...
        # ...lora_b is still zero, so the delta is exactly zero.
        y_mol = mol(x, step=1)
        y_base = base(x)
    max_diff = (y_mol - y_base).abs().max().item()

    print("MixtureLoRAFFN")
    print(f"  shape: {tuple(x.shape)} -> {tuple(y_mol.shape)}")
    print(f"  cold-start max|mol-base| (lora_b=0): {max_diff:.2e}")
    print(f"  params: {sum(p.numel() for p in mol.parameters()):,}")

    # Step-aware routing actually changes the mix across steps.
    with torch.no_grad():
        mol.lora_b.normal_(std=0.02)
        w0 = torch.softmax(
            mol.router_bias
            + mol.router_alpha * torch.tanh(mol.router_proj(mol.router_norm(x) + mol.step_embed[0])),
            dim=-1,
        )
        w1 = torch.softmax(
            mol.router_bias
            + mol.router_alpha * torch.tanh(mol.router_proj(mol.router_norm(x) + mol.step_embed[1])),
            dim=-1,
        )
    print(f"  step0 vs step1 router differ: {not torch.allclose(w0, w1)}")

    x2 = torch.randn(2, 16, d, requires_grad=True)
    mol(x2, step=2).sum().backward()
    print(f"  grad_norm: {x2.grad.norm().item():.6f}")
    print(f"  lora_a.grad finite: {bool(torch.isfinite(mol.lora_a.grad).all())}")


if __name__ == "__main__":
    _check()
