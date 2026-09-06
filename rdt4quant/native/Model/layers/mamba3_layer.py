# -*- coding: utf-8 -*-

from __future__ import annotations

import inspect
import math
from importlib.metadata import PackageNotFoundError, version as distribution_version

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.config import MIN_OFFICIAL_MAMBA3_D_STATE
from Model.inference.cache import MambaCache
from Model.layers.rmsnorm import GroupedRMSNorm


try:
    from mamba_ssm.modules.mamba3 import Mamba3 as OfficialMamba3
except ImportError:
    OfficialMamba3 = None

try:
    from mamba_ssm.utils.generation import InferenceParams as OfficialInferenceParams
except ImportError:
    OfficialInferenceParams = None

try:
    _OFFICIAL_MAMBA_VERSION = distribution_version("mamba-ssm")
except PackageNotFoundError:
    _OFFICIAL_MAMBA_VERSION = None


_SUPPORTED_OFFICIAL_CACHE_VERSION = "2.3.2.post1"


def official_available() -> bool:
    return OfficialMamba3 is not None


def _validate_official_cfg(cfg) -> None:
    if cfg.mamba_d_state < MIN_OFFICIAL_MAMBA3_D_STATE:
        raise ValueError(
            "official Mamba3 requires mamba_d_state >= "
            f"{MIN_OFFICIAL_MAMBA3_D_STATE}; got {cfg.mamba_d_state}"
        )


def _validate_official_cache_api(mamba: nn.Module, layer_idx: int | None) -> None:
    """Fail closed unless the installed upstream cache API is the audited one."""

    if _OFFICIAL_MAMBA_VERSION != _SUPPORTED_OFFICIAL_CACHE_VERSION:
        raise RuntimeError(
            "official Mamba3 incremental cache requires the audited "
            f"mamba-ssm=={_SUPPORTED_OFFICIAL_CACHE_VERSION}; found "
            f"{_OFFICIAL_MAMBA_VERSION!r}"
        )
    if OfficialInferenceParams is None:
        raise RuntimeError(
            "official Mamba3 incremental cache requires "
            "mamba_ssm.utils.generation.InferenceParams"
        )
    if type(layer_idx) is not int or layer_idx < 0:
        raise RuntimeError(
            "official Mamba3 incremental cache requires a non-negative "
            "integer layer_idx"
        )
    if getattr(mamba, "layer_idx", None) != layer_idx:
        raise RuntimeError("official Mamba3 layer_idx does not match its wrapper")

    required = {
        "forward": {"inference_params"},
        "allocate_inference_cache": {"batch_size", "max_seqlen"},
        "step": {"u", "angle_state", "ssm_state", "k_state", "v_state"},
    }
    for method_name, parameter_names in required.items():
        method = getattr(mamba, method_name, None)
        if not callable(method):
            raise RuntimeError(
                f"official Mamba3 cache API is missing callable {method_name}()"
            )
        try:
            actual = set(inspect.signature(method).parameters)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"cannot inspect official Mamba3 {method_name}() cache API"
            ) from exc
        missing = sorted(parameter_names - actual)
        if missing:
            raise RuntimeError(
                f"official Mamba3 {method_name}() cache API is incompatible; "
                f"missing parameters: {', '.join(missing)}"
            )

    try:
        inference_params = set(
            inspect.signature(OfficialInferenceParams).parameters
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cannot inspect official InferenceParams API") from exc
    missing = {"max_seqlen", "max_batch_size"} - inference_params
    if missing:
        raise RuntimeError(
            "official InferenceParams API is incompatible; missing parameters: "
            + ", ".join(sorted(missing))
        )


def _init_dt_bias(
    nfeatures: int,
    dt_min: float,
    dt_max: float,
    dt_init_floor: float,
) -> torch.Tensor:
    """Upstream Mamba-3 dt_bias initialization.

    Samples ``dt`` uniformly in log space between ``dt_min`` and ``dt_max``,
    floors it, and converts to the additive bias ``dt + log(-expm1(-dt))``
    so that ``softplus(dt_bias) ≈ dt`` at init.
    """

    log_lo = math.log(dt_min)
    log_hi = math.log(dt_max)
    dt = torch.exp(torch.rand(nfeatures) * (log_hi - log_lo) + log_lo)
    dt = torch.clamp(dt, min=dt_init_floor)
    return dt + torch.log(-torch.expm1(-dt))


class NaiveSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        headdim: int = 64,
        d_conv: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_state <= 0:
            raise ValueError("d_state must be positive")
        if expand <= 0:
            raise ValueError("expand must be positive")
        if headdim <= 0:
            raise ValueError("headdim must be positive")
        if d_conv <= 0:
            raise ValueError("d_conv must be positive")

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.headdim = headdim
        self.d_conv = d_conv

        if self.d_inner % headdim != 0:
            raise ValueError("d_model * expand must be divisible by headdim")

        self.nheads = self.d_inner // headdim

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.nheads + 2 * d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.nheads, self.d_inner, bias=True)

        with torch.no_grad():
            self.dt_proj.bias.copy_(
                _init_dt_bias(self.d_inner, dt_min, dt_max, dt_init_floor)
            )
        self.dt_proj.bias._no_weight_decay = True  # type: ignore[attr-defined]

        self.A_log = nn.Parameter(torch.log(torch.rand(self.nheads, d_state) + 0.5))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {x.shape[-1]}")

        if cache is not None:
            if attn_mask is not None and not bool(attn_mask.all()):
                raise ValueError(
                    "NaiveSSM cached decode requires an all-ones attn_mask "
                    "(no padding); pad-free generation only"
                )
            return self._forward_cached(x, cache)

        bsz, seq_len, _ = x.shape
        dtype = x.dtype

        mask = None
        if attn_mask is not None:
            if attn_mask.shape != (bsz, seq_len):
                raise ValueError("attn_mask must have shape [B, L]")
            mask = attn_mask.to(device=x.device, dtype=torch.float32)

        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        u = self.conv1d(u.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        u = F.silu(u)

        if mask is not None:
            u = u * mask.unsqueeze(-1).to(dtype=u.dtype)
            z = z * mask.unsqueeze(-1).to(dtype=z.dtype)

        params = self.x_proj(u)
        dt, b_param, c_param = params.split(
            [self.nheads, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))

        u_h = u.reshape(bsz, seq_len, self.nheads, self.headdim)
        dt_h = dt.reshape(bsz, seq_len, self.nheads, self.headdim)

        a = -torch.exp(self.A_log.float())

        state = torch.zeros(
            bsz,
            self.nheads,
            self.headdim,
            self.d_state,
            device=x.device,
            dtype=torch.float32,
        )

        ys: list[torch.Tensor] = []

        for t in range(seq_len):
            dt_t = dt_h[:, t].float()
            u_t = u_h[:, t].float()
            b_t = b_param[:, t].float().view(bsz, 1, 1, self.d_state)
            c_t = c_param[:, t].float().view(bsz, 1, 1, self.d_state)

            da = torch.exp(dt_t.unsqueeze(-1) * a.view(1, self.nheads, 1, self.d_state))
            bu = dt_t.unsqueeze(-1) * b_t * u_t.unsqueeze(-1)

            next_state = state * da + bu
            if mask is not None:
                active = mask[:, t].view(bsz, 1, 1, 1).bool()
                state = torch.where(active, next_state, state)
            else:
                state = next_state

            y_t = (state * c_t).sum(dim=-1)
            y_t = y_t + self.D.view(1, self.nheads, 1) * u_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y.reshape(bsz, seq_len, self.d_inner).to(dtype)

        if mask is not None:
            y = y * mask.unsqueeze(-1).to(dtype=y.dtype)

        y = y * F.silu(z)

        return self.out_proj(y)

    def _forward_cached(self, x: torch.Tensor, cache) -> torch.Tensor:
        """Bit-exact incremental scan over ``x`` (``[B, m, d_model]``).

        Reuses ``cache.conv_window`` (last ``d_conv - 1`` pre-conv columns) and
        ``cache.ssm_state`` so that processing the prompt in one call followed
        by single-token steps reproduces the full :meth:`forward` scan exactly.
        Both fields are updated in place.
        """

        if not isinstance(cache, MambaCache):
            raise TypeError("NaiveSSM cache must be a MambaCache")
        if cache.backend not in {None, "naive"}:
            raise RuntimeError(
                f"MambaCache is bound to backend={cache.backend!r}, not 'naive'"
            )
        if cache.official_inference_params is not None:
            raise RuntimeError("NaiveSSM cache contains official Mamba state")
        cache.backend = "naive"

        bsz, m, _ = x.shape
        dtype = x.dtype
        k = self.d_conv

        xz = self.in_proj(x)
        u_pre, z = xz.chunk(2, dim=-1)

        u_t = u_pre.transpose(1, 2)  # [B, d_inner, m]

        if k > 1:
            if cache.conv_window is None:
                window = u_t.new_zeros(bsz, self.d_inner, k - 1)
            else:
                window = cache.conv_window
            conv_in = torch.cat([window, u_t], dim=2)  # [B, d_inner, (k-1)+m]
            cache.conv_window = conv_in[..., -(k - 1):]
        else:
            conv_in = u_t

        conv_out = F.conv1d(
            conv_in,
            self.conv1d.weight,
            self.conv1d.bias,
            padding=0,
            groups=self.d_inner,
        )  # [B, d_inner, m]
        u = F.silu(conv_out.transpose(1, 2))  # [B, m, d_inner]

        params = self.x_proj(u)
        dt, b_param, c_param = params.split(
            [self.nheads, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))

        u_h = u.reshape(bsz, m, self.nheads, self.headdim)
        dt_h = dt.reshape(bsz, m, self.nheads, self.headdim)

        a = -torch.exp(self.A_log.float())

        if cache.ssm_state is None:
            state = torch.zeros(
                bsz,
                self.nheads,
                self.headdim,
                self.d_state,
                device=x.device,
                dtype=torch.float32,
            )
        else:
            state = cache.ssm_state

        ys: list[torch.Tensor] = []
        for t in range(m):
            dt_t = dt_h[:, t].float()
            u_tt = u_h[:, t].float()
            b_t = b_param[:, t].float().view(bsz, 1, 1, self.d_state)
            c_t = c_param[:, t].float().view(bsz, 1, 1, self.d_state)

            da = torch.exp(dt_t.unsqueeze(-1) * a.view(1, self.nheads, 1, self.d_state))
            bu = dt_t.unsqueeze(-1) * b_t * u_tt.unsqueeze(-1)

            state = state * da + bu

            y_t = (state * c_t).sum(dim=-1)
            y_t = y_t + self.D.view(1, self.nheads, 1) * u_tt
            ys.append(y_t)

        cache.ssm_state = state

        y = torch.stack(ys, dim=1)
        y = y.reshape(bsz, m, self.d_inner).to(dtype)
        y = y * F.silu(z)

        return self.out_proj(y)


class Mamba3Layer(nn.Module):
    def __init__(
        self,
        cfg,
        layer_idx: int | None = None,
        num_groups: int = 8,
    ):
        super().__init__()

        self.cfg = cfg
        self.layer_idx = layer_idx

        self.norm = GroupedRMSNorm(
            cfg.d_model,
            num_groups=num_groups,
            eps=cfg.rmsnorm_eps,
        )

        use_official = bool(cfg.use_official_mamba and OfficialMamba3 is not None)

        if cfg.use_official_mamba and OfficialMamba3 is None:
            raise RuntimeError(
                "use_official_mamba=True but mamba_ssm is not importable. "
                "Install with `pip install mamba-ssm` (requires CUDA) or set "
                "cfg.use_official_mamba=False to use the NaiveSSM fallback."
            )

        if use_official:
            _validate_official_cfg(cfg)
            self.mamba = self._build_official(cfg, layer_idx)
            self.backend = "official"
        else:
            self.mamba = NaiveSSM(
                d_model=cfg.d_model,
                d_state=cfg.mamba_d_state,
                expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim,
                d_conv=cfg.mamba_d_conv,
                dt_min=cfg.mamba_dt_min,
                dt_max=cfg.mamba_dt_max,
                dt_init_floor=cfg.mamba_dt_init_floor,
            )
            self.backend = "fallback"

    def _build_official(self, cfg, layer_idx: int | None):
        """Build the upstream Mamba-3 module with cfg-aligned kwargs.

        Upstream `Mamba3.__init__` does NOT accept ``d_conv`` (it bakes the
        causal short conv into the fused kernel). We forward the parameters
        it does accept; mismatches will surface as a hard ``TypeError`` so
        we never silently degrade.
        """

        return OfficialMamba3(
            d_model=cfg.d_model,
            d_state=cfg.mamba_d_state,
            expand=cfg.mamba_expand,
            headdim=cfg.mamba_headdim,
            chunk_size=cfg.mamba_chunk_size,
            dt_min=cfg.mamba_dt_min,
            dt_max=cfg.mamba_dt_max,
            dt_init_floor=cfg.mamba_dt_init_floor,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache=None,
        **kwargs,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, d_model]")

        if cache is not None:
            if not isinstance(cache, MambaCache):
                raise TypeError("Mamba3Layer cache must be a MambaCache")
            if attn_mask is not None and not bool(attn_mask.all()):
                raise ValueError(
                    "cached Mamba decode requires an all-ones attn_mask "
                    "(no padding); pad-free generation only"
                )
            residual = x
            mamba_input = self.norm(residual)
            if isinstance(self.mamba, NaiveSSM):
                y = self.mamba(mamba_input, attn_mask=attn_mask, cache=cache)
            else:
                y = self._forward_official_cached(mamba_input, cache)
            return residual + y

        mask = None
        if attn_mask is not None:
            if attn_mask.shape != x.shape[:2]:
                raise ValueError("attn_mask must have shape [B, L]")
            if (
                not isinstance(self.mamba, NaiveSSM)
                and not bool(attn_mask.all())
                and not _is_right_padding_mask(attn_mask)
            ):
                raise ValueError(
                    "official Mamba backend only supports all-ones or right-padded "
                    "attention masks; use fallback Mamba for left/interior padding"
                )
            mask = attn_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

        residual = x if mask is None else x * mask
        mamba_input = self.norm(residual)

        if isinstance(self.mamba, NaiveSSM):
            y = self.mamba(mamba_input, attn_mask=attn_mask)
        else:
            y = self.mamba(mamba_input)
            if mask is not None:
                y = y * mask

        out = residual + y
        if mask is not None:
            out = out * mask
        return out

    def _forward_official_cached(
        self,
        x: torch.Tensor,
        cache: MambaCache,
    ) -> torch.Tensor:
        """Run one isolated upstream prefill followed by single-token steps."""

        if cache.official_poisoned:
            raise RuntimeError(
                "official MambaCache is poisoned by a failed step; start a new "
                "DecodeCache"
            )
        if cache.backend not in {None, "official"}:
            raise RuntimeError(
                f"MambaCache is bound to backend={cache.backend!r}, not 'official'"
            )
        if cache.conv_window is not None or cache.ssm_state is not None:
            raise RuntimeError("official MambaCache contains NaiveSSM state")

        bsz, chunk_len, _ = x.shape
        max_seqlen = int(self.cfg.max_seq_len)
        if max_seqlen <= 0:
            raise RuntimeError("official Mamba cache max_seq_len must be positive")
        if chunk_len <= 0:
            raise ValueError("official Mamba cache cannot process an empty chunk")

        if cache.official_inference_params is None:
            _validate_official_cache_api(self.mamba, self.layer_idx)
            if cache.backend is not None or cache.official_owner is not None:
                raise RuntimeError("official MambaCache is only partially initialized")
            if chunk_len > max_seqlen:
                raise ValueError(
                    "official Mamba prefill exceeds max_seq_len: "
                    f"{chunk_len} > {max_seqlen}"
                )
            return self._official_prefill(
                x,
                cache,
                batch_size=bsz,
                max_seqlen=max_seqlen,
            )

        self._validate_bound_official_cache(
            x,
            cache,
            batch_size=bsz,
            max_seqlen=max_seqlen,
        )
        if chunk_len != 1:
            raise ValueError(
                "official Mamba cache accepts exactly one token after prefill; "
                f"got chunk length {chunk_len}"
            )

        inference_params = cache.official_inference_params
        offset = int(inference_params.seqlen_offset)
        if offset + 1 > max_seqlen:
            raise ValueError(
                "official Mamba cache would exceed max_seq_len: "
                f"{offset} + 1 > {max_seqlen}"
            )
        states = self._official_states(inference_params)

        try:
            result = self.mamba.step(x[:, 0, :], *states)
            if not isinstance(result, tuple) or len(result) != 5:
                raise RuntimeError(
                    "official Mamba3 step() must return "
                    "(out, angle_state, ssm_state, k_state, v_state)"
                )
            out = result[0]
            next_states = tuple(result[1:])
            self._validate_official_step_result(out, states, next_states, x)
        except Exception:
            cache.official_poisoned = True
            raise

        inference_params.key_value_memory_dict[self.layer_idx] = next_states
        inference_params.seqlen_offset = offset + 1
        return out.unsqueeze(1)

    def _official_prefill(
        self,
        x: torch.Tensor,
        cache: MambaCache,
        *,
        batch_size: int,
        max_seqlen: int,
    ) -> torch.Tensor:
        states = self.mamba.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            device=x.device,
            dtype=x.dtype,
        )
        if not isinstance(states, tuple) or len(states) != 4:
            raise RuntimeError(
                "official Mamba3 allocate_inference_cache() must return four states"
            )
        if not all(isinstance(state, torch.Tensor) for state in states):
            raise RuntimeError("official Mamba3 cache states must be tensors")

        inference_params = OfficialInferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            key_value_memory_dict={self.layer_idx: states},
        )
        out = self.mamba(x, inference_params=inference_params)
        if not isinstance(out, torch.Tensor) or out.shape != x.shape:
            raise RuntimeError(
                "official Mamba3 prefill returned an incompatible output shape"
            )
        if int(inference_params.seqlen_offset) != 0:
            raise RuntimeError(
                "official Mamba3 mutated seqlen_offset during prefill"
            )
        self._official_states(inference_params)

        inference_params.seqlen_offset = x.shape[1]
        cache.backend = "official"
        cache.official_inference_params = inference_params
        cache.official_owner = self.mamba
        cache.official_batch_size = batch_size
        cache.official_max_seqlen = max_seqlen
        cache.official_device = x.device
        cache.official_dtype = x.dtype
        return out

    def _validate_bound_official_cache(
        self,
        x: torch.Tensor,
        cache: MambaCache,
        *,
        batch_size: int,
        max_seqlen: int,
    ) -> None:
        inference_params = cache.official_inference_params
        if cache.backend != "official":
            raise RuntimeError("official MambaCache backend binding is invalid")
        if cache.official_owner is not self.mamba:
            raise RuntimeError(
                "official MambaCache belongs to a different Mamba call site"
            )
        if cache.official_batch_size != batch_size:
            raise ValueError(
                "official Mamba cache batch size is fixed after prefill: "
                f"expected {cache.official_batch_size}, got {batch_size}"
            )
        if cache.official_max_seqlen != max_seqlen:
            raise RuntimeError("official Mamba cache max_seq_len changed within a run")
        if cache.official_device != x.device or cache.official_dtype != x.dtype:
            raise ValueError(
                "official Mamba cache device and dtype are fixed after prefill"
            )
        if not isinstance(inference_params, OfficialInferenceParams):
            raise RuntimeError("official MambaCache has incompatible InferenceParams")
        if int(inference_params.max_batch_size) != batch_size:
            raise RuntimeError("official InferenceParams max_batch_size changed")
        if int(inference_params.max_seqlen) != max_seqlen:
            raise RuntimeError("official InferenceParams max_seqlen changed")
        offset = int(inference_params.seqlen_offset)
        if offset <= 0 or offset > max_seqlen:
            raise RuntimeError("official InferenceParams seqlen_offset is invalid")

    def _official_states(self, inference_params) -> tuple[torch.Tensor, ...]:
        state_dict = getattr(inference_params, "key_value_memory_dict", None)
        if not isinstance(state_dict, dict) or set(state_dict) != {self.layer_idx}:
            raise RuntimeError(
                "official InferenceParams must contain exactly one call-site state"
            )
        states = state_dict[self.layer_idx]
        if not isinstance(states, tuple) or len(states) != 4:
            raise RuntimeError("official Mamba3 call-site state must contain four tensors")
        if not all(isinstance(state, torch.Tensor) for state in states):
            raise RuntimeError("official Mamba3 call-site states must be tensors")
        return states

    @staticmethod
    def _validate_official_step_result(
        out: object,
        previous_states: tuple[torch.Tensor, ...],
        next_states: tuple[object, ...],
        x: torch.Tensor,
    ) -> None:
        if not isinstance(out, torch.Tensor) or out.shape != (x.shape[0], x.shape[2]):
            raise RuntimeError("official Mamba3 step() returned an incompatible output")
        if len(next_states) != len(previous_states) or not all(
            isinstance(state, torch.Tensor) for state in next_states
        ):
            raise RuntimeError("official Mamba3 step() returned incompatible states")
        for previous, current in zip(previous_states, next_states):
            if current.shape != previous.shape:
                raise RuntimeError("official Mamba3 step() changed a state shape")


def _is_right_padding_mask(attn_mask: torch.Tensor) -> bool:
    if attn_mask.ndim != 2:
        return False
    mask = attn_mask.to(dtype=torch.long)
    if mask.numel() == 0:
        return True
    if not bool(((mask == 0) | (mask == 1)).all()):
        return False
    return bool((mask[:, 1:] <= mask[:, :-1]).all())


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    layer = Mamba3Layer(cfg, layer_idx=0)
    layer.eval()

    print("Mamba3Layer")
    print(f"  backend: {layer.backend}")
    print(f"  official_available: {official_available()}")

    bsz, seq_len = 2, 16
    x = torch.randn(bsz, seq_len, cfg.d_model)

    y = layer(x)

    print(f"  shape: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  params: {sum(p.numel() for p in layer.parameters()):,}")

    x2 = torch.randn(bsz, seq_len, cfg.d_model, requires_grad=True)
    loss = layer(x2).sum()
    loss.backward()

    print(f"  grad_norm: {x2.grad.norm().item():.6f}")

    x3 = torch.randn(1, 8, cfg.d_model)
    out_full = layer(x3)

    x3_mod = x3.clone()
    x3_mod[0, 5:] += 10.0
    out_mod = layer(x3_mod)

    diff = (out_full[0, :5] - out_mod[0, :5]).abs().max().item()
    print(f"  causal_diff_before_changed_span: {diff:.6e}")

    ignored = layer(
        x,
        word_pos=torch.zeros(bsz, seq_len, dtype=torch.long),
        morph_depth=torch.zeros(bsz, seq_len, dtype=torch.long),
        attn_mask=torch.ones(bsz, seq_len, dtype=torch.long),
    )

    print(f"  kwargs_ignored_shape: {tuple(ignored.shape)}")


if __name__ == "__main__":
    _check()
