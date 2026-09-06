# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


def _build_freqs(dim: int, theta: float, device=None) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive")
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    idx = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (theta ** (idx / dim))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != cos.shape[-1] or x.shape[-1] != sin.shape[-1]:
        raise ValueError("x/cos/sin last dim mismatch")

    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    return x * cos.to(dtype=x.dtype, device=x.device) + rotate_half(x) * sin.to(
        dtype=x.dtype,
        device=x.device,
    )


class MorphologicalRoPE(nn.Module):
    def __init__(
        self,
        rope_dim: int,
        theta: float = 10000.0,
        max_morph_depth: int = 8,
        use_morphological: bool = True,
        morph_theta_scale: float = 100.0,
    ):
        super().__init__()

        if rope_dim <= 0:
            raise ValueError("rope_dim must be positive")
        if rope_dim % 2 != 0:
            raise ValueError("rope_dim must be even")
        if max_morph_depth <= 0:
            raise ValueError("max_morph_depth must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")
        if morph_theta_scale <= 0:
            raise ValueError("morph_theta_scale must be positive")

        self.rope_dim = rope_dim
        self.theta = theta
        self.max_morph_depth = max_morph_depth
        self.use_morphological = use_morphological
        self.morph_theta_scale = morph_theta_scale

        if use_morphological:
            if (rope_dim // 2) % 2 != 0:
                raise ValueError("rope_dim // 2 must be even in morphological mode")

            self.word_dim = rope_dim // 2
            self.morph_dim = rope_dim // 2

            self.register_buffer(
                "word_freqs",
                _build_freqs(self.word_dim, theta),
                persistent=False,
            )
            self.register_buffer(
                "morph_freqs",
                _build_freqs(self.morph_dim, theta / morph_theta_scale),
                persistent=False,
            )
        else:
            self.word_dim = rope_dim
            self.morph_dim = 0
            self.register_buffer(
                "freqs",
                _build_freqs(rope_dim, theta),
                persistent=False,
            )

    def forward(
        self,
        seq_len: int,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        device=None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if pos_offset < 0:
            raise ValueError("pos_offset must be non-negative")

        if device is None:
            device = self._device()

        if not self.use_morphological or word_pos is None:
            return self._standard(seq_len, device, pos_offset=pos_offset)

        if word_pos.ndim != 2:
            raise ValueError("word_pos must have shape [B, L]")
        if word_pos.shape[1] != seq_len:
            raise ValueError("word_pos length must equal seq_len")

        if morph_depth is None:
            morph_depth = torch.zeros_like(word_pos)
        elif morph_depth.shape != word_pos.shape:
            raise ValueError("morph_depth must have same shape as word_pos")

        word_pos = word_pos.to(device=device, dtype=torch.float32)
        morph_depth = morph_depth.to(device=device, dtype=torch.float32)
        morph_depth = morph_depth.clamp(min=0, max=self.max_morph_depth)

        word = self._angles_to_emb(
            word_pos.unsqueeze(-1) * self.word_freqs.to(device=device),
        )
        morph = self._angles_to_emb(
            morph_depth.unsqueeze(-1) * self.morph_freqs.to(device=device),
        )

        emb = torch.cat([word, morph], dim=-1)
        return emb.cos(), emb.sin()

    def _standard(
        self,
        seq_len: int,
        device,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token-index RoPE. ``pos_offset`` shifts the absolute positions so
        cached incremental decoding sees the same angles as a full forward
        (morphological mode carries absolute positions in ``word_pos`` values
        and needs no offset)."""

        freqs = (
            self.freqs.to(device=device)
            if hasattr(self, "freqs")
            else _build_freqs(self.rope_dim, self.theta, device=device)
        )
        pos = torch.arange(
            pos_offset,
            pos_offset + seq_len,
            dtype=torch.float32,
            device=device,
        )
        angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = self._angles_to_emb(angles)
        return emb.cos(), emb.sin()

    @staticmethod
    def _angles_to_emb(angles: torch.Tensor) -> torch.Tensor:
        return torch.cat([angles, angles], dim=-1)

    def _device(self):
        for buf in self.buffers():
            return buf.device
        return torch.device("cpu")


def derive_morph_info_from_offsets(
    token_offsets: list[tuple[int, int]],
) -> tuple[list[int], list[int]]:
    word_positions: list[int] = []
    morph_depths: list[int] = []

    cur_word = -1
    cur_depth = 0
    prev_end: int | None = None

    for start, end in token_offsets:
        if start < 0 or end < 0:
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            continue

        if prev_end is None or start != prev_end:
            cur_word += 1
            cur_depth = 0
        else:
            cur_depth += 1

        word_positions.append(cur_word)
        morph_depths.append(cur_depth)
        prev_end = end

    return word_positions, morph_depths


def _check() -> None:
    torch.manual_seed(0)

    rope = MorphologicalRoPE(rope_dim=32, use_morphological=False)
    cos, sin = rope(seq_len=10)

    q = torch.randn(2, 8, 10, 32)
    q_rot = apply_rope(q, cos, sin)
    diff = (q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item()

    print("RoPE")
    print(f"  cos: {tuple(cos.shape)}")
    print(f"  apply: {tuple(q.shape)} -> {tuple(q_rot.shape)}")
    print(f"  norm_diff: {diff:.6e}")

    rope_m = MorphologicalRoPE(rope_dim=32, use_morphological=True)
    word_pos = torch.tensor([[0, 0, 0, 1, 1, 2]])
    morph_depth = torch.tensor([[0, 1, 2, 0, 1, 0]])

    cos_m, sin_m = rope_m(
        seq_len=6,
        word_pos=word_pos,
        morph_depth=morph_depth,
    )

    q2 = torch.randn(1, 8, 6, 32)
    q2_rot = apply_rope(q2, cos_m, sin_m)
    diff2 = (q2.norm(dim=-1) - q2_rot.norm(dim=-1)).abs().max().item()

    print("MorphologicalRoPE")
    print(f"  cos: {tuple(cos_m.shape)}")
    print(f"  apply: {tuple(q2.shape)} -> {tuple(q2_rot.shape)}")
    print(f"  norm_diff: {diff2:.6e}")

    offsets = [(0, 3), (3, 5), (5, 7), (8, 11), (11, 12)]
    wp, md = derive_morph_info_from_offsets(offsets)

    print("Offsets")
    print(f"  word_pos: {wp}")
    print(f"  morph_depth: {md}")

    q3 = torch.randn(1, 8, 6, 32, requires_grad=True)
    loss = apply_rope(q3, cos_m, sin_m).sum()
    loss.backward()

    print("Backward")
    print(f"  grad_norm: {q3.grad.norm().item():.6f}")


if __name__ == "__main__":
    _check()
