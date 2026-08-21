"""3D neighborhood attention + its RoPE, for the LTX-2.5 DiffVAE decoder.

Ported from the reference's *eager* fallback
(`ltx_core/model/video_vae/transformer/fallback_na/eager.py`) and
`transformer/rope_math.py` — deliberately NOT the natten / Triton / CuTe paths,
none of which exist on Metal.

## The one fact that makes this simple

The reference's window is **shift-clamped**, never truncated
(`eager.py:_window_bounds`)::

    K     = min(kernel, length)      # kernel clamps to a short axis
    start = clamp(i - K // 2, 0, length - K)
    end   = start + K                # ALWAYS a full-size window

so every query attends to exactly ``prod(kernel)`` keys, all in bounds.
Therefore **a gather formulation needs no attention mask at all**. The additive
mask in the reference exists only because its tiled formulation shares one
contiguous key box across a block of queries with slightly different windows —
a memory device, not semantics.

Softmax is permutation-invariant over keys, so the within-window key order is
irrelevant as long as K and V are gathered in the *same* order.

## Traps this module encodes

* RoPE pairing is **GPT-J interleaved** (adjacent element pairs), not
  NeoX split-half (`rope_math.py:50`). Split-half produces plausible-looking
  garbage.
* The three axes use **different frequency ladders**: T over ``d_t`` dims,
  H and W over ``d_hw`` each, with ``d_t = (head_dim // 4) // 2 * 2`` and
  ``d_hw = (head_dim - d_t) // 2`` (`rope_math.py:19-29`) — for head_dim 64
  that is (16, 24, 24).
* Positions are plain 0-based **absolute** coordinates of the volume, not
  window-relative offsets (`det_attn_rope.py:1-13`).
* RoPE runs in **fp32** then casts back (`attention.py:118`, `rope_math.py:61`).
* The ``head_dim ** -0.5`` scale is folded into Q *before* the NA call, so the
  attention itself uses ``scale = 1.0`` (`combined/attn.py:78`).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """(d_t, d_h, d_w) — reference ``rope_math.default_rope_dim_split``."""
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    return d_t, d_hw, d_hw


def rope_inv_freqs(dim: int, base: float = 10000.0) -> mx.array:
    """``1 / base ** (2j / dim)`` for j in [0, dim/2) — computed in fp64 upstream.

    numpy/fp64 is used for the exponent to match the reference's float64
    construction before its float32 cast (`rope_math.py:32-37`).
    """
    import numpy as np

    j = np.arange(0, dim, 2, dtype=np.float64)
    return mx.array((1.0 / (base ** (j / dim))).astype(np.float32))


def _rotate_axis(x: mx.array, positions: mx.array, inv: mx.array) -> mx.array:
    """Rotate the last axis of ``x`` by ``positions`` (GPT-J interleaved pairs).

    ``x``: (..., A, ..., D) where the axis being rotated is broadcast-compatible
    with ``positions``; here callers pass x shaped (B, T, H, W, NH, D_axis) and
    positions shaped so it broadcasts against the matching spatial axis.
    """
    xf = x.astype(mx.float32)
    *lead, d = xf.shape
    pairs = xf.reshape(*lead, d // 2, 2)
    xe, xo = pairs[..., 0], pairs[..., 1]

    ang = positions.astype(mx.float32)[..., None] * inv  # (..., d/2)
    c, s = mx.cos(ang), mx.sin(ang)
    re = xe * c - xo * s
    ro = xe * s + xo * c
    out = mx.stack([re, ro], axis=-1).reshape(*lead, d)
    return out.astype(x.dtype)


def apply_rope_3d(
    x: mx.array,
    inv_t: mx.array,
    inv_h: mx.array,
    inv_w: mx.array,
    dim_split: tuple[int, int, int],
) -> mx.array:
    """Apply per-axis RoPE to ``x`` of shape (B, T, H, W, NH, HD).

    Head-dim is partitioned [0:d_t) by T, [d_t:d_t+d_h) by H, the rest by W
    (`combined/attn.py:35-43`), each rotated at its own absolute coordinate.
    """
    B, T, H, W, NH, HD = x.shape
    d_t, d_h, d_w = dim_split
    assert d_t + d_h + d_w == HD, f"rope split {dim_split} != head_dim {HD}"

    xt = x[..., :d_t]
    xh = x[..., d_t : d_t + d_h]
    xw = x[..., d_t + d_h :]

    # positions broadcast against (B, T, H, W, NH, d/2)
    pt = mx.arange(T, dtype=mx.float32).reshape(1, T, 1, 1, 1)
    ph = mx.arange(H, dtype=mx.float32).reshape(1, 1, H, 1, 1)
    pw = mx.arange(W, dtype=mx.float32).reshape(1, 1, 1, W, 1)

    return mx.concatenate(
        [
            _rotate_axis(xt, pt, inv_t),
            _rotate_axis(xh, ph, inv_h),
            _rotate_axis(xw, pw, inv_w),
        ],
        axis=-1,
    )


def window_starts(length: int, kernel: int) -> tuple[mx.array, int]:
    """Per-index window start along one axis, plus the effective kernel.

    Exact port of the non-causal branch of ``eager._window_bounds``.
    """
    k = min(kernel, length)
    lo = length - k
    half = k // 2
    i = mx.arange(length)
    start = mx.minimum(mx.maximum(i - half, 0), lo)
    return start, k


def _axis_index(length: int, kernel: int) -> tuple[mx.array, int]:
    """(length, k) absolute key indices for every query along one axis."""
    start, k = window_starts(length, kernel)
    return start[:, None] + mx.arange(k)[None, :], k


def na3d(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    kernel: tuple[int, int, int],
    scale: float = 1.0,
    query_tile_t: int | None = None,
    query_tile_h: int | None = None,
) -> mx.array:
    """Mask-free 3D neighborhood attention.

    Args:
        q, k, v: (B, T, H, W, NH, HD) — already normed, scaled and RoPE'd.
        kernel: (k_t, k_h, k_w); each axis clamps to its length.
        scale: kept at 1.0 by the caller (the scale is folded into Q).
        query_tile_t, query_tile_h: optional chunking of the *query* T and H
            axes to bound the gathered K/V working set. Purely a memory device
            — each query's window is gathered independently, so any tiling is
            bit-exact (gated). Needed on real volumes: the smallest legal
            latent (3,7,7) already gives a 17x56x56 stage-5 grid, which at
            11^3 keys/query would need ~72 GB untiled.

    Returns:
        (B, T, H, W, NH, HD)
    """
    B, T, H, W, NH, HD = q.shape
    it, kt = _axis_index(T, kernel[0])
    ih, kh = _axis_index(H, kernel[1])
    iw, kw = _axis_index(W, kernel[2])

    query_tile_t = T if query_tile_t is None else query_tile_t
    query_tile_h = H if query_tile_h is None else query_tile_h

    t_rows = []
    for t0 in range(0, T, query_tile_t):
        t1 = min(t0 + query_tile_t, T)
        it_s = it[t0:t1]
        h_cols = []
        for h0 in range(0, H, query_tile_h):
            h1 = min(h0 + query_tile_h, H)
            ih_s = ih[h0:h1]
            tt, th = t1 - t0, h1 - h0

            # ONE flat gather. Three sequential per-axis takes materialise
            # oversized intermediates (the first spans the full H x W plane
            # before narrowing) — measured as the dominant cost. Building a
            # single flat token index and taking once allocates only the
            # result.
            flat = (
                it_s.reshape(tt, 1, 1, kt, 1, 1) * (H * W)
                + ih_s.reshape(1, th, 1, 1, kh, 1) * W
                + iw.reshape(1, 1, W, 1, 1, kw)
            ).reshape(-1)  # (tt*th*W*kt*kh*kw,), key order (t, h, w)

            def gather(src: mx.array) -> mx.array:
                g = mx.take(src.reshape(B, T * H * W, NH, HD), flat, axis=1)
                return g.reshape(B, tt, th, W, kt * kh * kw, NH, HD).transpose(0, 1, 2, 3, 5, 4, 6)

            kk = gather(k)
            vv = gather(v)
            qq = q[:, t0:t1, h0:h1][..., None, :].astype(mx.float32)

            scores = (qq * kk.astype(mx.float32)).sum(-1) * scale
            probs = mx.softmax(scores, axis=-1, precise=True)
            out = (probs[..., None] * vv.astype(mx.float32)).sum(-2)
            h_cols.append(out.astype(q.dtype))
            mx.eval(h_cols[-1])
            del kk, vv, qq, scores, probs, out
        t_rows.append(h_cols[0] if len(h_cols) == 1 else mx.concatenate(h_cols, axis=2))

    return t_rows[0] if len(t_rows) == 1 else mx.concatenate(t_rows, axis=1)


def na3d_dense_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    kernel: tuple[int, int, int],
    scale: float = 1.0,
) -> mx.array:
    """Obviously-correct O(N^2) reference: dense scores + an explicit window mask.

    Independent of :func:`na3d` (no shared index machinery beyond the documented
    window formula), so agreement between the two is real evidence. Only usable
    on tiny volumes — see LTX25-PORT-PLAN.md §D for the cost table.

    ⚠️ **Run this under ``with mx.stream(mx.cpu)``.** It is matmul-based, and
    Apple-GPU fp32 matmul accumulates ~1e-3 here, which swamps the comparison:
    on the GPU stream this disagrees with :func:`na3d` by ~5e-3 and looks like a
    port bug. On the CPU stream the two agree to ~2e-6. (:func:`na3d` itself is
    unaffected — it scores with an explicit multiply-and-sum over head_dim, not
    a matmul, and matches a float64 hand computation to 1e-7 on either stream.)
    """
    B, T, H, W, NH, HD = q.shape
    N = T * H * W
    st, kt = window_starts(T, kernel[0])
    sh, kh = window_starts(H, kernel[1])
    sw, kw = window_starts(W, kernel[2])

    ti = mx.arange(T)[:, None]
    hi = mx.arange(H)[:, None]
    wi = mx.arange(W)[:, None]
    vt = (mx.arange(T)[None, :] >= st[:, None]) & (mx.arange(T)[None, :] < (st + kt)[:, None])
    vh = (mx.arange(H)[None, :] >= sh[:, None]) & (mx.arange(H)[None, :] < (sh + kh)[:, None])
    vw = (mx.arange(W)[None, :] >= sw[:, None]) & (mx.arange(W)[None, :] < (sw + kw)[:, None])
    del ti, hi, wi

    vis = vt[:, None, None, :, None, None] & vh[None, :, None, None, :, None] & vw[None, None, :, None, None, :]
    vis = vis.reshape(N, N)

    qf = q.reshape(B, N, NH, HD).transpose(0, 2, 1, 3).astype(mx.float32)
    kf = k.reshape(B, N, NH, HD).transpose(0, 2, 1, 3).astype(mx.float32)
    vf = v.reshape(B, N, NH, HD).transpose(0, 2, 1, 3).astype(mx.float32)

    scores = (qf @ kf.transpose(0, 1, 3, 2)) * scale
    neg = mx.array(-1e9, dtype=mx.float32)
    scores = mx.where(vis[None, None], scores, neg)
    out = mx.softmax(scores, axis=-1, precise=True) @ vf
    return out.transpose(0, 2, 1, 3).reshape(B, T, H, W, NH, HD).astype(q.dtype)


def na3d_box(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    kernel: tuple[int, int, int],
    scale: float = 1.0,
    tile: tuple[int, int, int] = (2, 16, 64),
) -> mx.array:
    """Neighborhood attention via shared key boxes + MLX's fused SDPA.

    Alternative to :func:`na3d`. Where the gather path copies each key into
    every window that contains it (traffic ``N * prod(kernel) * heads *
    head_dim``, invariant to tiling), this gathers ONE contiguous key box per
    query tile — the tile plus a ``kernel - 1`` halo — and lets the fused
    attention kernel consume it without materialising the score matrix. Queries
    then need a boolean mask restricting each to its true neighborhood.

    Two things make it pay:

    * **Boolean masks.** MLX's fused SDPA accepts them and they are bit-identical
      to the additive form (verified), at a quarter of the bytes.
    * **Mask sharing.** The mask depends only on each query's window *relative*
      to the box corner, which is identical for every interior tile — so a
      handful of masks are built once and reused across all tiles. (This is what
      upstream's "group tiles by window geometry" code is doing.)

    Modelled traffic at the DiffVAE's stage-5 size: ~135 GB vs ~1140 GB for the
    gather path. Gated bit-against :func:`na3d`.
    """
    B, T, H, W, NH, HD = q.shape
    st_t, kt = window_starts(T, kernel[0])
    st_h, kh = window_starts(H, kernel[1])
    st_w, kw = window_starts(W, kernel[2])
    starts = [np.array(st_t), np.array(st_h), np.array(st_w)]
    ks = [kt, kh, kw]
    dims = [T, H, W]
    tiles = [max(1, min(tile[i], dims[i])) for i in range(3)]

    kf = k.reshape(B, T * H * W, NH, HD).transpose(0, 2, 1, 3)
    vf = v.reshape(B, T * H * W, NH, HD).transpose(0, 2, 1, 3)

    mask_cache: dict = {}
    out = mx.zeros((B, T, H, W, NH, HD), dtype=q.dtype)
    pieces = []

    for t0 in range(0, T, tiles[0]):
        t1 = min(t0 + tiles[0], T)
        for h0 in range(0, H, tiles[1]):
            h1 = min(h0 + tiles[1], H)
            row = []
            for w0 in range(0, W, tiles[2]):
                w1 = min(w0 + tiles[2], W)
                lo = [starts[i][[t0, h0, w0][i]] for i in range(3)]
                hi = [starts[i][[t1, h1, w1][i] - 1] + ks[i] for i in range(3)]
                ext = [int(hi[i] - lo[i]) for i in range(3)]
                rel = tuple(
                    tuple((starts[i][[range(t0, t1), range(h0, h1), range(w0, w1)][i]] - lo[i]).tolist())
                    for i in range(3)
                )
                sig = (rel, tuple(ext))
                m = mask_cache.get(sig)
                if m is None:
                    per_axis = []
                    for i in range(3):
                        r = np.asarray(rel[i])[:, None]
                        j = np.arange(ext[i])[None, :]
                        per_axis.append((j >= r) & (j < r + ks[i]))
                    m3 = (
                        per_axis[0][:, None, None, :, None, None]
                        & per_axis[1][None, :, None, None, :, None]
                        & per_axis[2][None, None, :, None, None, :]
                    )
                    m = mx.array(m3.reshape(int(np.prod(m3.shape[:3])), -1))
                    mask_cache[sig] = m

                # key box -> flat token ids
                bt = np.arange(lo[0], hi[0])[:, None, None] * (H * W)
                bh = np.arange(lo[1], hi[1])[None, :, None] * W
                bw = np.arange(lo[2], hi[2])[None, None, :]
                box_idx = mx.array((bt + bh + bw).reshape(-1))

                kb = mx.take(kf, box_idx, axis=2)
                vb = mx.take(vf, box_idx, axis=2)
                qt_ = q[:, t0:t1, h0:h1, w0:w1].reshape(B, -1, NH, HD).transpose(0, 2, 1, 3)
                o = mx.fast.scaled_dot_product_attention(qt_, kb, vb, scale=scale, mask=m)
                o = o.transpose(0, 2, 1, 3).reshape(B, t1 - t0, h1 - h0, w1 - w0, NH, HD)
                row.append(o.astype(q.dtype))
                mx.eval(row[-1])
            pieces.append((t0, t1, h0, h1, row))

    # reassemble
    t_groups: dict = {}
    for t0, t1, h0, h1, row in pieces:
        t_groups.setdefault((t0, t1), []).append(mx.concatenate(row, axis=3) if len(row) > 1 else row[0])
    rows = [mx.concatenate(v, axis=2) if len(v) > 1 else v[0] for _, v in sorted(t_groups.items())]
    return rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=1)
