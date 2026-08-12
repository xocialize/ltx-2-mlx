"""Self-consistency gate for the DiffVAE's 3D neighborhood attention.

Two independent implementations must agree before either is compared to torch:

  * :func:`na3d` — the shipping gather path (mask-free; the window is
    shift-clamped so every query has a full in-bounds neighborhood).
  * :func:`na3d_dense_reference` — O(N^2) dense scores + an explicit window
    mask, built from the window formula alone.

⚠️ The dense reference MUST run on the CPU stream: it is matmul-based, and
Apple-GPU fp32 matmul accumulates ~1e-3, which reads as a port bug (measured:
5e-3 on GPU vs 2e-6 on CPU for the same inputs).

Also checks the documented window bounds and that query-tiling is exact.

Usage:
    uv run --no-sync python scripts/gate_ltx25_na3d.py
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

TOL = 5e-5


def main() -> None:
    from ltx_core_mlx.model.video_vae.na3d import (
        na3d,
        na3d_dense_reference,
        window_starts,
    )

    ok = True

    # [1] window bounds vs the reference's documented behaviour
    checks = [
        (5, 3, {0: 0, 2: 1, 4: 2}),
        (9, 7, {0: 0, 4: 1, 8: 2}),
    ]
    for L, K, expect in checks:
        st, k = window_starts(L, K)
        got = {i: int(np.array(st)[i]) for i in expect}
        good = got == expect and k == K
        ok &= good
        print(f"  [1] window_starts(L={L}, K={K}) spot {got} {'PASS' if good else 'FAIL'}")
    st, k = window_starts(8, 11)
    good = k == 8 and set(np.array(st).tolist()) == {0}
    ok &= good
    print(f"  [1] short-axis clamp L=8 K=11 -> k={k} {'PASS' if good else 'FAIL'}")

    # [2] gather vs dense reference (CPU stream)
    rng = np.random.default_rng(3)
    for shape, kern in (
        ((1, 5, 9, 9, 2, 16), (3, 7, 7)),
        ((1, 4, 6, 6, 2, 16), (3, 5, 5)),
        ((1, 3, 4, 4, 1, 8), (11, 11, 11)),
    ):
        arrs = [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]
        got = na3d(*(mx.array(a) for a in arrs), kern)
        mx.eval(got)
        with mx.stream(mx.cpu):
            ref = na3d_dense_reference(*(mx.array(a) for a in arrs), kern)
            mx.eval(ref)
        d = float(np.abs(np.array(got) - np.array(ref)).max())
        good = d < TOL
        ok &= good
        print(f"  [2] gather vs dense {str(shape):22s} {str(kern):12s} maxAbs {d:.3e} "
              f"{'PASS' if good else 'FAIL'}")

    # [3] query tiling is exact
    arrs = [rng.standard_normal((1, 8, 6, 6, 2, 16)).astype(np.float32) for _ in range(3)]
    full = na3d(*(mx.array(a) for a in arrs), (3, 5, 5))
    tiled = na3d(*(mx.array(a) for a in arrs), (3, 5, 5), query_tile_t=2)
    mx.eval(full, tiled)
    d = float(mx.abs(full - tiled).max())
    ok &= d == 0.0
    print(f"  [3] query-tile exactness (t=2 vs full): maxAbs {d:.1e} {'PASS' if d == 0.0 else 'FAIL'}")

    print(f"NA3D GATE: {'PASS' if ok else 'FAIL'}")
    assert ok


if __name__ == "__main__":
    main()
