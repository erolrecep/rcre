#!/usr/bin/env python3
"""H9 fixture generator (infra-immune: numpy only needed to REGENERATE
this fixture; the test reads the checked-in text file).

Produces tests/fixtures/h9_poisson3d_reference.txt: the 3-D Poisson
solution (7-point Dirichlet stencil, h=1, zero boundary) on the
irregular 33x31x29 grid for the fixed LCG RHS used by
tests/test_h9.c (seed 0x424242, the C lcg01). Irregular on purpose:
an axis-stride or layout bug in the C 3-D spectral solver shows up
immediately (33/31/29 are all prime — every DST line runs the
Bluestein path, the generic-FFT half of the H9 solver).

DST-I solver in numpy (the external oracle): the 1-D DST-I via the
even (length 2(n+1)) odd extension and a real FFT, per axis, with the
eigenvalue divide in between (the same math as src/core/pde3d.c,
independent implementation).
"""
import numpy as np
import os


def lcg01_array(n, seed):
    # the C test's lcg01 (uint64 wrap, (st>>11) & 2^53-1, /2^53)
    M = 0xFFFFFFFFFFFFFFFF
    st = seed
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        st = (st * 6364136223846793005 + 1442695040888963407) & M
        out[i] = ((st >> 11) & ((1 << 53) - 1)) / 9007199254740992.0
    return out


def dst1_axis(x, axis, invert):
    """1-D DST-I along `axis` via the odd extension of length 2(n+1).
    forward: D[k] = sum_j x[j] sin(pi j k/(n+1))  (k=1..n)
    inverse: x = (2/(n+1)) D (same transform, scaled) — implemented as
    a forward transform here (the caller applies the scale)."""
    n = x.shape[axis]
    # odd extension line-by-line: y[j] = x[j] (j=1..n), y[2(n+1)-j] = -x[j]
    x1 = np.moveaxis(x, axis, 0).reshape(n, -1)
    m = x1.shape[1]
    N = 2 * (n + 1)
    y = np.zeros((N, m), dtype=np.float64)
    y[1 : n + 1] = x1
    y[N - np.arange(1, n + 1)] = -x1
    Y = np.fft.fft(y, axis=0)
    d = -0.5 * Y.imag[1 : n + 1]
    out = d.reshape((n,) + x.shape[:axis] + x.shape[axis + 1 :])
    return np.moveaxis(out, 0, axis)


def poisson3d(b, nx, ny, nz):
    """Solve the 7-point Dirichlet Poisson on (nx, ny, nz), zero
    boundary, h=1, via the separable DST-I."""
    w = b
    # forward: x, y, z
    w = dst1_axis(w, 0, 0)
    w = dst1_axis(w, 1, 0)
    w = dst1_axis(w, 2, 0)
    # eigenvalue divide
    # eigenvalue divide. Layout (ny, nz, nx): axis 0 length ny (y),
    # axis 1 length nz (z), axis 2 length nx (x) — separable, order
    # irrelevant.
    ly = 2.0 - 2.0 * np.cos(np.pi * (np.arange(ny) + 1) / (ny + 1))
    lz = 2.0 - 2.0 * np.cos(np.pi * (np.arange(nz) + 1) / (nz + 1))
    lx = 2.0 - 2.0 * np.cos(np.pi * (np.arange(nx) + 1) / (nx + 1))
    lam = ly[:, None, None] + lz[None, :, None] + lx[None, None, :]
    w = w / lam
    # inverse: z, y, x (same transforms)
    w = dst1_axis(w, 2, 1)
    w = dst1_axis(w, 1, 1)
    w = dst1_axis(w, 0, 1)
    scale = (2.0 / (nx + 1)) * (2.0 / (ny + 1)) * (2.0 / (nz + 1))
    return scale * w


def main():
    nx, ny, nz = 33, 31, 29
    n3 = nx * ny * nz
    # the C test's RHS: lcg01(seed=0x424242) * 2 - 1, flat order
    # c = j*nz*nx + k*nx + i  ->  array shape (ny, nz, nx)
    flat = lcg01_array(n3, 0x424242) * 2.0 - 1.0
    b = flat.reshape((ny, nz, nx))
    u = poisson3d(b, nx, ny, nz)
    out = os.path.join(os.path.dirname(__file__), "fixtures", "h9_poisson3d_reference.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        # flat order (j, k, i) — the C reader reads n3 lines in the
        # same c = j*nz*nx + k*nx + i order
        for v in u.reshape(-1):
            # float() first: numpy 2.x reprs np.float64 as "np.float64(...)"
            f.write(repr(float(v)) + "\n")
    print(f"wrote {out} ({n3} values)")
    # self-check: residual
    u2 = u
    # 7-point stencil
    a = np.zeros_like(u)
    a += 6.0 * u2
    a[:, :, 1:] -= u2[:, :, :-1]
    a[:, :, :-1] -= u2[:, :, 1:]
    a[1:, :, :] -= u2[:-1, :, :]
    a[:-1, :, :] -= u2[1:, :, :]
    a[:, 1:, :] -= u2[:, :-1, :]
    a[:, :-1, :] -= u2[:, 1:, :]
    rel = np.abs(a - b).max() / np.abs(b).max()
    print(f"numpy self-check residual = {rel:.3e}")
    assert rel < 1e-12


if __name__ == "__main__":
    main()
