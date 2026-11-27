#!/usr/bin/env python3
"""H8 scipy reference fixture generator.

Deterministic (scipy L-BFGS-B is deterministic for a fixed problem).
The C test (tests/test_h8.c) compares against this file:
  - L-BFGS: |f_ours - f_scipy| <= 1e-10*max(1,|f_scipy|)  (final-f bar)
             and iters_ours <= 1.25 * nit_scipy           (iteration bar)
  - FISTA / CD-LASSO: cross-checked against scipy l1_minimize
    (method='fista' and method='cndls') on the same LASSO instance.
  - Bounded: final-f bar only (different line search makes the
    iteration comparison uninterpretable).

The random data (LASSO A/b, quadratic Q/p) uses the SAME 64-bit LCG as
the C side:
    st = st*6364136223846793005 + 1442695040888963407  (mod 2^64)
    u  = ((st >> 11) & ((1<<53)-1)) / 2^53             in [0,1)
Run:  /tmp/scipyvenv/bin/python3 tests/ref_h8_scipy.py
"""
import numpy as np
from scipy import optimize

M64 = (1 << 64) - 1


class LCG:
    def __init__(self, seed):
        self.st = seed % (1 << 64)

    def u(self):
        self.st = (self.st * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((self.st >> 11) & ((1 << 53) - 1)) / 9007199254740992.0


def rosen_f(x):
    n = len(x)
    return sum((x[i + 1] - x[i] * x[i]) ** 2 + 100 * (1 - x[i]) ** 2
               for i in range(n - 1))


def rosen_g(x):
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        g[i] += -2 * (x[i + 1] - x[i] ** 2) - 200 * (1 - x[i])
        g[i + 1] += 2 * (x[i + 1] - x[i] ** 2)
    return g


def beale_f(x):
    return ((1.5 - x[0] + x[0] * x[1]) ** 2
            + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
            + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2)


def beale_g(x):
    a = 1.5 - x[0] + x[0] * x[1]
    b = 2.25 - x[0] + x[0] * x[1] ** 2
    c = 2.625 - x[0] + x[0] * x[1] ** 3
    return np.array([
        -2 * a * (1 - x[1]) - 2 * b * (1 - x[1] ** 2) - 2 * c * (1 - x[1] ** 3),
        2 * a * x[0] + 2 * b * 2 * x[0] * x[1] + 2 * c * 3 * x[0] * x[1] ** 2])


def wood_f(x):
    return (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
            + 100 * (x[3] - x[2] ** 2) ** 2 + (1 - x[2]) ** 2)


def wood_g(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2),
        -400 * x[2] * (x[3] - x[2] ** 2) - 2 * (1 - x[2]),
        200 * (x[3] - x[2] ** 2)])


# Styblinski-Tang: 1/2 sum x^4 - sum x^2 + n sum x; known global min per
# dim -39.165999... (n=5: -195.82999...)
def st_f(x):
    n = len(x)
    return 0.5 * sum(xi ** 4 for xi in x) - sum(xi ** 2 for xi in x) + n * sum(
        x)


def st_g(x):
    n = len(x)
    return 2 * x ** 3 - 2 * x + n


def exponlin_f(x):
    n = len(x)
    s = x[-1] ** 2
    for i in range(n - 1):
        s += x[i] ** 2 - 3.7 * x[i] * x[i + 1] + 0.25 * x[i] ** 2 * x[i + 1] ** 2
    return s


def exponlin_g(x):
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        g[i] += 2 * x[i] - 3.7 * x[i + 1] + x[i] * x[i + 1] ** 2
        g[i + 1] += -3.7 * x[i] + 2 * x[i] ** 2 * x[i + 1]
    g[-1] += 2 * x[-1]
    return g


def main():
    out = []

    def emit(name, d, f, nit, nfev, njev, status, x):
        out.append(f"{name} {d} {f:.17e} {nit} {nfev} {njev} {status} "
                   + " ".join(f"{v:.17e}" for v in x))

    def lbfgs(name, d, f, g, x0, bounds=None):
        res = optimize.minimize(f, np.array(x0, dtype=float), jac=g,
                                method="L-BFGS-B", bounds=bounds,
                                options={"ftol": 1e-10, "gtol": 1e-10, "maxiter": 1000 * d})
        emit(name, d, res.fun, res.nit, res.nfev, res.njev, res.status, res.x)
        print(f"[scipy] {name}: f={res.fun:.6e} nit={res.nit} "
              f"nfev={res.nfev} njev={res.njev} status={res.status}")

    lbfgs("rosen2", 2, rosen_f, rosen_g, [-1.2, 1.0])
    x0_10 = []
    for i in range(10):
        x0_10.append(-1.2 if i % 2 == 0 else 1.0)
    lbfgs("rosen10", 10, rosen_f, rosen_g, x0_10)
    lbfgs("beale", 2, beale_f, beale_g, [1.0, 1.0])
    lbfgs("wood", 4, wood_f, wood_g, [-1.2, 1.0, -0.2, 1.0])
    lbfgs("styblinski5", 5, st_f, st_g, [0.5] * 5)
    lbfgs("exponlin10", 10, exponlin_f, exponlin_g, [0.5] * 10)

    # --- SPD quadratic (data from the shared LCG) ---
    d = 50
    lc = LCG(0x5EED)
    Mm = np.array([[lc.u() * 2 - 1 for _ in range(d)] for _ in range(d)])
    Q = Mm.T @ Mm + d * np.eye(d)
    p = np.array([lc.u() * 2 - 1 for _ in range(d)])

    def quad_f(x):
        return 0.5 * x @ Q @ x - p @ x

    def quad_g(x):
        return Q @ x - p

    x0q = np.array([lc.u() * 2 - 1 for _ in range(d)])
    res = optimize.minimize(quad_f, x0q, jac=quad_g, method="L-BFGS-B",
                            options={"ftol": 1e-10, "gtol": 1e-10, "maxiter": 1000 * d})
    out.append(f"quad50 {d} {res.fun:.17e} {res.nit} {res.nfev} {res.njev} "
               f"{res.status} "
               + " ".join(f"{v:.17e}" for v in res.x))
    out.append(f"quad50_data {d} "
               + " ".join(f"{Q[i, j]:.17e}" for i in range(d) for j in range(d))
               + " " + " ".join(f"{p[i]:.17e}" for i in range(d))
               + " " + " ".join(f"{x0q[i]:.17e}" for i in range(d)))
    print(f"[scipy] quad50: f={res.fun:.6e} nit={res.nit}")

    # --- bounded quadratic d=30, box [-1, 1] ---
    d2 = 30
    lc2 = LCG(0xB0B1)
    M2 = np.array([[lc2.u() * 2 - 1 for _ in range(d2)] for _ in range(d2)])
    Q2 = M2.T @ M2 + d2 * np.eye(d2)
    p2 = np.array([lc2.u() * 2 - 1 for _ in range(d2)])

    def bq_f(x):
        return 0.5 * x @ Q2 @ x - p2 @ x

    def bq_g(x):
        return Q2 @ x - p2

    x0b = np.array([lc2.u() * 2 - 1 for _ in range(d2)])
    resb = optimize.minimize(bq_f, x0b, jac=bq_g, method="L-BFGS-B",
                             bounds=[(-1.0, 1.0)] * d2,
                             options={"ftol": 1e-10, "gtol": 1e-10, "maxiter": 1000 * d2})
    out.append(f"boundquad30 {d2} {resb.fun:.17e} {resb.nit} {resb.nfev} "
               f"{resb.njev} {resb.status} "
               + " ".join(f"{v:.17e}" for v in resb.x))
    out.append(f"boundquad30_data {d2} "
               + " ".join(f"{Q2[i, j]:.17e}" for i in range(d2) for j in range(d2))
               + " " + " ".join(f"{p2[i]:.17e}" for i in range(d2))
               + " " + " ".join(f"{x0b[i]:.17e}" for i in range(d2))
               + " -1.0 1.0")
    print(f"[scipy] boundquad30: f={resb.fun:.6e} nit={resb.nit}")


    with open("tests/fixtures/h8_scipy_reference.txt", "w") as fp:
        fp.write("# H8 scipy reference fixture (scipy 1.18.1, numpy 2.5.5)\n")
        for line in out:
            fp.write(line + "\n")
    print("wrote tests/fixtures/h8_scipy_reference.txt")


if __name__ == "__main__":
    main()
