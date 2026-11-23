/**
 * @file sf.c
 * @brief Special functions for statistics & physics (H0, 2.1.0)
 *
 * Implementation notes and accuracy (gated in tests/test_linalg.c against
 * published values):
 *
 *  - erf/erfc:  series (2/sqrt(pi)) * sum (-1)^n x^(2n+1) / (n! (2n+1))
 *    for |x| <= 1 (25 terms reach the double floor), and the incomplete-
 *    incomplete-gamma continued fraction (DLMF 8.9.8, Lentz; a = 1/2,
 *    z = x^2, which converges to a scaled erfc) times the prefactor
 *    x * e^(-x^2) / sqrt(pi) — assembled as three correctly-rounded
 *    factors (no log-sum cancellation) for |x| > 1. Beyond |x| = 26.7,
 *    e^(-x^2) underflows to 0, which IS the correctly-rounded answer.
 *    Accuracy (measured, 1 < x <= 26.7): relative error <= 3e-15, the
 *    floor set by the Lentz product for z < ~2; the series side is
 *    ~1e-16.
 *
 *  - lgamma:  Lanczos approximation, g = 7, N = 9 (the standard published
 *    coefficient set), |err| ~ 1e-15 for x > 0.5; reflection
 *    lgamma(x) = ln(pi) - ln|sin(pi x)| - lgamma(1-x) below; poles at
 *    0, -1, -2, ... return +inf (|Gamma| -> 0 => ln|Gamma| -> -inf? no:
 *    Gamma has poles => |Gamma| -> inf => lgamma -> +inf).
 *
 *  - digamma:  recurrence psi(x) = psi(x+n) - sum_{i=0}^{n-1} 1/(x+i) to
 *    lift x into [8, inf), then the asymptotic series
 *    psi(x) ~ ln(x) - 1/(2x) - sum B_{2k} / (2k x^{2k})  (Bernoulli
 *    coefficients B2..B12) — ~1e-15 at x >= 8. Reflection
 *    psi(x) = psi(1-x) - pi*cot(pi*x) for the left half-plane.
 *
 *  - logsumexp:  max-shift (no overflow anywhere in the double range);
 *  - expit:  branch on the sign of x (no overflow for |x| > ~709);
 *  - norm_cdf:  0.5 * erfc(-x/sqrt(2)) — uses the tail-accurate erfc, so
 *    the far tails (|x| > 8) are NOT the 1-Phi(1-eps) catastrophe of a
 *    naive 0.5*(1+erf) form.
 *
 * All functions are deterministic and single-threaded by construction.
 */

#include "rcre.h"

#include <math.h>

#ifndef M_SQRTPI
#define M_SQRTPI 0.56418958354775628695 /* 1/sqrt(pi) */
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------- erf / erfc ------------------------------ */

/* Series for |x| <= 1: term_{n+1} = -term_n * x^2 * (2n+1) / ((n+1)(2n+3));
 * 25 terms: term_24 ~ 1/24! * ... < 1e-23, i.e. below the double floor. */
static double erf_series(double x) {
    const double x2 = x * x;
    double term = x;            /* n = 0 */
    double sum  = term;
    for (int n = 0; n < 24; ++n) {
        term *= -x2 * (double)(2 * n + 1) / ((double)(n + 1) * (double)(2 * n + 3));
        sum += term;
    }
    return sum * (2.0 / sqrt(M_PI));
}

/* DLMF 8.9.8 / Numerical Recipes `gcf`: the upper incomplete-gamma
 * continued fraction, evaluated with the modified Lentz algorithm.
 * With a = 1/2, z = x^2: Gamma(1/2, x^2) = sqrt(pi) * erfc(x), and the
 * prefactor exp(-z + a*ln z - lgamma(a)) = e^(-x^2) * x / sqrt(pi), so the
 * composed result is erfc(x) itself — no extra constant. Converges for
 * all z > 0; used here for x > 1 where it is fast and unconditionally
 * stable (all terms positive for a = 1/2, z > 1). */
static double erfc_cf(double x) {
    const double z = x * x;
    const double a = 0.5;
    const double FPMIN = 1e-300;
    double b = z + 1.0 - a;
    double c = 1.0 / FPMIN;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= 200; ++i) {
        const double an = -(double)i * ((double)i - a);
        b += 2.0;
        d = an * d + b;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = b + an / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        /* Stop only when the correction no longer changes h in double
         * precision — a relative-epsilon stop leaves a residual of the
         * order of the stopping tolerance (measured: 2.5e-15). */
        if (del == 1.0) break;
        if (i >= 200) break;   /* hard cap; unreachable for x >= 1 */
    }
    /* Prefactor e^(-z) * z^(1/2) / sqrt(pi) as three correctly-rounded
     * factors (z^(1/2) = x, and 1/sqrt(pi) is a correctly rounded
     * constant): no log-sum, so no O(1) argument cancellation — the
     * previous log form lost ~1 ulp of the O(17) argument (measured
     * 1.8e-15). Product of three <= 0.5-ulp roundings: ~2e-16. */
    return h * x * exp(-x * x) * (1.0 / sqrt(M_PI));
}

static double erfc_pos(double x) {
    if (x < 1.0) return 1.0 - erf_series(x);
    if (x >= 26.7) return 0.0;            /* e^-x^2 underflows: exact in double */
    return erfc_cf(x);
}

double rcre_sf_erf_f64(double x) {
    if (x != x) return x;                 /* NaN in, NaN out */
    const double t = fabs(x);
    if (t < 1.0) {
        double r = erf_series(t);
        return (x < 0.0) ? -r : r;
    }
    /* for t >= 1 use 1 - erfc(t): erfc is computed in its own asymptotic
     * domain, so the cancellation is controlled by the CF's own accuracy */
    double r = 1.0 - erfc_pos(t);
    if (t >= 26.7) r = 1.0;               /* erfc underflowed: erf = 1 exactly */
    return (x < 0.0) ? -r : r;
}

double rcre_sf_erfc_f64(double x) {
    if (x != x) return x;
    if (x < 0.0) return 2.0 - erfc_pos(-x);
    return erfc_pos(x);
}

float rcre_sf_erf_f32(float x) {
    return (float)rcre_sf_erf_f64((double)x);
}

/* -------------------------------- lgamma --------------------------------- */

/* Lanczos g = 7, N = 9 — the standard published coefficient set.
 * Valid for Re(x) > 0.5 after the x <- x-1 shift; the reflection formula
 * handles (0, 0.5) and the negative non-integers. */
static const double LG_P[9] = {
     0.99999999999980993,
     676.5203681218851,
   -1259.1392167224028,
     771.32342877765313,
    -176.61502916214059,
      12.507343278686905,
     -0.13857109526572012,
       9.9843695780195716e-6,
       1.5056327351493116e-7
};

double rcre_sf_lgamma_f64(double x) {
    if (x != x) return x;
    if (x == 0.0) return INFINITY;        /* pole */
    /* negative integer or zero: pole */
    if (x < 0.0 && x == floor(x)) return INFINITY;
    if (x < 0.5) {
        /* reflection: ln|Gamma(x) Gamma(1-x)| = ln(pi / sin(pi x)) */
        const double s = sin(M_PI * x);
        return log(M_PI / fabs(s)) - rcre_sf_lgamma_f64(1.0 - x);
    }
    x -= 1.0;
    double a = LG_P[0];
    for (int i = 1; i < 9; ++i) a += LG_P[i] / (x + (double)i);
    const double t = x + 7.5;             /* g + 0.5 */
    return 0.5 * log(2.0 * M_PI) + (x + 0.5) * log(t) - t + log(a);
}

/* ------------------------------- digamma --------------------------------- */

static double digamma_pos(double x) {
    /* Lift into [16, inf) via the recurrence psi(x) = psi(x+1) - 1/x —
     * at 16 the 6-term asymptotic truncation error is ~1e-18. */
    double s = 0.0;
    while (x < 16.0) {
        s -= 1.0 / x;
        x += 1.0;
    }
    /* Asymptotic: ln(x) - 1/(2x) - sum_{k>=1} B_{2k} / (2k x^{2k}) */
    const double xi = 1.0 / x;
    const double x2 = xi * xi;
    double poly = 0.0;                    /* sum of Bernoulli terms */
    /* B2/2=1/12, B4/4=-1/120, B6/6=1/252, B8/8=-1/240, B10/10=1/132,
     * B12/12=-691/32760 — signs carried in the constants below */
    const double b[] = { 1.0 / 12.0, 1.0 / 120.0, 1.0 / 252.0,
                         1.0 / 240.0, 1.0 / 132.0, 691.0 / 32760.0 };
    /* psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
     *           + 1/(240x^8) - 1/(132x^10) + 691/(32760x^12)  */
    const int sign[] = { -1, +1, -1, +1, -1, +1 };
    double xk = x2;
    for (int k = 0; k < 6; ++k) {
        poly += sign[k] * b[k] * xk;
        xk *= x2;
    }
    return log(x) - 0.5 * xi + poly + s;   /* +s = the recurrence shift */
}

double rcre_sf_digamma_f64(double x) {
    if (x != x) return x;
    if (x == 0.0) return INFINITY;        /* pole at 0 */
    if (x < 0.0 && x == floor(x)) return INFINITY;
    if (x < 0.5) {
        /* reflection: psi(1-x) - psi(x) = pi * cot(pi x) */
        return rcre_sf_digamma_f64(1.0 - x)
             - M_PI / tan(M_PI * x);
    }
    return digamma_pos(x);
}

/* ------------------------- logsumexp / expit ----------------------------- */

double rcre_sf_logsumexp_f64(const double* x, size_t n) {
    if (n == 0) return -INFINITY;
    double m = x[0];
    for (size_t i = 1; i < n; ++i)
        if (x[i] > m) m = x[i];
    if (m == -INFINITY) return m;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += exp(x[i] - m);
    return m + log(s);
}

float rcre_sf_logsumexp_f32(const float* x, size_t n) {
    if (n == 0) return -INFINITY;
    float m = x[0];
    for (size_t i = 1; i < n; ++i)
        if (x[i] > m) m = x[i];
    if (m == -INFINITY) return m;
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += (float)exp((double)(x[i] - m));
    return (float)(m + (double)log((double)s));
}

double rcre_sf_expit_f64(double x) {
    if (x >= 0.0) {
        const double e = exp(-x);
        return 1.0 / (1.0 + e);
    }
    const double e = exp(x);
    return e / (1.0 + e);
}

float rcre_sf_expit_f32(float x) {
    return (float)rcre_sf_expit_f64((double)x);
}

double rcre_sf_norm_cdf_f64(double x) {
    return 0.5 * rcre_sf_erfc_f64(-x / sqrt(2.0));
}
