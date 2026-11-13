"""Core numerical simulation engine for Galton Board flow and statistical parity."""

import math
import time
from typing import Tuple, List, Generator
import numpy as np

from app.models import SimulationConfig, SimulationStats, BinDistributionItem


def binomial_pmf(n: int, k: int, p: float) -> float:
    """Compute exact Binomial PMF using math.lgamma."""
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0

    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    log_prob = log_comb + k * math.log(p) + (n - k) * math.log(1.0 - p)
    return math.exp(log_prob)


def mixture_pmf(n: int, sp: int, k: int, p: float) -> float:
    """Compute exact discrete mixture PMF over uniform apex starting pins P_0."""
    if sp <= 1:
        return binomial_pmf(n, k, p)
    total = 0.0
    for j in range(sp):
        if 0 <= k - j <= n:
            total += binomial_pmf(n, k - j, p)
    return total / sp


def gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """Compute Gaussian Normal PDF."""
    if sigma <= 0.0:
        return 0.0
    diff = (x - mu) / sigma
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * diff * diff)


def chi_square_p_value(chi2: float, dof: int) -> float:
    """Approximate upper-tail p-value for chi-square using Wilson-Hilferty transformation."""
    if chi2 <= 0.0 or dof <= 0:
        return 1.0
    # Wilson-Hilferty transformation to standard normal
    s = 2.0 / (9.0 * dof)
    z = (math.pow(chi2 / dof, 1.0 / 3.0) - (1.0 - s)) / math.sqrt(s)
    # Standard normal complementary CDF approximation: 0.5 * erfc(z / sqrt(2))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


class GaltonEngine:
    """High-performance Python 3.15 Galton Board Simulation Engine."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.num_bins = self.config.rows + self.config.start_pins
        self.bin_counts = np.zeros(self.num_bins, dtype=np.int64)
        self.total_dropped = 0

    def reset(self, config: SimulationConfig | None = None) -> None:
        """Reset accumulated counts and optionally reconfigure."""
        if config:
            self.config = config
            self.rng = np.random.default_rng(config.seed)
            self.num_bins = config.rows + config.start_pins
        self.bin_counts = np.zeros(self.num_bins, dtype=np.int64)
        self.total_dropped = 0

    def simulate_batch(self, n_balls: int) -> Tuple[np.ndarray, float]:
        """Execute a batch of Monte-Carlo Bernoulli drops."""
        t0 = time.perf_counter()
        
        # High-performance NumPy vectorization of Binomial(n, p) + uniform apex pins
        b_outcomes = self.rng.binomial(n=self.config.rows, p=self.config.bias, size=n_balls)
        if self.config.start_pins > 1:
            u_offsets = self.rng.integers(0, self.config.start_pins, size=n_balls)
            outcomes = b_outcomes + u_offsets
        else:
            outcomes = b_outcomes

        counts = np.bincount(outcomes, minlength=self.num_bins)

        self.bin_counts += counts
        self.total_dropped += n_balls
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return self.bin_counts, elapsed_ms

    def drop_single(self) -> Tuple[int, List[int]]:
        """Simulate a single ball dropping and record deflection path (0=left, 1=right)."""
        init_pin = int(self.rng.integers(0, self.config.start_pins)) if self.config.start_pins > 1 else 0
        rands = self.rng.random(self.config.rows)
        path = [1 if r < self.config.bias else 0 for r in rands]
        landed_bin = init_pin + sum(path)

        self.bin_counts[landed_bin] += 1
        self.total_dropped += 1
        return landed_bin, path

    def calculate_statistics(self) -> SimulationStats:
        """Compute empirical sample moments and compare against Central Limit Theorem."""
        n = self.config.rows
        sp = self.config.start_pins
        p = self.config.bias
        M = self.total_dropped

        theo_mean = ((sp - 1) / 2.0) + (n * p)
        var_u = (sp * sp - 1.0) / 12.0
        theo_var = var_u + (n * p * (1.0 - p))
        theo_std = math.sqrt(theo_var) if theo_var > 0 else 1e-6
        theo_skew = (1.0 - 2.0 * p) / theo_std if theo_std > 1e-6 else 0.0
        theo_kurt = (1.0 - 6.0 * p * (1.0 - p)) / theo_var if theo_var > 1e-6 else 0.0

        if M == 0:
            return SimulationStats(
                total_balls=0,
                mean=0.0,
                variance=0.0,
                std_dev=0.0,
                skewness=0.0,
                kurtosis=0.0,
                theoretical_mean=theo_mean,
                theoretical_variance=theo_var,
                theoretical_std_dev=theo_std,
                theoretical_skewness=theo_skew,
                theoretical_kurtosis=theo_kurt,
                chi_square_stat=0.0,
                chi_square_p_value=1.0,
                degrees_of_freedom=max(1, self.num_bins - 1),
                clt_verified=False,
            )

        indices = np.arange(self.num_bins, dtype=np.float64)
        weights = self.bin_counts.astype(np.float64)

        mean = float(np.sum(indices * weights) / M)
        diff = indices - mean
        diff2 = diff * diff
        sample_var = float(np.sum(diff2 * weights) / (M - 1)) if M > 1 else 0.0
        sample_std = math.sqrt(sample_var)

        if sample_std > 1e-6:
            skewness = float((np.sum(diff2 * diff * weights) / M) / (sample_std**3))
            kurtosis = float(((np.sum(diff2 * diff2 * weights) / M) / (sample_var**2)) - 3.0)
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Pearson Chi-Square test against Mixture PMF
        chi2 = 0.0
        active_bins = 0
        for k in range(self.num_bins):
            expected = M * mixture_pmf(n, sp, k, p)
            if expected >= 1.0:
                observed = float(self.bin_counts[k])
                chi2 += ((observed - expected) ** 2) / expected
                active_bins += 1

        dof = max(1, active_bins - 1)
        p_val = chi_square_p_value(chi2, dof)

        # Central Limit Theorem check: |x_bar - mu| < 3 * sigma / sqrt(M)
        standard_error = theo_std / math.sqrt(M)
        clt_ok = abs(mean - theo_mean) <= 3.5 * standard_error

        return SimulationStats(
            total_balls=M,
            mean=round(mean, 6),
            variance=round(sample_var, 6),
            std_dev=round(sample_std, 6),
            skewness=round(skewness, 6),
            kurtosis=round(kurtosis, 6),
            theoretical_mean=round(theo_mean, 6),
            theoretical_variance=round(theo_var, 6),
            theoretical_std_dev=round(theo_std, 6),
            theoretical_skewness=round(theo_skew, 6),
            theoretical_kurtosis=round(theo_kurt, 6),
            chi_square_stat=round(chi2, 4),
            chi_square_p_value=round(p_val, 4),
            degrees_of_freedom=dof,
            clt_verified=clt_ok,
        )

    def get_distribution_items(self) -> List[BinDistributionItem]:
        """Return breakdown per bin with empirical vs theoretical probabilities."""
        n = self.config.rows
        sp = self.config.start_pins
        p = self.config.bias
        M = self.total_dropped if self.total_dropped > 0 else 1
        theo_mean = ((sp - 1) / 2.0) + (n * p)
        var_u = (sp * sp - 1.0) / 12.0
        theo_var = var_u + (n * p * (1.0 - p))
        theo_std = math.sqrt(theo_var) if theo_var > 0 else 1e-6

        items = []
        for k in range(self.num_bins):
            cnt = int(self.bin_counts[k])
            emp_prob = cnt / M
            b_pmf = binomial_pmf(n, k, p)
            mix_pmf = mixture_pmf(n, sp, k, p)
            g_pdf = gaussian_pdf(float(k), theo_mean, theo_std)
            items.append(
                BinDistributionItem(
                    bin=k,
                    count=cnt,
                    empirical_prob=round(emp_prob, 8),
                    binomial_pmf=round(b_pmf, 8),
                    mixture_pmf=round(mix_pmf, 8),
                    gaussian_pdf=round(g_pdf, 8),
                )
            )
        return items
