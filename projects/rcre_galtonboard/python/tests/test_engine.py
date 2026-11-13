"""Unit and statistical convergence tests for Python Galton engine."""

import pytest
import math
from app.models import SimulationConfig
from app.engine import GaltonEngine, binomial_pmf, gaussian_pdf, chi_square_p_value


def test_binomial_pmf_values():
    """Verify exact analytical values of Binomial PMF."""
    # (4 choose 2) * 0.5^4 = 6/16 = 0.375
    assert math.isclose(binomial_pmf(4, 2, 0.5), 0.375, rel_tol=1e-7)

    # Check that sum of PMF for n=10, p=0.4 is 1.0
    total = sum(binomial_pmf(10, k, 0.4) for k in range(11))
    assert math.isclose(total, 1.0, rel_tol=1e-7)


def test_gaussian_pdf():
    """Verify Gaussian normal PDF values."""
    # Peak at mean: 1 / (sigma * sqrt(2*pi))
    sigma = 2.0
    expected_peak = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    assert math.isclose(gaussian_pdf(10.0, 10.0, sigma), expected_peak, rel_tol=1e-7)


def test_clt_convergence_symmetric():
    """Verify Central Limit Theorem convergence with 100,000 balls."""
    config = SimulationConfig(rows=16, balls=100000, bias=0.5, seed=42)
    engine = GaltonEngine(config)
    engine.simulate_batch(config.balls)

    stats = engine.calculate_statistics()

    # Target: mean = 8.0, var = 4.0, std = 2.0
    assert abs(stats.mean - 8.0) < 0.03
    assert abs(stats.variance - 4.0) < 0.06
    assert abs(stats.skewness) < 0.03
    assert stats.clt_verified is True
    assert stats.chi_square_p_value > 0.001


def test_skewed_distribution():
    """Verify asymmetric distribution (bias = 0.25)."""
    config = SimulationConfig(rows=12, balls=100000, bias=0.25, seed=99)
    engine = GaltonEngine(config)
    engine.simulate_batch(config.balls)

    stats = engine.calculate_statistics()

    # Target: mean = 3.0, positive skewness ~ +0.33
    assert abs(stats.mean - 3.0) < 0.04
    assert stats.skewness > 0.15
    assert stats.clt_verified is True
