"""Pydantic schemas and data models for RCRE Galton Board FastAPI Service."""

from typing import List, Optional
from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    """Configuration for a Galton Board simulation run."""
    rows: int = Field(default=12, ge=1, le=64, description="Number of pin rows N")
    balls: int = Field(default=50000, ge=1, le=50_000_000, description="Total balls to simulate")
    bias: float = Field(default=0.50, ge=0.0, le=1.0, description="Right bounce deflection probability p")
    seed: Optional[int] = Field(default=None, description="Optional PRNG seed for deterministic runs")


class BinDistributionItem(BaseModel):
    """Statistical distribution breakdown for a single collector bin."""
    bin: int
    count: int
    empirical_prob: float
    binomial_pmf: float
    gaussian_pdf: float


class SimulationStats(BaseModel):
    """Comprehensive statistical moments and goodness-of-fit metrics."""
    total_balls: int
    mean: float
    variance: float
    std_dev: float
    skewness: float
    kurtosis: float
    theoretical_mean: float
    theoretical_variance: float
    theoretical_std_dev: float
    theoretical_skewness: float
    theoretical_kurtosis: float
    chi_square_stat: float
    chi_square_p_value: float
    degrees_of_freedom: int
    clt_verified: bool


class SimulationResponse(BaseModel):
    """Complete response payload for a simulation query."""
    configuration: SimulationConfig
    stats: SimulationStats
    distribution: List[BinDistributionItem]
    elapsed_ms: float


class HealthResponse(BaseModel):
    """Health status and environment metadata."""
    status: str
    service: str
    python_version: str
    rcre_engine: str
    uptime_seconds: float
