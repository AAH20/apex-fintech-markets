"""
Apex FinTech Markets — Risk Management Module.

Provides Value-at-Risk (VaR), Conditional VaR (CVaR), Expected Shortfall (ES),
stress testing, and scenario analysis for portfolio risk quantification.

All methods support historical, parametric (variance-covariance), and Monte Carlo
approaches. Designed for production trading environments with full type safety.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FloatArray = NDArray[np.float64]
Matrix = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RiskManagementError(Exception):
    """Base exception for risk management errors."""


class InsufficientDataError(RiskManagementError):
    """Raised when insufficient data is provided for risk calculation."""


class InvalidConfidenceLevelError(RiskManagementError):
    """Raised when confidence level is outside (0, 1)."""


class InvalidHorizonError(RiskManagementError):
    """Raised when risk horizon is non-positive."""


class ModelRiskError(RiskManagementError):
    """Raised when model assumptions are violated."""


class StressScenarioNotFoundError(RiskManagementError):
    """Raised when a named stress scenario does not exist."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class VaRMethod(str, Enum):
    """Supported VaR calculation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class Distribution(str, Enum):
    """Supported return distributions for parametric VaR."""

    NORMAL = "normal"
    STUDENT_T = "student_t"


class StressScenarioType(str, Enum):
    """Types of stress scenarios."""

    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    REGULATORY = "regulatory"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VaRResult:
    """Container for VaR calculation results.

    Attributes:
        var: Value-at-Risk estimate (positive number = loss).
        confidence_level: Confidence level used (e.g., 0.99).
        horizon_days: Risk horizon in trading days.
        method: Calculation method used.
        timestamp: UTC timestamp of calculation.
        portfolio_value: Portfolio market value at calculation time.
        var_pct: VaR as percentage of portfolio value.
        metadata: Additional method-specific metadata.
    """

    var: float
    confidence_level: float
    horizon_days: int
    method: VaRMethod
    timestamp: datetime = field(default_factory=datetime.utcnow)
    portfolio_value: float = 0.0
    var_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.var < 0:
            raise ValueError("VaR must be non-negative (loss expressed as positive).")
        if not 0 < self.confidence_level < 1:
            raise InvalidConfidenceLevelError(
                f"Confidence level must be in (0, 1), got {self.confidence_level}"
            )
        if self.horizon_days <= 0:
            raise InvalidHorizonError(
                f"Horizon must be positive, got {self.horizon_days}"
            )


@dataclass(frozen=True)
class CVaRResult:
    """Container for Conditional VaR / Expected Shortfall results.

    Attributes:
        cvar: Conditional VaR estimate (expected loss beyond VaR).
        var: The corresponding VaR threshold.
        confidence_level: Confidence level used.
        horizon_days: Risk horizon in trading days.
        method: Calculation method used.
        timestamp: UTC timestamp of calculation.
        cvar_pct: CVaR as percentage of portfolio value.
    """

    cvar: float
    var: float
    confidence_level: float
    horizon_days: int
    method: VaRMethod
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cvar_pct: float = 0.0


@dataclass(frozen=True)
class StressTestResult:
    """Container for stress test results.

    Attributes:
        scenario_name: Name of the stress scenario.
        scenario_type: Type of stress scenario.
        portfolio_pnl: Portfolio P&L under the scenario.
        portfolio_return_pct: Portfolio return percentage.
        var_estimate: VaR estimate under stressed conditions.
        cvar_estimate: CVaR estimate under stressed conditions.
        per_asset_pnl: Per-asset P&L breakdown.
        timestamp: UTC timestamp of calculation.
    """

    scenario_name: str
    scenario_type: StressScenarioType
    portfolio_pnl: float
    portfolio_return_pct: float
    var_estimate: float
    cvar_estimate: float
    per_asset_pnl: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class ScenarioShock:
    """A single shock applied in scenario analysis.

    Attributes:
        factor: Risk factor name (e.g., 'equity_sp500', 'rates_10y').
        shock_pct: Shock magnitude as decimal (e.g., -0.20 for -20%).
        description: Human-readable description of the shock.
    """

    factor: str
    shock_pct: float
    description: str = ""


@dataclass(frozen=True)
class ScenarioResult:
    """Container for scenario analysis results.

    Attributes:
        scenario_name: Name of the scenario.
        portfolio_pnl: Portfolio P&L under the scenario.
        portfolio_return_pct: Portfolio return percentage.
        per_asset_pnl: Per-asset P&L breakdown.
        shocks_applied: List of shocks that were applied.
        timestamp: UTC timestamp of calculation.
    """

    scenario_name: str
    portfolio_pnl: float
    portfolio_return_pct: float
    per_asset_pnl: Dict[str, float] = field(default_factory=dict)
    shocks_applied: Tuple[ScenarioShock, ...] = field(default_factory=tuple)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Portfolio protocol
# ---------------------------------------------------------------------------
class Portfolio(Protocol):
    """Protocol for portfolio objects usable by risk calculators."""

    @property
    def market_value(self) -> float:
        """Total market value of the portfolio."""
        ...

    @property
    def positions(self) -> Dict[str, float]:
        """Mapping of asset identifier to position value."""
        ...

    def get_returns(self, lookback_days: int) -> FloatArray:
        """Return historical returns for the portfolio."""
        ...


# ---------------------------------------------------------------------------
# Base calculator
# ---------------------------------------------------------------------------
class RiskCalculator(ABC):
    """Abstract base class for risk calculators."""

    MIN_OBSERVATIONS: int = 30

    def __init__(self, confidence_level: float = 0.99, horizon_days: int = 1) -> None:
        """Initialize the risk calculator.

        Args:
            confidence_level: Confidence level for VaR/CVaR (e.g., 0.99).
            horizon_days: Risk horizon in trading days.

        Raises:
            InvalidConfidenceLevelError: If confidence level is not in (0, 1).
            InvalidHorizonError: If horizon is not positive.
        """
        if not 0 < confidence_level < 1:
            raise InvalidConfidenceLevelError(
                f"Confidence level must be in (0, 1), got {confidence_level}"
            )
        if horizon_days <= 0:
            raise InvalidHorizonError(
                f"Horizon must be positive, got {horizon_days}"
            )
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days

    @abstractmethod
    def calculate_var(self, returns: FloatArray, portfolio_value: float) -> VaRResult:
        """Calculate Value-at-Risk.

        Args:
            returns: Array of historical portfolio returns.
            portfolio_value: Current portfolio market value.

        Returns:
            VaRResult with the VaR estimate.
        """
        ...

    @abstractmethod
    def calculate_cvar(self, returns: FloatArray, portfolio_value: float) -> CVaRResult:
        """Calculate Conditional VaR / Expected Shortfall.

        Args:
            returns: Array of historical portfolio returns.
            portfolio_value: Current portfolio market value.

        Returns:
            CVaRResult with the CVaR estimate.
        """
        ...

    def _validate_returns(self, returns: FloatArray) -> None:
        """Validate returns array.

        Args:
            returns: Array to validate.

        Raises:
            InsufficientDataError: If array is too short or contains invalid values.
        """
        if len(returns) < self.MIN_OBSERVATIONS:
            raise InsufficientDataError(
                f"Need at least {self.MIN_OBSERVATIONS} observations, "
                f"got {len(returns)}"
            )
        if not np.all(np.isfinite(returns)):
            raise ModelRiskError("Returns contain NaN or infinite values.")

    def _scale_to_horizon(self, daily_estimate: float) -> float:
        """Scale a daily risk estimate to the target horizon using square-root-of-time.

        Args:
            daily_estimate: Risk estimate for 1-day horizon.

        Returns:
            Risk estimate scaled to self.horizon_days.
        """
        return daily_estimate * np.sqrt(self.horizon_days)


# ---------------------------------------------------------------------------
# Historical VaR / CVaR
# ---------------------------------------------------------------------------
class HistoricalVaR(RiskCalculator):
    """Historical simulation VaR and CVaR calculator.

    Uses empirical quantiles of historical returns. No distributional assumptions.
    """

    def calculate_var(self, returns: FloatArray, portfolio_value: float) -> VaRResult:
        """Calculate historical VaR.

        Args:
            returns: Array of historical portfolio returns.
            portfolio_value: Current portfolio market value.

        Returns:
            VaRResult with the historical VaR estimate.
        """
        self._validate_returns(returns)
        var_pct = float(np.percentile(returns, (1 - self.confidence_level) * 100))
        var = abs(var_pct) * portfolio_value
        var_scaled = self._scale_to_horizon(var)
        return VaRResult(
            var=var_scaled,
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            method=VaRMethod.HISTORICAL,
            portfolio_value=portfolio_value,
            var_pct=var_scaled / portfolio_value if portfolio_value > 0 else 0.0,
            metadata={"num_observations": len(returns)},
        )

    def calculate_cvar(self, returns: FloatArray, portfolio_value: float) -> CVaRResult:
        """Calculate historical CVaR (Expected Shortfall).

        Args:
            returns: Array of historical portfolio returns.
            portfolio_value: Current portfolio market value.

        Returns:
            CVaRResult with the historical CVaR estimate.
        """
        self._validate_returns(returns)
        var_pct = float(np.percentile(returns, (1 - self.confidence_level) * 100))
        tail_returns = returns[returns <= var_pct]
        if len(tail_returns) == 0:
            cvar_pct = var_pct
        else:
            cvar_pct = float(np.mean(tail_returns))
        cvar = abs(cvar_pct) * portfolio_value
        cvar_scaled = self._scale_to_horizon(cvar)
        var_result = self.calculate_var(returns, portfolio_value)
        return CVaRResult(
            cvar=cvar_scaled,
            var=var_result.var,
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            method=VaRMethod.HISTORICAL,
            cvar_pct=cvar_scaled / portfolio_value if portfolio_value > 0 else 0.0,
        )


# ---------------------------------------------------------------------------
# Parametric VaR / CVaR
# ---------------------------------------------------------------------------
class ParametricVaR(RiskCalculator):
    """Parametric (variance-covariance) VaR and CVaR calculator.

    Assumes returns follow a specified distribution (normal or Student-t).
    """

    def __init__(
        self,
        confidence_level: float = 0.99,
        horizon_days: int = 1,
        distribution: Distribution = Distribution.NORMAL,
        degrees_of_freedom: float = 5.0,
    ) -> None:
        """Initialize parametric VaR calculator.

        Args:
            confidence_level: Confidence level for VaR/CVaR.
            horizon_days: Risk horizon in trading days.
            distribution: Return distribution assumption.
            degrees_of_freedom: Degrees of freedom for Student-t distribution.

        Raises:
            ValueError: If degrees_of_freedom <= 2 for Student-t.
        """
        super().__init__(confidence_level, horizon_days)
        self.distribution = distribution
        self.degrees_of_freedom = degrees_of_freedom
        if distribution == Distribution.STUDENT_T and degrees_of_freedom <= 2:
            raise ValueError(
                "Student-t distribution requires degrees_of_freedom > 2."
            )

    def calculate_var(self, returns: FloatArray, portfolio_value: float) -> VaRResult:
        """Calculate parametric VaR.

        Args:
            returns: Array of historical portfolio returns.
            portfolio_value: Current portfolio market value.

        Returns:
            VaRResult with the parametric VaR estimate.
        """
        self._validate_returns(returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))

        if self.distribution == Distribution.NORMAL:
            from scipy import stats

            z_score = float(stats.norm.ppf(1 - self.confidence_level))
            var_pct = -(mu - z_score * sigma)
        else:
            from scipy import stats

            t_score = float(stats.t.ppf(1 - self.confidence_level, self.degrees_of_freedom))
            # Adjust for heavier tails
            var_pct = -(mu - t_score * sigma * np.sqrt(
                (self.degrees_of_freedom - 2) / self.degrees_of_freedom
            ))

        var = abs(var_pct) * portfolio_value
        var_scaled = self._scale_to_horizon(var)
        return VaRResult(
            var=var_scaled,
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            method=VaRMethod.PARAMETRIC,
            portfolio_value=portfolio_value,
            var_pct=var_scaled / portfolio_value if portfolio_value > 0 else 0.0,
            metadata={
                "mean_return": mu,
                "volatility": sigma,
                "distribution": self.distribution.value,
                "degrees_of_freedom": self.degrees_of_freedom
                if self.distribution == Distribution.STUDENT_T
                else None,
            },
        )

    def calculate_cvar(self, returns: FloatArray, portfolio_value: float) -> CVaRResult:
        """Calculate parametric CVaR.

        Args:
            returns: Array of historical portfolio returns.
            portfolio_value: Current portfolio market value.

        Returns:
            CVaRResult with the parametric CVaR estimate.
        """
        self._validate_returns(returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        var_result = self.calculate_var(returns, portfolio_value)

        if self.distribution == Distribution.NORMAL:
            from scipy import stats

            z = float(stats.norm.ppf(1 - self.confidence_level))
            # ES for normal: mu - sigma * phi(z) / (1 - alpha)
            pdf_z = float(stats.norm.pdf(z))
            cvar_pct = -(mu - sigma * pdf_z / (1 - self.confidence_level))
        else:
            from scipy import stats

            t_score = float(stats.t.ppf(1 - self.confidence_level, self.degrees_of_freedom))
            pdf_t = float(stats.t.pdf(t_score, self.degrees_of_freedom))
            cvar_pct = -(
                mu
                - sigma
                * pdf_t
                * (self.degrees_of_freedom + t_score**2)
                / ((self.degrees_of_freedom - 1) * (1 - self.confidence_level))
            )

        cvar = abs(cvar_pct) * portfolio_value
        cvar_scaled = self._scale_to_horizon(cvar)
        return CVaRResult(
            cvar=cvar_scaled,
            var=var_result.var,
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            method=VaRMethod.PARAMETRIC,
            cvar_pct=cvar_scaled / portfolio_value if portfolio_value > 0 else 0.0,
        )


# ---------------------------------------------------------------------------
# Monte Carlo VaR / CVaR
# ---------------------------------------------------------------------------
class MonteCarloVaR(RiskCalculator):
    """Monte Carlo simulation VaR and CVaR calculator.

    Simulates future portfolio returns using a specified stochastic model.
    """

    def __init__(
        self,
        confidence_level: float = 0.99,
        horizon_days: int = 1,
        num_simulations: int = 100_000,
        distribution: Distribution = Distribution.NORMAL,
        degrees_of_freedom: float = 5.0,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize Monte Carlo VaR calculator.

        Args:
            confidence_level: Confidence level for VaR/CVaR.
            horizon_days: Risk horizon in trading days.
            num_simulations: Number of Monte Carlo paths.
            distribution: Return distribution for simulations.
            degrees_of_freedom: Degrees of freedom for Student-t.
            random_seed: Random seed for reproducibility.

        Raises:
            ValueError: If num_simulations is not positive.
        """
        super().__init__(confidence_level, horizon_days)
        if num_simulations <= 0:
            raise ValueError(
                f"Number of simulations must be positive, got {num_simulations}"
            )
        self.num_simulations = num_simulations
        self.distribution = distribution
        self.degrees_of_freedom = degrees_of_freedom
        self.random_seed = random_seed

    def calculate_var(self, returns: FloatArray, portfolio_value: float) -> VaRResult:
        """Calculate Monte Carlo VaR.

        Args:
            returns: Array of historical portfolio returns (used for parameter estimation).
            portfolio_value: Current portfolio market value.

        Returns:
            VaRResult with the Monte Carlo VaR estimate.
        """
        self._validate_returns(returns)
        simulated_returns = self._simulate_returns(returns)
        var_pct = float(
            np.percentile(simulated_returns, (1 - self.confidence_level) * 100)
        )
        var = abs(var_pct) * portfolio_value
        var_scaled = self._scale_to_horizon(var)
        return VaRResult(
            var=var_scaled,
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            method=VaRMethod.MONTE_CARLO,
            portfolio_value=portfolio_value,
            var_pct=var_scaled / portfolio_value if portfolio_value > 0 else 0.0,
            metadata={
                "num_simulations": self.num_simulations,
                "distribution": self.distribution.value,
                "random_seed": self.random_seed,
            },
        )

    def calculate_cvar(self, returns: FloatArray, portfolio_value: float) -> CVaRResult:
        """Calculate Monte Carlo CVaR.

        Args:
            returns: Array of historical portfolio returns.
            portfolio_value: Current portfolio market value.

        Returns:
            CVaRResult with the Monte Carlo CVaR estimate.
        """
        self._validate_returns(returns)
        simulated_returns = self._simulate_returns(returns)
        var_pct = float(
            np.percentile(simulated_returns, (1 - self.confidence_level) * 100)
        )
        tail_returns = simulated_returns[simulated_returns <= var_pct]
        if len(tail_returns) == 0:
            cvar_pct = var_pct
        else:
            cvar_pct = float(np.mean(tail_returns))
        cvar = abs(cvar_pct) * portfolio_value
        cvar_scaled = self._scale_to_horizon(cvar)
        var_result = self.calculate_var(returns, portfolio_value)
        return CVaRResult(
            cvar=cvar_scaled,
            var=var_result.var,
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            method=VaRMethod.MONTE_CARLO,
            cvar_pct=cvar_scaled / portfolio_value if portfolio_value > 0 else 0.0,
        )

    def _simulate_returns(self, historical_returns: FloatArray) -> FloatArray:
        """Simulate portfolio returns using the configured distribution.

        Args:
            historical_returns: Historical returns for parameter estimation.

        Returns:
            Array of simulated returns.
        """
        rng = np.random.default_rng(self.random_seed)
        mu = float(np.mean(historical_returns))
        sigma = float(np.std(historical_returns, ddof=1))

        if self.distribution == Distribution.NORMAL:
            simulated = rng.normal(mu, sigma, self.num_simulations)
        else:
            simulated = rng.standard_t(self.degrees_of_freedom, self.num_simulations)
            simulated = mu + simulated * sigma * np.sqrt(
                (self.degrees_of_freedom - 2) / self.degrees_of_freedom
            )
        return simulated


# ---------------------------------------------------------------------------
# Stress Testing
# ---------------------------------------------------------------------------
class StressTester:
    """Stress testing engine for portfolio risk assessment.

    Applies predefined or custom stress scenarios to a portfolio and
    calculates resulting P&L, VaR, and CVaR under stressed conditions.
    """

    def __init__(self) -> None:
        """Initialize the stress tester with an empty scenario registry."""
        self._scenarios: Dict[str, Tuple[StressScenarioType, Tuple[ScenarioShock, ...]]] = {}
        self._setup_default_scenarios()

    def register_scenario(
        self,
        name: str,
        scenario_type: StressScenarioType,
        shocks: Sequence[ScenarioShock],
    ) -> None:
        """Register a new stress scenario.

        Args:
            name: Unique scenario name.
            scenario_type: Type of stress scenario.
            shocks: Sequence of shocks to apply.

        Raises:
            ValueError: If a scenario with the same name already exists.
        """
        if name in self._scenarios:
            raise ValueError(f"Scenario '{name}' already exists.")
        self._scenarios[name] = (scenario_type, tuple(shocks))
        logger.info("Registered stress scenario: %s", name)

    def remove_scenario(self, name: str) -> None:
        """Remove a registered stress scenario.

        Args:
            name: Scenario name to remove.

        Raises:
            StressScenarioNotFoundError: If scenario does not exist.
        """
        if name not in self._scenarios:
            raise StressScenarioNotFoundError(f"Scenario '{name}' not found.")
        del self._scenarios[name]

    def run_stress_test(
        self,
        portfolio: Portfolio,
        scenario_name: str,
        returns: FloatArray,
        confidence_level: float = 0.99,
    ) -> StressTestResult:
        """Run a stress test on a portfolio.

        Args:
            portfolio: Portfolio to stress test.
            scenario_name: Name of the registered scenario.
            returns: Historical returns for VaR/CVaR calculation.
            confidence_level: Confidence level for risk metrics.

        Returns:
            StressTestResult with stress test results.

        Raises:
            StressScenarioNotFoundError: If scenario does not exist.
        """
        if scenario_name not in self._scenarios:
            raise StressScenarioNotFoundError(
                f"Scenario '{scenario_name}' not found."
            )
        scenario_type, shocks = self._scenarios[scenario_name]

        # Calculate stressed portfolio P&L
        per_asset_pnl: Dict[str, float] = {}
        total_pnl = 0.0
        for asset, value in portfolio.positions.items():
            # Apply the first shock that matches (simplified model)
            asset_shock = 0.0
            for shock in shocks:
                # In production, map assets to risk factors properly
                asset_shock = shock.shock_pct
                break
            pnl = value * asset_shock
            per_asset_pnl[asset] = pnl
            total_pnl += pnl

        portfolio_return_pct = (
            total_pnl / portfolio.market_value if portfolio.market_value > 0 else 0.0
        )

        # Calculate stressed VaR/CVaR using historical returns scaled by shock
        stressed_returns = returns * (1 + sum(s.shock_pct for s in shocks))
        hist_var = HistoricalVaR(confidence_level=confidence_level)
        var_result = hist_var.calculate_var(stressed_returns, portfolio.market_value)
        cvar_result = hist_var.calculate_cvar(stressed_returns, portfolio.market_value)

        return StressTestResult(
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            portfolio_pnl=total_pnl,
            portfolio_return_pct=portfolio_return_pct,
            var_estimate=var_result.var,
            cvar_estimate=cvar_result.cvar,
            per_asset_pnl=per_asset_pnl,
        )

    def run_all_scenarios(
        self,
        portfolio: Portfolio,
        returns: FloatArray,
        confidence_level: float = 0.99,
    ) -> List[StressTestResult]:
        """Run all registered stress scenarios.

        Args:
            portfolio: Portfolio to stress test.
            returns: Historical returns.
            confidence_level: Confidence level for risk metrics.

        Returns:
            List of StressTestResult for all scenarios.
        """
        results: List[StressTestResult] = []
        for name in self._scenarios:
            result = self.run_stress_test(portfolio, name, returns, confidence_level)
            results.append(result)
        return results

    def _setup_default_scenarios(self) -> None:
        """Register default regulatory and historical stress scenarios."""
        # 2008 Financial Crisis
        self.register_scenario(
            "financial_crisis_2008",
            StressScenarioType.HISTORICAL,
            [
                ScenarioShock("equity_sp500", -0.40, "S&P 500 crash"),
                ScenarioShock("credit_spread", 0.05, "Credit spread widening"),
                ScenarioShock("rates_10y", -0.02, "Flight to quality"),
            ],
        )

        # COVID-19 Market Crash (March 2020)
        self.register_scenario(
            "covid_crash_2020",
            StressScenarioType.HISTORICAL,
            [
                ScenarioShock("equity_sp500", -0.34, "COVID equity crash"),
                ScenarioShock("volatility_vix", 60.0, "VIX spike"),
                ScenarioShock("oil", -0.50, "Oil price collapse"),
            ],
        )

        # Interest Rate Shock
        self.register_scenario(
            "rate_shock_200bp",
            StressScenarioType.HYPOTHETICAL,
            [
                ScenarioShock("rates_10y", 0.02, "200bp rate rise"),
                ScenarioShock("equity_sp500", -0.10, "Equity de-rating"),
            ],
        )

        # Liquidity Crisis
        self.register_scenario(
            "liquidity_crisis",
            StressScenarioType.HYPOTHETICAL,
            [
                ScenarioShock("bid_ask_spread", 0.05, "Spread widening"),
                ScenarioShock("equity_sp500", -0.20, "Forced selling"),
            ],
        )


# ---------------------------------------------------------------------------
# Scenario Analysis
# ---------------------------------------------------------------------------
class ScenarioAnalyzer:
    """Scenario analysis engine for what-if portfolio evaluation.

    Applies user-defined shocks to portfolio positions and calculates
    resulting P&L and risk metrics.
    """

    def __init__(self) -> None:
        """Initialize the scenario analyzer."""
        self._scenarios: Dict[str, Tuple[ScenarioShock, ...]] = {}

    def register_scenario(self, name: str, shocks: Sequence[ScenarioShock]) -> None:
        """Register a custom scenario.

        Args:
            name: Unique scenario name.
            shocks: Sequence of shocks to apply.

        Raises:
            ValueError: If a scenario with the same name already exists.
        """
        if name in self._scenarios:
            raise ValueError(f"Scenario '{name}' already exists.")
        self._scenarios[name] = tuple(shocks)

    def run_scenario(
        self,
        portfolio: Portfolio,
        scenario_name: str,
    ) -> ScenarioResult:
        """Run a scenario analysis.

        Args:
            portfolio: Portfolio to analyze.
            scenario_name: Name of the registered scenario.

        Returns:
            ScenarioResult with scenario analysis results.

        Raises:
            StressScenarioNotFoundError: If scenario does not exist.
        """
        if scenario_name not in self._scenarios:
            raise StressScenarioNotFoundError(
                f"Scenario '{scenario_name}' not found."
            )
        shocks = self._scenarios[scenario_name]

        per_asset_pnl: Dict[str, float] = {}
        total_pnl = 0.0
        for asset, value in portfolio.positions.items():
            asset_shock = 0.0
            for shock in shocks:
                asset_shock = shock.shock_pct
                break
            pnl = value * asset_shock
            per_asset_pnl[asset] = pnl
            total_pnl += pnl

        portfolio_return_pct = (
            total_pnl / portfolio.market_value if portfolio.market_value > 0 else 0.0
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            portfolio_pnl=total_pnl,
            portfolio_return_pct=portfolio_return_pct,
            per_asset_pnl=per_asset_pnl,
            shocks_applied=shocks,
        )

    def run_custom_scenario(
        self,
        portfolio: Portfolio,
        shocks: Sequence[ScenarioShock],
        scenario_name: str = "custom",
    ) -> ScenarioResult:
        """Run a one-off custom scenario without registering it.

        Args:
            portfolio: Portfolio to analyze.
            shocks: Shocks to apply.
            scenario_name: Name for the result.

        Returns:
            ScenarioResult with scenario analysis results.
        """
        per_asset_pnl: Dict[str, float] = {}
        total_pnl = 0.0
        for asset, value in portfolio.positions.items():
            asset_shock = 0.0
            for shock in shocks:
                asset_shock = shock.shock_pct
                break
            pnl = value * asset_shock
            per_asset_pnl[asset] = pnl
            total_pnl += pnl

        portfolio_return_pct = (
            total_pnl / portfolio.market_value if portfolio.market_value > 0 else 0.0
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            portfolio_pnl=total_pnl,
            portfolio_return_pct=portfolio_return_pct,
            per_asset_pnl=per_asset_pnl,
            shocks_applied=tuple(shocks),
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_var_calculator(
    method: VaRMethod,
    confidence_level: float = 0.99,
    horizon_days: int = 1,
    **kwargs: Any,
) -> RiskCalculator:
    """Factory function to create a VaR calculator.

    Args:
        method: VaR calculation method.
        confidence_level: Confidence level.
        horizon_days: Risk horizon in trading days.
        **kwargs: Additional method-specific parameters.

    Returns:
        Configured RiskCalculator instance.

    Raises:
        ValueError: If method is not supported.
    """
    if method == VaRMethod.HISTORICAL:
        return HistoricalVaR(confidence_level, horizon_days)
    elif method == VaRMethod.PARAMETRIC:
        return ParametricVaR(
            confidence_level,
            horizon_days,
            distribution=kwargs.get("distribution", Distribution.NORMAL),
            degrees_of_freedom=kwargs.get("degrees_of_freedom", 5.0),
        )
    elif method == VaRMethod.MONTE_CARLO:
        return MonteCarloVaR(
            confidence_level,
            horizon_days,
            num_simulations=kwargs.get("num_simulations", 100_000),
            distribution=kwargs.get("distribution", Distribution.NORMAL),
            degrees_of_freedom=kwargs.get("degrees_of_freedom", 5.0),
            random_seed=kwargs.get("random_seed"),
        )
    else:
        raise ValueError(f"Unsupported VaR method: {method}")
