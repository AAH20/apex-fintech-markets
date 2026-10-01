"""
Apex FinTech Markets — T0 Market Making Systems.

Implements three foundational market-making models:

1. **Avellaneda-Stoikov (AS)** — Optimal quoting with inventory management.
   Derives reservation prices and optimal spreads that balance profit against
   inventory risk and time decay.

2. **Almgren-Chriss (AC)** — Optimal execution scheduling.
   Minimizes the trade-off between market impact and timing risk for
   large-order liquidation/acquisition.

3. **Glosten-Milgrom (GM)** — Adverse selection and information-based quoting.
   Models how market makers adjust quotes in response to informed trading.

All models use continuous-time stochastic frameworks with explicit
parameter validation and comprehensive error handling.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class MarketMakingError(Exception):
    """Base exception for market-making errors."""


class ParameterValidationError(MarketMakingError):
    """Raised when model parameters fail validation."""


class InventoryLimitError(MarketMakingError):
    """Raised when inventory exceeds configured limits."""


class QuoteError(MarketMakingError):
    """Raised when quote computation fails."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""

    LIMIT = "limit"
    MARKET = "market"


class ExecutionUrgency(Enum):
    """Execution urgency level for Almgren-Chriss."""

    LOW = "low"        # Patient — minimize impact
    MEDIUM = "medium"  # Balanced
    HIGH = "high"      # Aggressive — minimize timing risk


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketState:
    """Snapshot of current market conditions.

    Attributes:
        mid_price: Current mid-price of the instrument.
        bid: Best bid price.
        ask: Best ask price.
        bid_size: Size available at best bid.
        ask_size: Size available at best ask.
        volatility: Annualized volatility (e.g., 0.20 for 20%).
        timestamp: Unix timestamp of the observation.
    """

    mid_price: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    volatility: float
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.mid_price <= 0:
            raise ParameterValidationError(
                f"mid_price must be positive, got {self.mid_price}"
            )
        if self.bid <= 0 or self.ask <= 0:
            raise ParameterValidationError(
                f"bid/ask must be positive, got bid={self.bid}, ask={self.ask}"
            )
        if self.bid > self.ask:
            raise ParameterValidationError(
                f"Crossed market: bid={self.bid} > ask={self.ask}"
            )
        if self.volatility < 0:
            raise ParameterValidationError(
                f"volatility must be non-negative, got {self.volatility}"
            )

    @property
    def spread(self) -> float:
        """Current bid-ask spread."""
        return self.ask - self.bid

    @property
    def half_spread(self) -> float:
        """Half-spread (distance from mid to best quote)."""
        return self.spread / 2.0


@dataclass(frozen=True)
class Quote:
    """A two-sided quote.

    Attributes:
        bid_price: Price at which we are willing to buy.
        ask_price: Price at which we are willing to sell.
        bid_size: Size of buy order.
        ask_size: Size of sell order.
        side: Primary side of the quote.
    """

    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    side: Side = Side.BUY

    def __post_init__(self) -> None:
        if self.bid_price <= 0 or self.ask_price <= 0:
            raise ParameterValidationError(
                f"Quote prices must be positive, got bid={self.bid_price}, "
                f"ask={self.ask_price}"
            )
        if self.bid_price >= self.ask_price:
            raise ParameterValidationError(
                f"Invalid quote: bid={self.bid_price} >= ask={self.ask_price}"
            )
        if self.bid_size < 0 or self.ask_size < 0:
            raise ParameterValidationError(
                f"Quote sizes must be non-negative, got bid_size={self.bid_size}, "
                f"ask_size={self.ask_size}"
            )

    @property
    def spread(self) -> float:
        """Spread of this quote."""
        return self.ask_price - self.bid_price

    @property
    def mid(self) -> float:
        """Mid-price of this quote."""
        return (self.bid_price + self.ask_price) / 2.0


@dataclass(frozen=True)
class InventoryPosition:
    """Current inventory position.

    Attributes:
        quantity: Signed quantity held (positive = long, negative = short).
        average_entry_price: Volume-weighted average entry price.
        unrealized_pnl: Unrealized profit/loss.
        max_inventory: Maximum allowed absolute inventory.
    """

    quantity: float = 0.0
    average_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    max_inventory: float = 100.0

    def __post_init__(self) -> None:
        if self.max_inventory <= 0:
            raise ParameterValidationError(
                f"max_inventory must be positive, got {self.max_inventory}"
            )
        if abs(self.quantity) > self.max_inventory:
            raise InventoryLimitError(
                f"Inventory |{self.quantity}| exceeds max {self.max_inventory}"
            )

    @property
    def is_long(self) -> bool:
        """True if net long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if net short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """True if flat."""
        return self.quantity == 0

    @property
    def utilization(self) -> float:
        """Fraction of max inventory utilized."""
        return abs(self.quantity) / self.max_inventory if self.max_inventory > 0 else 0.0


@dataclass(frozen=True)
class ASParameters:
    """Parameters for the Avellaneda-Stoikov model.

    Attributes:
        gamma: Risk aversion coefficient (higher = more risk-averse).
        sigma: Volatility of the underlying (annualized).
        k: Order arrival rate parameter (higher = more aggressive quoting).
        A: Order arrival rate scale factor.
        T: Terminal time in years.
        max_inventory: Maximum absolute inventory allowed.
        min_spread: Minimum spread to quote (in price units).
        max_spread: Maximum spread to quote (in price units).
    """

    gamma: float = 0.1
    sigma: float = 0.2
    k: float = 1.5
    A: float = 0.1
    T: float = 1.0 / 252.0  # One trading day
    max_inventory: float = 100.0
    min_spread: float = 0.01
    max_spread: float = 10.0

    def __post_init__(self) -> None:
        if self.gamma < 0:
            raise ParameterValidationError(f"gamma must be >= 0, got {self.gamma}")
        if self.sigma <= 0:
            raise ParameterValidationError(f"sigma must be > 0, got {self.sigma}")
        if self.k <= 0:
            raise ParameterValidationError(f"k must be > 0, got {self.k}")
        if self.A <= 0:
            raise ParameterValidationError(f"A must be > 0, got {self.A}")
        if self.T <= 0:
            raise ParameterValidationError(f"T must be > 0, got {self.T}")
        if self.max_inventory <= 0:
            raise ParameterValidationError(
                f"max_inventory must be > 0, got {self.max_inventory}"
            )
        if self.min_spread < 0:
            raise ParameterValidationError(
                f"min_spread must be >= 0, got {self.min_spread}"
            )
        if self.max_spread <= self.min_spread:
            raise ParameterValidationError(
                f"max_spread ({self.max_spread}) must exceed min_spread ({self.min_spread})"
            )


@dataclass(frozen=True)
class ACParameters:
    """Parameters for the Almgren-Chriss optimal execution model.

    Attributes:
        X: Total quantity to execute.
        T: Total execution time in years.
        sigma: Volatility of the underlying (annualized).
        eta: Temporary market impact coefficient.
        gamma_perm: Permanent market impact coefficient.
        epsilon: Temporary impact exponent (typically 0.5-1.0).
        urgency: Execution urgency level.
        num_slices: Number of execution slices.
        risk_aversion: Risk aversion parameter (lambda).
    """

    X: float = 10000.0
    T: float = 1.0 / 252.0
    sigma: float = 0.2
    eta: float = 0.01
    gamma_perm: float = 0.001
    epsilon: float = 0.5
    urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    num_slices: int = 10
    risk_aversion: float = 1e-6

    def __post_init__(self) -> None:
        if self.X <= 0:
            raise ParameterValidationError(f"X must be > 0, got {self.X}")
        if self.T <= 0:
            raise ParameterValidationError(f"T must be > 0, got {self.T}")
        if self.sigma < 0:
            raise ParameterValidationError(f"sigma must be >= 0, got {self.sigma}")
        if self.eta < 0:
            raise ParameterValidationError(f"eta must be >= 0, got {self.eta}")
        if self.gamma_perm < 0:
            raise ParameterValidationError(
                f"gamma_perm must be >= 0, got {self.gamma_perm}"
            )
        if self.epsilon <= 0:
            raise ParameterValidationError(
                f"epsilon must be > 0, got {self.epsilon}"
            )
        if self.num_slices <= 0:
            raise ParameterValidationError(
                f"num_slices must be > 0, got {self.num_slices}"
            )
        if self.risk_aversion < 0:
            raise ParameterValidationError(
                f"risk_aversion must be >= 0, got {self.risk_aversion}"
            )


@dataclass(frozen=True)
class GMParameters:
    """Parameters for the Glosten-Milgrom model.

    Attributes:
        mu: Probability of an informed trader arriving.
        sigma_v: Standard deviation of the asset's true value.
        sigma_u: Standard deviation of noise trader volume.
        alpha: Fraction of traders who are informed.
        max_spread: Maximum spread to quote.
        learning_rate: Rate at which beliefs are updated.
    """

    mu: float = 0.5
    sigma_v: float = 1.0
    sigma_u: float = 0.5
    alpha: float = 0.3
    max_spread: float = 5.0
    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        if not 0 <= self.mu <= 1:
            raise ParameterValidationError(f"mu must be in [0,1], got {self.mu}")
        if self.sigma_v <= 0:
            raise ParameterValidationError(
                f"sigma_v must be > 0, got {self.sigma_v}"
            )
        if self.sigma_u <= 0:
            raise ParameterValidationError(
                f"sigma_u must be > 0, got {self.sigma_u}"
            )
        if not 0 <= self.alpha <= 1:
            raise ParameterValidationError(
                f"alpha must be in [0,1], got {self.alpha}"
            )
        if self.max_spread <= 0:
            raise ParameterValidationError(
                f"max_spread must be > 0, got {self.max_spread}"
            )
        if not 0 < self.learning_rate <= 1:
            raise ParameterValidationError(
                f"learning_rate must be in (0,1], got {self.learning_rate}"
            )


@dataclass(frozen=True)
class ExecutionSlice:
    """A single execution slice in an Almgren-Chriss schedule.

    Attributes:
        time: Time at which to execute (in years from start).
        quantity: Quantity to execute at this time.
        price: Expected execution price.
    """

    time: float
    quantity: float
    price: float


@dataclass(frozen=True)
class ExecutionSchedule:
    """Complete execution schedule from Almgren-Chriss.

    Attributes:
        slices: List of execution slices.
        total_cost: Expected total execution cost.
        expected_shortfall: Expected implementation shortfall.
        risk_measure: Variance of execution cost.
    """

    slices: list[ExecutionSlice]
    total_cost: float
    expected_shortfall: float
    risk_measure: float


@dataclass(frozen=True)
class BeliefState:
    """Belief state for Glosten-Milgrom model.

    Attributes:
        v_high: Current estimate of high true value.
        v_low: Current estimate of low true value.
        p_informed: Posterior probability of informed trading.
        spread: Current quoted spread.
    """

    v_high: float
    v_low: float
    p_informed: float
    spread: float

    def __post_init__(self) -> None:
        if self.v_high < self.v_low:
            raise ParameterValidationError(
                f"v_high ({self.v_high}) must be >= v_low ({self.v_low})"
            )
        if not 0 <= self.p_informed <= 1:
            raise ParameterValidationError(
                f"p_informed must be in [0,1], got {self.p_informed}"
            )
        if self.spread < 0:
            raise ParameterValidationError(
                f"spread must be >= 0, got {self.spread}"
            )


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class QuoteGenerator(Protocol):
    """Protocol for quote generators."""

    def generate_quote(
        self, market_state: MarketState, inventory: InventoryPosition
    ) -> Quote:
        """Generate a two-sided quote given market state and inventory."""
        ...


@runtime_checkable
class InventoryManager(Protocol):
    """Protocol for inventory management."""

    def update_inventory(self, quantity: float, price: float) -> InventoryPosition:
        """Update inventory after a fill."""
        ...

    def get_position(self) -> InventoryPosition:
        """Get current inventory position."""
        ...


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov Model
# ---------------------------------------------------------------------------

class AvellanedaStoikov:
    """Avellaneda-Stoikov optimal quoting model with inventory management.

    The AS model derives optimal bid/ask quotes that maximize expected
    exponential utility of terminal wealth, accounting for inventory risk.

    Reservation price: r(t) = s(t) - q * gamma * sigma^2 * (T - t)
    Optimal spread: delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    where:
        s(t) = current mid-price
        q = current inventory
        gamma = risk aversion
        sigma = volatility
        T - t = time remaining
        k = order arrival rate parameter
    """

    def __init__(self, params: Optional[ASParameters] = None) -> None:
        """Initialize the AS model.

        Args:
            params: Model parameters. Uses defaults if None.
        """
        self.params = params or ASParameters()
        self._inventory = InventoryPosition(max_inventory=self.params.max_inventory)
        self._quote_history: list[Quote] = []
        self._reservation_price_history: list[float] = []

    @property
    def inventory(self) -> InventoryPosition:
        """Current inventory position."""
        return self._inventory

    def reservation_price(self, market_state: MarketState, time_remaining: float) -> float:
        """Compute the reservation price.

        The reservation price is the price at which the market maker is
        indifferent between holding current inventory and executing.

        Args:
            market_state: Current market state.
            time_remaining: Time remaining until horizon (in years).

        Returns:
            Reservation price.
        """
        if time_remaining < 0:
            raise ParameterValidationError(
                f"time_remaining must be >= 0, got {time_remaining}"
            )

        q = self._inventory.quantity
        gamma = self.params.gamma
        sigma = self.params.sigma

        # Reservation price adjusts mid-price by inventory risk premium
        r = market_state.mid_price - q * gamma * sigma**2 * time_remaining
        self._reservation_price_history.append(r)
        return r

    def optimal_spread(self, time_remaining: float) -> float:
        """Compute the optimal half-spread.

        Args:
            time_remaining: Time remaining until horizon (in years).

        Returns:
            Optimal half-spread (distance from reservation price to quote).
        """
        if time_remaining < 0:
            raise ParameterValidationError(
                f"time_remaining must be >= 0, got {time_remaining}"
            )

        gamma = self.params.gamma
        sigma = self.params.sigma
        k = self.params.k

        # Spread = inventory risk component + adverse selection component
        spread = gamma * sigma**2 * time_remaining + (2.0 / gamma) * math.log(1.0 + gamma / k)

        # Clamp to configured bounds
        spread = max(self.params.min_spread, min(self.params.max_spread, spread))
        return spread

    def generate_quote(
        self, market_state: MarketState, time_remaining: Optional[float] = None
    ) -> Quote:
        """Generate optimal two-sided quote.

        Args:
            market_state: Current market state.
            time_remaining: Time remaining until horizon. Uses params.T if None.

        Returns:
            Optimal Quote.
        """
        if time_remaining is None:
            time_remaining = self.params.T

        r = self.reservation_price(market_state, time_remaining)
        delta = self.optimal_spread(time_remaining)

        bid_price = r - delta
        ask_price = r + delta

        # Ensure quotes are positive
        bid_price = max(bid_price, 0.01)
        ask_price = max(ask_price, bid_price + self.params.min_spread)

        # Size quotes based on inventory skew
        base_size = 1.0
        if self._inventory.is_long:
            # Skew: larger bid to reduce long inventory
            bid_size = base_size * (1.0 + self._inventory.utilization)
            ask_size = base_size * (1.0 - 0.5 * self._inventory.utilization)
        elif self._inventory.is_short:
            # Skew: larger ask to reduce short inventory
            ask_size = base_size * (1.0 + self._inventory.utilization)
            bid_size = base_size * (1.0 - 0.5 * self._inventory.utilization)
        else:
            bid_size = base_size
            ask_size = base_size

        quote = Quote(
            bid_price=round(bid_price, 4),
            ask_price=round(ask_price, 4),
            bid_size=round(bid_size, 4),
            ask_size=round(ask_size, 4),
        )
        self._quote_history.append(quote)
        return quote

    def update_inventory(self, quantity: float, price: float) -> InventoryPosition:
        """Update inventory after a fill.

        Args:
            quantity: Signed quantity filled (positive = bought, negative = sold).
            price: Fill price.

        Returns:
            Updated inventory position.
        """
        new_qty = self._inventory.quantity + quantity

        if abs(new_qty) > self.params.max_inventory:
            raise InventoryLimitError(
                f"Fill would exceed max inventory: |{new_qty}| > {self.params.max_inventory}"
            )

        # Update average entry price
        if self._inventory.quantity == 0 or (
            (self._inventory.quantity > 0 and quantity > 0)
            or (self._inventory.quantity < 0 and quantity < 0)
        ):
            # Adding to position
            total_cost = (
                self._inventory.quantity * self._inventory.average_entry_price
                + quantity * price
            )
            avg_price = total_cost / new_qty if new_qty != 0 else 0.0
        else:
            # Reducing or flipping position
            avg_price = price if abs(new_qty) > 0 else 0.0

        self._inventory = InventoryPosition(
            quantity=new_qty,
            average_entry_price=avg_price,
            max_inventory=self.params.max_inventory,
        )
        return self._inventory

    def get_position(self) -> InventoryPosition:
        """Get current inventory position."""
        return self._inventory

    def reset_inventory(self) -> None:
        """Reset inventory to flat."""
        self._inventory = InventoryPosition(max_inventory=self.params.max_inventory)
        self._quote_history.clear()
        self._reservation_price_history.clear()

    def expected_profit_rate(self, market_state: MarketState, time_remaining: float) -> float:
        """Compute expected profit rate per unit time.

        Args:
            market_state: Current market state.
            time_remaining: Time remaining until horizon.

        Returns:
            Expected profit rate.
        """
        delta = self.optimal_spread(time_remaining)
        k = self.params.k
        A = self.params.A

        # Expected profit = order_rate * (half_spread + inventory_adjustment)
        # Order arrival rate: lambda = A * exp(-k * delta)
        order_rate = A * math.exp(-k * delta)
        profit_per_order = delta  # Half-spread captured per fill

        return order_rate * profit_per_order


# ---------------------------------------------------------------------------
# Almgren-Chriss Optimal Execution
# ---------------------------------------------------------------------------

class AlmgrenChriss:
    """Almgren-Chriss optimal execution scheduling model.

    The AC model finds the optimal trading trajectory that minimizes:

        E[Cost] + lambda * Var[Cost]

    where cost includes:
        - Permanent impact: linear in trading rate
        - Temporary impact: power-law in trading rate
        - Timing risk: volatility of unexecuted quantity

    The optimal trajectory is:

        x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)

    where kappa = sqrt(lambda * sigma^2 / eta)
    """

    def __init__(self, params: Optional[ACParameters] = None) -> None:
        """Initialize the AC model.

        Args:
            params: Model parameters. Uses defaults if None.
        """
        self.params = params or ACParameters()

    def _compute_kappa(self) -> float:
        """Compute the AC kappa parameter.

        Returns:
            Kappa = sqrt(lambda * sigma^2 / eta).
        """
        lam = self.params.risk_aversion
        sigma = self.params.sigma
        eta = self.params.eta

        if eta == 0:
            return 0.0
        return math.sqrt(lam * sigma**2 / eta)

    def optimal_trajectory(self) -> NDArray[np.float64]:
        """Compute the optimal trading trajectory.

        Returns:
            Array of remaining quantities at each time step.
        """
        X = self.params.X
        T = self.params.T
        n = self.params.num_slices
        kappa = self._compute_kappa()

        times = np.linspace(0, T, n + 1)
        trajectory = np.zeros(n + 1)

        if kappa == 0:
            # Linear trajectory (no risk aversion)
            trajectory = X * (1.0 - times / T)
        else:
            # Hyperbolic sine trajectory
            trajectory = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

        return trajectory

    def generate_schedule(self, current_price: float) -> ExecutionSchedule:
        """Generate a complete execution schedule.

        Args:
            current_price: Current market price.

        Returns:
            ExecutionSchedule with slices and cost estimates.
        """
        if current_price <= 0:
            raise ParameterValidationError(
                f"current_price must be positive, got {current_price}"
            )

        trajectory = self.optimal_trajectory()
        n = self.params.num_slices
        T = self.params.T
        dt = T / n

        slices: list[ExecutionSlice] = []
        total_impact = 0.0
        total_variance = 0.0

        for i in range(n):
            t = i * dt
            qty = trajectory[i] - trajectory[i + 1]

            # Expected execution price includes permanent impact
            perm_impact = self.params.gamma_perm * qty
            exec_price = current_price + perm_impact

            slices.append(
                ExecutionSlice(
                    time=round(t, 6),
                    quantity=round(qty, 4),
                    price=round(exec_price, 4),
                )
            )

            # Temporary impact cost
            temp_impact = self.params.eta * (qty / dt) ** self.params.epsilon
            total_impact += temp_impact * qty

            # Timing risk variance
            remaining = trajectory[i + 1]
            total_variance += self.params.sigma**2 * remaining**2 * dt

        expected_shortfall = total_impact
        risk_measure = total_variance

        return ExecutionSchedule(
            slices=slices,
            total_cost=round(total_impact, 4),
            expected_shortfall=round(expected_shortfall, 4),
            risk_measure=round(risk_measure, 4),
        )

    def expected_cost(self, schedule: ExecutionSchedule) -> float:
        """Compute the expected total cost of a schedule.

        Args:
            schedule: The execution schedule.

        Returns:
            Expected total cost.
        """
        return schedule.total_cost

    def implementation_shortfall(
        self, schedule: ExecutionSchedule, arrival_price: float
    ) -> float:
        """Compute implementation shortfall.

        Args:
            schedule: The execution schedule.
            arrival_price: Price at decision time.

        Returns:
            Implementation shortfall (positive = cost).
        """
        if arrival_price <= 0:
            raise ParameterValidationError(
                f"arrival_price must be positive, got {arrival_price}"
            )

        total_executed = sum(s.quantity for s in schedule.slices)
        vwap = (
            sum(s.quantity * s.price for s in schedule.slices) / total_executed
            if total_executed > 0
            else 0.0
        )
        return vwap - arrival_price

    def urgency_adjusted_params(self, urgency: ExecutionUrgency) -> ACParameters:
        """Get urgency-adjusted parameters.

        Args:
            urgency: Desired urgency level.

        Returns:
            Adjusted ACParameters.
        """
        base = self.params
        if urgency == ExecutionUrgency.LOW:
            return ACParameters(
                X=base.X,
                T=base.T * 2.0,
                sigma=base.sigma,
                eta=base.eta,
                gamma_perm=base.gamma_perm,
                epsilon=base.epsilon,
                urgency=urgency,
                num_slices=base.num_slices * 2,
                risk_aversion=base.risk_aversion * 0.5,
            )
        elif urgency == ExecutionUrgency.HIGH:
            return ACParameters(
                X=base.X,
                T=base.T * 0.5,
                sigma=base.sigma,
                eta=base.eta,
                gamma_perm=base.gamma_perm,
                epsilon=base.epsilon,
                urgency=urgency,
                num_slices=max(1, base.num_slices // 2),
                risk_aversion=base.risk_aversion * 2.0,
            )
        return base


# ---------------------------------------------------------------------------
# Glosten-Milgrom Model
# ---------------------------------------------------------------------------

class GlostenMilgrom:
    """Glosten-Milgrom information-based market making model.

    The GM model describes how a market maker updates beliefs about
    an asset's true value based on the flow of buy/sell orders, and
    how adverse selection affects quoted spreads.

    Key insight: The market maker learns from order flow. Informed
    traders buy when value is high and sell when value is low,
    causing the market maker to adjust quotes after each trade.
    """

    def __init__(self, params: Optional[GMParameters] = None) -> None:
        """Initialize the GM model.

        Args:
            params: Model parameters. Uses defaults if None.
        """
        self.params = params or GMParameters()
        self._belief: Optional[BeliefState] = None
        self._trade_history: list[tuple[Side, float]] = []

    def initialize_beliefs(
        self, v_high: float, v_low: float, initial_spread: Optional[float] = None
    ) -> BeliefState:
        """Initialize belief state.

        Args:
            v_high: High true value estimate.
            v_low: Low true value estimate.
            initial_spread: Initial spread. Computed if None.

        Returns:
            Initial BeliefState.
        """
        if v_high < v_low:
            raise ParameterValidationError(
                f"v_high ({v_high}) must be >= v_low ({v_low})"
            )

        if initial_spread is None:
            initial_spread = self._compute_spread(v_high, v_low, self.params.alpha)

        self._belief = BeliefState(
            v_high=v_high,
            v_low=v_low,
            p_informed=self.params.alpha,
            spread=initial_spread,
        )
        return self._belief

    def _compute_spread(
        self, v_high: float, v_low: float, p_informed: float
    ) -> float:
        """Compute the spread given beliefs.

        Args:
            v_high: High value estimate.
            v_low: Low value estimate.
            p_informed: Probability of informed trading.

        Returns:
            Computed spread.
        """
        if p_informed <= 0:
            return 0.0

        # Spread = (v_high - v_low) * p_informed / (1 - p_informed)
        # This is the zero-profit condition for the market maker
        spread = (v_high - v_low) * p_informed / (1.0 - p_informed)
        return min(spread, self.params.max_spread)

    def update_beliefs(self, side: Side, trade_size: float) -> BeliefState:
        """Update beliefs after observing a trade.

        Uses Bayesian updating to revise the probability of informed
        trading and the true value estimates.

        Args:
            side: Side of the observed trade (BUY or SELL).
            trade_size: Size of the trade.

        Returns:
            Updated BeliefState.
        """
        if self._belief is None:
            raise MarketMakingError("Beliefs not initialized. Call initialize_beliefs first.")

        if trade_size <= 0:
            raise ParameterValidationError(
                f"trade_size must be positive, got {trade_size}"
            )

        self._trade_history.append((side, trade_size))

        # Bayesian update of informed probability
        # P(informed | BUY) = P(BUY | informed) * P(informed) / P(BUY)
        p_inf = self._belief.p_informed
        mu = self.params.mu

        if side == Side.BUY:
            # Informed traders buy when value is high
            p_buy_given_inf = mu  # P(buy | informed) = mu
            p_buy_given_uninf = 1.0 - mu  # P(buy | uninformed) = 1 - mu
        else:
            # Informed traders sell when value is low
            p_buy_given_inf = 1.0 - mu
            p_buy_given_uninf = mu

        # Total probability of observing this side
        p_side = p_buy_given_inf * p_inf + p_buy_given_uninf * (1.0 - p_inf)

        if p_side > 0:
            p_inf_updated = (p_buy_given_inf * p_inf) / p_side
        else:
            p_inf_updated = p_inf

        # Apply learning rate for smooth updates
        p_inf_new = p_inf + self.params.learning_rate * (p_inf_updated - p_inf)
        p_inf_new = max(0.0, min(1.0, p_inf_new))

        # Update value estimates
        if side == Side.BUY:
            v_high_new = self._belief.v_high + self.params.learning_rate * trade_size * 0.01
            v_low_new = self._belief.v_low
        else:
            v_high_new = self._belief.v_high
            v_low_new = self._belief.v_low - self.params.learning_rate * trade_size * 0.01

        v_high_new = max(v_high_new, v_low_new)

        spread = self._compute_spread(v_high_new, v_low_new, p_inf_new)

        self._belief = BeliefState(
            v_high=v_high_new,
            v_low=v_low_new,
            p_informed=p_inf_new,
            spread=spread,
        )
        return self._belief

    def generate_quote(self) -> Quote:
        """Generate a quote based on current beliefs.

        Returns:
            Quote centered on the expected value.
        """
        if self._belief is None:
            raise MarketMakingError("Beliefs not initialized. Call initialize_beliefs first.")

        expected_value = (self._belief.v_high + self._belief.v_low) / 2.0
        half_spread = self._belief.spread / 2.0

        bid_price = expected_value - half_spread
        ask_price = expected_value + half_spread

        # Ensure positive prices
        bid_price = max(bid_price, 0.01)
        ask_price = max(ask_price, bid_price + 0.01)

        return Quote(
            bid_price=round(bid_price, 4),
            ask_price=round(ask_price, 4),
            bid_size=1.0,
            ask_size=1.0,
        )

    def adverse_selection_cost(self) -> float:
        """Estimate the adverse selection cost per trade.

        Returns:
            Estimated adverse selection cost.
        """
        if self._belief is None:
            return 0.0

        # Adverse selection cost = spread * P(informed) * E[|value - mid| | informed]
        p_inf = self._belief.p_informed
        spread = self._belief.spread
        expected_abs_dev = (self._belief.v_high - self._belief.v_low) / 2.0

        return p_inf * spread * expected_abs_dev

    def get_belief(self) -> Optional[BeliefState]:
        """Get current belief state."""
        return self._belief

    def reset(self) -> None:
        """Reset the model to initial state."""
        self._belief = None
        self._trade_history.clear()


# ---------------------------------------------------------------------------
# Market Making Engine (Facade)
# ---------------------------------------------------------------------------

class MarketMakingEngine:
    """High-level market making engine combining all three models.

    Provides a unified interface for:
    - Quote generation (AS model)
    - Execution scheduling (AC model)
    - Adverse selection monitoring (GM model)
    """

    def __init__(
        self,
        as_params: Optional[ASParameters] = None,
        ac_params: Optional[ACParameters] = None,
        gm_params: Optional[GMParameters] = None,
    ) -> None:
        """Initialize the market making engine.

        Args:
            as_params: Avellaneda-Stoikov parameters.
            ac_params: Almgren-Chriss parameters.
            gm_params: Glosten-Milgrom parameters.
        """
        self.as_model = AvellanedaStoikov(as_params)
        self.ac_model = AlmgrenChriss(ac_params)
        self.gm_model = GlostenMilgrom(gm_params)

    def generate_quote(
        self, market_state: MarketState, time_remaining: Optional[float] = None
    ) -> Quote:
        """Generate a quote using the AS model.

        Args:
            market_state: Current market state.
            time_remaining: Time remaining until horizon.

        Returns:
            Optimal Quote.
        """
        return self.as_model.generate_quote(market_state, time_remaining)

    def schedule_execution(
        self, quantity: float, current_price: float, urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    ) -> ExecutionSchedule:
        """Generate an execution schedule using the AC model.

        Args:
            quantity: Total quantity to execute.
            current_price: Current market price.
            urgency: Execution urgency.

        Returns:
            ExecutionSchedule.
        """
        adjusted_params = self.ac_model.urgency_adjusted_params(urgency)
        adjusted_params = ACParameters(
            X=quantity,
            T=adjusted_params.T,
            sigma=adjusted_params.sigma,
            eta=adjusted_params.eta,
            gamma_perm=adjusted_params.gamma_perm,
            epsilon=adjusted_params.epsilon,
            urgency=urgency,
            num_slices=adjusted_params.num_slices,
            risk_aversion=adjusted_params.risk_aversion,
        )
        ac = AlmgrenChriss(adjusted_params)
        return ac.generate_schedule(current_price)

    def process_trade(self, side: Side, trade_size: float) -> BeliefState:
        """Process a trade through the GM model.

        Args:
            side: Side of the trade.
            trade_size: Size of the trade.

        Returns:
            Updated BeliefState.
        """
        return self.gm_model.update_beliefs(side, trade_size)

    def get_inventory(self) -> InventoryPosition:
        """Get current inventory position."""
        return self.as_model.get_position()

    def reset(self) -> None:
        """Reset all models."""
        self.as_model.reset_inventory()
        self.gm_model.reset()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def compute_impermanent_loss(
    price_ratio: float, fee_tier: float = 0.003
) -> float:
    """Compute impermanent loss for an AMM position.

    Args:
        price_ratio: Current price / initial price.
        fee_tier: Fee tier (e.g., 0.003 for 0.3%).

    Returns:
        Impermanent loss as a fraction (positive = loss).
    """
    if price_ratio <= 0:
        raise ParameterValidationError(
            f"price_ratio must be positive, got {price_ratio}"
        )

    # IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
    il = 2.0 * math.sqrt(price_ratio) / (1.0 + price_ratio) - 1.0
    return il


def compute_as_spread_components(
    gamma: float, sigma: float, k: float, time_remaining: float
) -> tuple[float, float]:
    """Decompose AS spread into inventory and adverse selection components.

    Args:
        gamma: Risk aversion.
        sigma: Volatility.
        k: Order arrival rate parameter.
        time_remaining: Time remaining.

    Returns:
        Tuple of (inventory_component, adverse_selection_component).
    """
    inventory_component = gamma * sigma**2 * time_remaining
    adverse_selection_component = (2.0 / gamma) * math.log(1.0 + gamma / k)
    return inventory_component, adverse_selection_component


def compute_ac_impact(
    quantity: float, eta: float, epsilon: float, dt: float
) -> float:
    """Compute temporary market impact.

    Args:
        quantity: Quantity traded.
        eta: Impact coefficient.
        epsilon: Impact exponent.
        dt: Time interval.

    Returns:
        Temporary impact cost.
    """
    if dt <= 0:
        raise ParameterValidationError(f"dt must be positive, got {dt}")
    return eta * (quantity / dt) ** epsilon


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def create_default_engine() -> MarketMakingEngine:
    """Create a market making engine with default parameters.

    Returns:
        Configured MarketMakingEngine.
    """
    return MarketMakingEngine()


def quick_quote(
    mid_price: float,
    volatility: float = 0.2,
    inventory: float = 0.0,
    risk_aversion: float = 0.1,
) -> Quote:
    """Generate a quick quote with minimal configuration.

    Args:
        mid_price: Current mid-price.
        volatility: Annualized volatility.
        inventory: Current inventory.
        risk_aversion: Risk aversion coefficient.

    Returns:
        Generated Quote.
    """
    market = MarketState(
        mid_price=mid_price,
        bid=mid_price * 0.999,
        ask=mid_price * 1.001,
        bid_size=1.0,
        ask_size=1.0,
        volatility=volatility,
    )
    params = ASParameters(gamma=risk_aversion, sigma=volatility)
    model = AvellanedaStoikov(params)
    if inventory != 0:
        model._inventory = InventoryPosition(
            quantity=inventory, max_inventory=params.max_inventory
        )
    return model.generate_quote(market)


__all__ = [
    "MarketMakingError",
    "ParameterValidationError",
    "InventoryLimitError",
    "QuoteError",
    "Side",
    "OrderType",
    "ExecutionUrgency",
    "MarketState",
    "Quote",
    "InventoryPosition",
    "ASParameters",
    "ACParameters",
    "GMParameters",
    "ExecutionSlice",
    "ExecutionSchedule",
    "BeliefState",
    "QuoteGenerator",
    "InventoryManager",
    "AvellanedaStoikov",
    "AlmgrenChriss",
    "GlostenMilgrom",
    "MarketMakingEngine",
    "compute_impermanent_loss",
    "compute_as_spread_components",
    "compute_ac_impact",
    "create_default_engine",
    "quick_quote",
]
