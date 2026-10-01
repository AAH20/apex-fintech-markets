"""
Apex FinTech Markets — Kill Switch Module.

Provides pre-trade risk controls, kill switches, circuit breakers,
and position limits for production trading environments.

All controls are designed to be fail-safe: any error or unexpected
condition results in trading being halted.
"""

from __future__ import annotations

import logging
import threading
import time
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
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class KillSwitchError(Exception):
    """Base exception for kill switch errors."""


class PreTradeRiskError(KillSwitchError):
    """Raised when a pre-trade risk check fails."""


class CircuitBreakerError(KillSwitchError):
    """Raised when a circuit breaker is triggered."""


class PositionLimitError(KillSwitchError):
    """Raised when a position limit is breached."""


class KillSwitchEngagedError(KillSwitchError):
    """Raised when attempting to trade while kill switch is engaged."""


class OrderRejectedError(KillSwitchError):
    """Raised when an order is rejected by risk controls."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class KillSwitchState(str, Enum):
    """Kill switch state."""

    DISENGAGED = "disengaged"
    ENGAGED = "engaged"
    HALF_OPEN = "half_open"


class CircuitBreakerState(str, Enum):
    """Circuit breaker state."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RiskCheckResult(str, Enum):
    """Result of a risk check."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class PositionLimitType(str, Enum):
    """Types of position limits."""

    NOTIONAL = "notional"
    QUANTITY = "quantity"
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    CONCENTRATION = "concentration"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Order:
    """Represents a trading order.

    Attributes:
        order_id: Unique order identifier.
        timestamp: Order timestamp.
        instrument_id: Instrument identifier.
        side: Buy or sell.
        quantity: Order quantity.
        price: Order price.
        order_type: Order type (market, limit, etc.).
        trader_id: Trader identifier.
        algorithm_id: Algorithm identifier (if algorithmic).
        client_id: Client identifier.
        notional: Notional value.
    """

    order_id: str
    timestamp: datetime
    instrument_id: str
    side: OrderSide
    quantity: float
    price: float
    order_type: str = "market"
    trader_id: str = ""
    algorithm_id: Optional[str] = None
    client_id: Optional[str] = None
    notional: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive.")
        if self.price <= 0:
            raise ValueError("Order price must be positive.")


@dataclass(frozen=True)
class Position:
    """Represents a current position.

    Attributes:
        instrument_id: Instrument identifier.
        quantity: Current quantity (positive = long, negative = short).
        avg_price: Average entry price.
        market_price: Current market price.
        unrealized_pnl: Unrealized profit/loss.
        timestamp: Position timestamp.
    """

    instrument_id: str
    quantity: float
    avg_price: float
    market_price: float
    unrealized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def notional(self) -> float:
        """Calculate position notional value."""
        return abs(self.quantity) * self.market_price

    @property
    def market_value(self) -> float:
        """Calculate position market value."""
        return self.quantity * self.market_price


@dataclass(frozen=True)
class RiskCheck:
    """Result of a pre-trade risk check.

    Attributes:
        check_name: Name of the risk check.
        result: Result of the check.
        message: Human-readable message.
        details: Additional details.
        timestamp: Check timestamp.
    """

    check_name: str
    result: RiskCheckResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class PositionLimit:
    """Position limit configuration.

    Attributes:
        limit_type: Type of position limit.
        instrument_id: Instrument identifier (or "ALL" for portfolio-wide).
        limit_value: Limit value.
        warning_threshold: Warning threshold (0.0-1.0).
        hard_limit: Whether this is a hard limit (cannot be overridden).
    """

    limit_type: PositionLimitType
    instrument_id: str
    limit_value: float
    warning_threshold: float = 0.8
    hard_limit: bool = True

    def __post_init__(self) -> None:
        if self.limit_value <= 0:
            raise ValueError("Limit value must be positive.")
        if not 0 < self.warning_threshold <= 1:
            raise ValueError("Warning threshold must be in (0, 1].")


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Circuit breaker configuration.

    Attributes:
        failure_threshold: Number of failures before opening.
        recovery_timeout: Seconds before attempting recovery.
        half_open_max_calls: Max calls in half-open state.
        success_threshold: Successes needed to close from half-open.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


@dataclass(frozen=True)
class KillSwitchEvent:
    """Kill switch event record.

    Attributes:
        event_id: Unique event identifier.
        timestamp: Event timestamp.
        event_type: Type of event (engage, disengage, test).
        reason: Reason for the event.
        triggered_by: Who/what triggered the event.
        metadata: Additional metadata.
    """

    event_id: str
    timestamp: datetime
    event_type: str
    reason: str
    triggered_by: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pre-trade risk checks
# ---------------------------------------------------------------------------
class PreTradeRiskCheck(ABC):
    """Abstract base class for pre-trade risk checks."""

    @abstractmethod
    def check(self, order: Order, positions: Dict[str, Position]) -> RiskCheck:
        """Execute the risk check.

        Args:
            order: Order to check.
            positions: Current positions.

        Returns:
            RiskCheck with the result.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the risk check."""
        ...


class NotionalLimitCheck(PreTradeRiskCheck):
    """Pre-trade check for order notional limit."""

    def __init__(self, max_order_notional: float) -> None:
        """Initialize notional limit check.

        Args:
            max_order_notional: Maximum allowed order notional.
        """
        self.max_order_notional = max_order_notional

    @property
    def name(self) -> str:
        return "notional_limit"

    def check(self, order: Order, positions: Dict[str, Position]) -> RiskCheck:
        """Check if order notional exceeds limit.

        Args:
            order: Order to check.
            positions: Current positions (unused).

        Returns:
            RiskCheck with the result.
        """
        order_notional = order.quantity * order.price
        if order_notional > self.max_order_notional:
            return RiskCheck(
                check_name=self.name,
                result=RiskCheckResult.FAIL,
                message=f"Order notional {order_notional:.2f} exceeds limit {self.max_order_notional:.2f}",
                details={
                    "order_notional": order_notional,
                    "limit": self.max_order_notional,
                },
            )
        return RiskCheck(
            check_name=self.name,
            result=RiskCheckResult.PASS,
            message=f"Order notional {order_notional:.2f} within limit",
        )


class PositionLimitCheck(PreTradeRiskCheck):
    """Pre-trade check for position limits."""

    def __init__(self, limits: Sequence[PositionLimit]) -> None:
        """Initialize position limit check.

        Args:
            limits: Position limits to enforce.
        """
        self.limits = list(limits)

    @property
    def name(self) -> str:
        return "position_limit"

    def check(self, order: Order, positions: Dict[str, Position]) -> RiskCheck:
        """Check if order would breach position limits.

        Args:
            order: Order to check.
            positions: Current positions.

        Returns:
            RiskCheck with the result.
        """
        current_pos = positions.get(order.instrument_id)
        current_qty = current_pos.quantity if current_pos else 0.0

        # Calculate post-trade position
        if order.side == OrderSide.BUY:
            post_trade_qty = current_qty + order.quantity
        else:
            post_trade_qty = current_qty - order.quantity

        for limit in self.limits:
            if limit.instrument_id not in (order.instrument_id, "ALL"):
                continue

            if limit.limit_type == PositionLimitType.QUANTITY:
                if abs(post_trade_qty) > limit.limit_value:
                    return RiskCheck(
                        check_name=self.name,
                        result=RiskCheckResult.FAIL,
                        message=f"Position quantity {abs(post_trade_qty):.2f} would exceed limit {limit.limit_value:.2f}",
                        details={
                            "instrument_id": order.instrument_id,
                            "post_trade_quantity": post_trade_qty,
                            "limit": limit.limit_value,
                        },
                    )
            elif limit.limit_type == PositionLimitType.NOTIONAL:
                post_trade_notional = abs(post_trade_qty) * order.price
                if post_trade_notional > limit.limit_value:
                    return RiskCheck(
                        check_name=self.name,
                        result=RiskCheckResult.FAIL,
                        message=f"Position notional {post_trade_notional:.2f} would exceed limit {limit.limit_value:.2f}",
                        details={
                            "instrument_id": order.instrument_id,
                            "post_trade_notional": post_trade_notional,
                            "limit": limit.limit_value,
                        },
                    )

        return RiskCheck(
            check_name=self.name,
            result=RiskCheckResult.PASS,
            message="Position within limits",
        )


class ConcentrationLimitCheck(PreTradeRiskCheck):
    """Pre-trade check for portfolio concentration."""

    def __init__(self, max_concentration_pct: float, portfolio_value: float) -> None:
        """Initialize concentration limit check.

        Args:
            max_concentration_pct: Maximum concentration as percentage (0-100).
            portfolio_value: Total portfolio value.
        """
        self.max_concentration_pct = max_concentration_pct
        self.portfolio_value = portfolio_value

    @property
    def name(self) -> str:
        return "concentration_limit"

    def check(self, order: Order, positions: Dict[str, Position]) -> RiskCheck:
        """Check if order would breach concentration limit.

        Args:
            order: Order to check.
            positions: Current positions.

        Returns:
            RiskCheck with the result.
        """
        if self.portfolio_value <= 0:
            return RiskCheck(
                check_name=self.name,
                result=RiskCheckResult.WARNING,
                message="Portfolio value is zero or negative, skipping concentration check",
            )

        order_notional = order.quantity * order.price
        concentration = (order_notional / self.portfolio_value) * 100

        if concentration > self.max_concentration_pct:
            return RiskCheck(
                check_name=self.name,
                result=RiskCheckResult.FAIL,
                message=f"Order concentration {concentration:.2f}% exceeds limit {self.max_concentration_pct:.2f}%",
                details={
                    "concentration_pct": concentration,
                    "limit_pct": self.max_concentration_pct,
                },
            )
        return RiskCheck(
            check_name=self.name,
            result=RiskCheckResult.PASS,
            message=f"Order concentration {concentration:.2f}% within limit",
        )


class VelocityLimitCheck(PreTradeRiskCheck):
    """Pre-trade check for order velocity (orders per second)."""

    def __init__(self, max_orders_per_second: int, window_seconds: float = 1.0) -> None:
        """Initialize velocity limit check.

        Args:
            max_orders_per_second: Maximum orders per second.
            window_seconds: Time window for velocity calculation.
        """
        self.max_orders_per_second = max_orders_per_second
        self.window_seconds = window_seconds
        self._order_timestamps: List[datetime] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "velocity_limit"

    def check(self, order: Order, positions: Dict[str, Position]) -> RiskCheck:
        """Check if order velocity exceeds limit.

        Args:
            order: Order to check.
            positions: Current positions (unused).

        Returns:
            RiskCheck with the result.
        """
        with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=self.window_seconds)
            self._order_timestamps = [
                t for t in self._order_timestamps if t > cutoff
            ]
            self._order_timestamps.append(now)

            current_velocity = len(self._order_timestamps) / self.window_seconds

            if current_velocity > self.max_orders_per_second:
                return RiskCheck(
                    check_name=self.name,
                    result=RiskCheckResult.FAIL,
                    message=f"Order velocity {current_velocity:.2f}/s exceeds limit {self.max_orders_per_second}/s",
                    details={
                        "current_velocity": current_velocity,
                        "limit": self.max_orders_per_second,
                    },
                )
            return RiskCheck(
                check_name=self.name,
                result=RiskCheckResult.PASS,
                message=f"Order velocity {current_velocity:.2f}/s within limit",
            )


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """Circuit breaker pattern for trading operations.

    Prevents cascading failures by halting trading after repeated failures.
    """

    def __init__(self, config: CircuitBreakerConfig, name: str = "default") -> None:
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration.
            name: Circuit breaker name.
        """
        self.config = config
        self.name = name
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            CircuitBreakerError: If circuit breaker is open.
        """
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker %s entering HALF_OPEN state", self.name)
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} is OPEN"
                    )

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} HALF_OPEN limit reached"
                    )
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset.

        Returns:
            True if recovery timeout has elapsed.
        """
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker %s CLOSED", self.name)
            else:
                self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                logger.warning("Circuit breaker %s OPEN (half-open failure)", self.name)
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    "Circuit breaker %s OPEN (%d failures)",
                    self.name,
                    self._failure_count,
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            logger.info("Circuit breaker %s manually reset", self.name)


# ---------------------------------------------------------------------------
# Kill Switch
# ---------------------------------------------------------------------------
class KillSwitch:
    """Kill switch for emergency trading halt.

    When engaged, all trading activity is immediately halted.
    Can be triggered manually or automatically by risk events.
    """

    def __init__(self, name: str = "default") -> None:
        """Initialize kill switch.

        Args:
            name: Kill switch name.
        """
        self.name = name
        self._state = KillSwitchState.DISENGAGED
        self._events: List[KillSwitchEvent] = []
        self._event_counter = 0
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[KillSwitchEvent], None]] = []

    @property
    def state(self) -> KillSwitchState:
        """Get current kill switch state."""
        with self._lock:
            return self._state

    @property
    def is_engaged(self) -> bool:
        """Check if kill switch is engaged."""
        return self.state == KillSwitchState.ENGAGED

    def engage(self, reason: str, triggered_by: str = "system") -> KillSwitchEvent:
        """Engage the kill switch.

        Args:
            reason: Reason for engaging.
            triggered_by: Who/what triggered the engagement.

        Returns:
            KillSwitchEvent record.
        """
        with self._lock:
            if self._state == KillSwitchState.ENGAGED:
                logger.warning("Kill switch %s already engaged", self.name)
                return self._events[-1]

            self._state = KillSwitchState.ENGAGED
            self._event_counter += 1
            event = KillSwitchEvent(
                event_id=f"KS-{self._event_counter:06d}",
                timestamp=datetime.utcnow(),
                event_type="engage",
                reason=reason,
                triggered_by=triggered_by,
            )
            self._events.append(event)
            logger.critical(
                "KILL SWITCH %s ENGAGED: %s (by %s)",
                self.name,
                reason,
                triggered_by,
            )

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error("Kill switch callback error: %s", e)

            return event

    def disengage(self, reason: str, triggered_by: str = "system") -> KillSwitchEvent:
        """Disengage the kill switch.

        Args:
            reason: Reason for disengaging.
            triggered_by: Who/what triggered the disengagement.

        Returns:
            KillSwitchEvent record.

        Raises:
            KillSwitchError: If kill switch is not engaged.
        """
        with self._lock:
            if self._state != KillSwitchState.ENGAGED:
                raise KillSwitchError(
                    f"Kill switch {self.name} is not engaged (state: {self._state})"
                )

            self._state = KillSwitchState.DISENGAGED
            self._event_counter += 1
            event = KillSwitchEvent(
                event_id=f"KS-{self._event_counter:06d}",
                timestamp=datetime.utcnow(),
                event_type="disengage",
                reason=reason,
                triggered_by=triggered_by,
            )
            self._events.append(event)
            logger.info(
                "Kill switch %s DISENGAGED: %s (by %s)",
                self.name,
                reason,
                triggered_by,
            )
            return event

    def register_callback(self, callback: Callable[[KillSwitchEvent], None]) -> None:
        """Register a callback for kill switch events.

        Args:
            callback: Function to call on kill switch events.
        """
        self._callbacks.append(callback)

    def get_events(self) -> List[KillSwitchEvent]:
        """Get all kill switch events.

        Returns:
            List of kill switch events.
        """
        with self._lock:
            return list(self._events)


# ---------------------------------------------------------------------------
# Pre-trade Risk Manager
# ---------------------------------------------------------------------------
class PreTradeRiskManager:
    """Pre-trade risk manager.

    Orchestrates all pre-trade risk checks and enforces kill switch status.
    """

    def __init__(self, kill_switch: KillSwitch) -> None:
        """Initialize pre-trade risk manager.

        Args:
            kill_switch: Kill switch instance.
        """
        self.kill_switch = kill_switch
        self._checks: List[PreTradeRiskCheck] = []
        self._check_results: List[RiskCheck] = []

    def add_check(self, check: PreTradeRiskCheck) -> None:
        """Add a pre-trade risk check.

        Args:
            check: Risk check to add.
        """
        self._checks.append(check)

    def remove_check(self, check_name: str) -> None:
        """Remove a pre-trade risk check by name.

        Args:
            check_name: Name of the check to remove.
        """
        self._checks = [c for c in self._checks if c.name != check_name]

    def validate_order(
        self,
        order: Order,
        positions: Dict[str, Position],
    ) -> Tuple[bool, List[RiskCheck]]:
        """Validate an order against all pre-trade risk checks.

        Args:
            order: Order to validate.
            positions: Current positions.

        Returns:
            Tuple of (is_valid, list of risk check results).

        Raises:
            KillSwitchEngagedError: If kill switch is engaged.
        """
        if self.kill_switch.is_engaged:
            raise KillSwitchEngagedError(
                f"Order {order.order_id} rejected: kill switch is engaged"
            )

        results: List[RiskCheck] = []
        all_passed = True

        for check in self._checks:
            result = check.check(order, positions)
            results.append(result)
            if result.result == RiskCheckResult.FAIL:
                all_passed = False
                logger.warning(
                    "Risk check %s FAILED for order %s: %s",
                    check.name,
                    order.order_id,
                    result.message,
                )

        self._check_results.extend(results)
        return all_passed, results

    def get_check_results(self) -> List[RiskCheck]:
        """Get all risk check results.

        Returns:
            List of risk check results.
        """
        return list(self._check_results)

    def clear_check_results(self) -> None:
        """Clear all risk check results."""
        self._check_results.clear()


# ---------------------------------------------------------------------------
# Position Limit Manager
# ---------------------------------------------------------------------------
class PositionLimitManager:
    """Manages and enforces position limits."""

    def __init__(self) -> None:
        """Initialize position limit manager."""
        self._limits: Dict[str, List[PositionLimit]] = {}
        self._breaches: List[Dict[str, Any]] = []

    def add_limit(self, limit: PositionLimit) -> None:
        """Add a position limit.

        Args:
            limit: Position limit to add.
        """
        key = limit.instrument_id
        if key not in self._limits:
            self._limits[key] = []
        self._limits[key].append(limit)

    def remove_limit(self, instrument_id: str, limit_type: PositionLimitType) -> None:
        """Remove a position limit.

        Args:
            instrument_id: Instrument identifier.
            limit_type: Type of limit to remove.
        """
        if instrument_id in self._limits:
            self._limits[instrument_id] = [
                l for l in self._limits[instrument_id]
                if l.limit_type != limit_type
            ]

    def check_limits(
        self,
        positions: Dict[str, Position],
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check all positions against limits.

        Args:
            positions: Current positions.

        Returns:
            Tuple of (all_within_limits, list of breaches).
        """
        breaches: List[Dict[str, Any]] = []

        for instrument_id, position in positions.items():
            limits = self._limits.get(instrument_id, []) + self._limits.get("ALL", [])

            for limit in limits:
                if limit.limit_type == PositionLimitType.QUANTITY:
                    if abs(position.quantity) > limit.limit_value:
                        breaches.append({
                            "instrument_id": instrument_id,
                            "limit_type": limit.limit_type.value,
                            "current_value": abs(position.quantity),
                            "limit_value": limit.limit_value,
                            "timestamp": datetime.utcnow(),
                        })
                elif limit.limit_type == PositionLimitType.NOTIONAL:
                    if position.notional > limit.limit_value:
                        breaches.append({
                            "instrument_id": instrument_id,
                            "limit_type": limit.limit_type.value,
                            "current_value": position.notional,
                            "limit_value": limit.limit_value,
                            "timestamp": datetime.utcnow(),
                        })

        self._breaches.extend(breaches)
        return len(breaches) == 0, breaches

    def get_breaches(self) -> List[Dict[str, Any]]:
        """Get all limit breaches.

        Returns:
            List of limit breaches.
        """
        return list(self._breaches)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_risk_manager(
    max_order_notional: float = 1_000_000.0,
    max_position_notional: float = 10_000_000.0,
    max_concentration_pct: float = 25.0,
    portfolio_value: float = 100_000_000.0,
) -> Tuple[KillSwitch, PreTradeRiskManager]:
    """Create a fully configured risk manager.

    Args:
        max_order_notional: Maximum order notional.
        max_position_notional: Maximum position notional.
        max_concentration_pct: Maximum concentration percentage.
        portfolio_value: Portfolio value for concentration calculation.

    Returns:
        Tuple of (KillSwitch, PreTradeRiskManager).
    """
    kill_switch = KillSwitch()
    risk_manager = PreTradeRiskManager(kill_switch)

    # Add default checks
    risk_manager.add_check(NotionalLimitCheck(max_order_notional))
    risk_manager.add_check(
        PositionLimitCheck([
            PositionLimit(
                limit_type=PositionLimitType.NOTIONAL,
                instrument_id="ALL",
                limit_value=max_position_notional,
            )
        ])
    )
    risk_manager.add_check(
        ConcentrationLimitCheck(max_concentration_pct, portfolio_value)
    )
    risk_manager.add_check(VelocityLimitCheck(max_orders_per_second=10))

    return kill_switch, risk_manager
