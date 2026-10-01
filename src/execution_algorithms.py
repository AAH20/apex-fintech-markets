"""
Apex FinTech Markets — Execution Algorithms.

Implements institutional-grade execution algorithms:

1. **TWAP (Time-Weighted Average Price)** — Slices orders evenly across
   a time horizon to minimize market impact and timing risk.

2. **VWAP (Volume-Weighted Average Price)** — Slices orders proportional
   to expected volume distribution to track the market's VWAP.

3. **POV (Percentage of Volume)** — Participates at a fixed percentage
   of market volume to minimize footprint while maintaining schedule.

4. **SOR (Smart Order Router)** — Routes orders across multiple venues
   to achieve best execution, considering fees, liquidity, and latency.

All algorithms support real-time adaptation, comprehensive error handling,
and detailed execution analytics.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ExecutionError(Exception):
    """Base exception for execution algorithm errors."""


class ScheduleError(ExecutionError):
    """Raised when schedule generation fails."""


class RoutingError(ExecutionError):
    """Raised when order routing fails."""


class ValidationError(ExecutionError):
    """Raised when parameters fail validation."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"
    FOK = "fok"


class ExecutionStatus(Enum):
    """Execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class RoutingStrategy(Enum):
    """Smart order router strategy."""

    COST_BASED = "cost_based"           # Minimize total cost
    LIQUIDITY_BASED = "liquidity_based"  # Maximize fill probability
    LATENCY_BASED = "latency_based"      # Minimize latency
    BALANCED = "balanced"                # Balance all factors


class VenueType(Enum):
    """Venue type."""

    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    OTC = "otc"
    AMM = "amm"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instrument:
    """Trading instrument.

    Attributes:
        symbol: Instrument symbol.
        exchange: Primary exchange.
        lot_size: Minimum tradable quantity.
        tick_size: Minimum price increment.
    """

    symbol: str
    exchange: str = "NYSE"
    lot_size: float = 1.0
    tick_size: float = 0.01

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValidationError("symbol cannot be empty")
        if self.lot_size <= 0:
            raise ValidationError(f"lot_size must be positive, got {self.lot_size}")
        if self.tick_size <= 0:
            raise ValidationError(f"tick_size must be positive, got {self.tick_size}")

    def round_quantity(self, qty: float) -> float:
        """Round quantity to lot size.

        Args:
            qty: Quantity to round.

        Returns:
            Rounded quantity.
        """
        return round(qty / self.lot_size) * self.lot_size

    def round_price(self, price: float) -> float:
        """Round price to tick size.

        Args:
            price: Price to round.

        Returns:
            Rounded price.
        """
        return round(price / self.tick_size) * self.tick_size


@dataclass(frozen=True)
class Venue:
    """Trading venue.

    Attributes:
        name: Venue name.
        venue_type: Type of venue.
        fee_bps: Fee in basis points.
        rebate_bps: Rebate in basis points (for liquidity providers).
        latency_ms: Expected latency in milliseconds.
        is_available: Whether venue is available.
    """

    name: str
    venue_type: VenueType = VenueType.EXCHANGE
    fee_bps: float = 0.0
    rebate_bps: float = 0.0
    latency_ms: float = 10.0
    is_available: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("Venue name cannot be empty")
        if self.fee_bps < 0:
            raise ValidationError(f"fee_bps must be >= 0, got {self.fee_bps}")
        if self.rebate_bps < 0:
            raise ValidationError(f"rebate_bps must be >= 0, got {self.rebate_bps}")
        if self.latency_ms < 0:
            raise ValidationError(f"latency_ms must be >= 0, got {self.latency_ms}")

    @property
    def net_cost_bps(self) -> float:
        """Net cost in basis points (fee - rebate)."""
        return self.fee_bps - self.rebate_bps


@dataclass(frozen=True)
class MarketSnapshot:
    """Market data snapshot for a venue.

    Attributes:
        venue: Venue name.
        bid: Best bid.
        ask: Best ask.
        bid_size: Best bid size.
        ask_size: Best ask size.
        last_price: Last trade price.
        volume: Current volume.
        timestamp: Snapshot timestamp.
    """

    venue: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    timestamp: float

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ValidationError(
                f"bid/ask must be positive, got bid={self.bid}, ask={self.ask}"
            )
        if self.bid > self.ask:
            raise ValidationError(f"Crossed market: bid={self.bid} > ask={self.ask}")

    @property
    def mid_price(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid_price) * 10_000 if self.mid_price > 0 else 0.0


@dataclass(frozen=True)
class OrderSlice:
    """A single execution slice.

    Attributes:
        slice_id: Slice identifier.
        start_time: Start time (seconds from start).
        end_time: End time (seconds from start).
        quantity: Target quantity.
        price_limit: Price limit (optional).
        venue: Target venue.
        order_type: Order type.
    """

    slice_id: int
    start_time: float
    end_time: float
    quantity: float
    price_limit: Optional[float] = None
    venue: Optional[str] = None
    order_type: OrderType = OrderType.LIMIT

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValidationError(f"quantity must be positive, got {self.quantity}")
        if self.start_time < 0:
            raise ValidationError(
                f"start_time must be >= 0, got {self.start_time}"
            )
        if self.end_time <= self.start_time:
            raise ValidationError(
                f"end_time ({self.end_time}) must be > start_time ({self.start_time})"
            )


@dataclass(frozen=True)
class ExecutionResult:
    """Result of an execution slice.

    Attributes:
        slice_id: Slice identifier.
        quantity_filled: Quantity filled.
        avg_price: Average execution price.
        fees: Fees paid.
        slippage_bps: Slippage in basis points.
        venue: Venue where executed.
        status: Execution status.
        timestamp: Execution timestamp.
    """

    slice_id: int
    quantity_filled: float
    avg_price: float
    fees: float
    slippage_bps: float
    venue: str
    status: ExecutionStatus
    timestamp: float

    def __post_init__(self) -> None:
        if self.quantity_filled < 0:
            raise ValidationError(
                f"quantity_filled must be >= 0, got {self.quantity_filled}"
            )
        if self.avg_price < 0:
            raise ValidationError(f"avg_price must be >= 0, got {self.avg_price}")


@dataclass(frozen=True)
class ExecutionSchedule:
    """Complete execution schedule.

    Attributes:
        slices: List of order slices.
        total_quantity: Total quantity to execute.
        total_duration: Total duration in seconds.
        strategy: Strategy name.
    """

    slices: list[OrderSlice]
    total_quantity: float
    total_duration: float
    strategy: str

    def __post_init__(self) -> None:
        if self.total_quantity <= 0:
            raise ValidationError(
                f"total_quantity must be positive, got {self.total_quantity}"
            )
        if self.total_duration <= 0:
            raise ValidationError(
                f"total_duration must be positive, got {self.total_duration}"
            )

    @property
    def num_slices(self) -> int:
        """Number of slices."""
        return len(self.slices)

    @property
    def slice_quantities(self) -> list[float]:
        """List of slice quantities."""
        return [s.quantity for s in self.slices]


@dataclass(frozen=True)
class ExecutionAnalytics:
    """Analytics for a completed execution.

    Attributes:
        total_filled: Total quantity filled.
        vwap: Volume-weighted average price.
        arrival_price: Price at decision time.
        implementation_shortfall_bps: Implementation shortfall in bps.
        total_fees: Total fees paid.
        total_slippage_bps: Total slippage in bps.
        completion_rate: Fraction of order completed.
        avg_slice_size: Average slice size.
        max_slice_size: Maximum slice size.
    """

    total_filled: float
    vwap: float
    arrival_price: float
    implementation_shortfall_bps: float
    total_fees: float
    total_slippage_bps: float
    completion_rate: float
    avg_slice_size: float
    max_slice_size: float

    def __post_init__(self) -> None:
        if self.total_filled < 0:
            raise ValidationError(
                f"total_filled must be >= 0, got {self.total_filled}"
            )
        if self.vwap < 0:
            raise ValidationError(f"vwap must be >= 0, got {self.vwap}")


@dataclass(frozen=True)
class VolumeProfile:
    """Volume profile for VWAP scheduling.

    Attributes:
        timestamps: Time points (seconds from start).
        volumes: Expected volume at each time point.
        total_volume: Total expected volume.
    """

    timestamps: NDArray[np.float64]
    volumes: NDArray[np.float64]
    total_volume: float

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.volumes):
            raise ValidationError("timestamps and volumes must have same length")
        if len(self.timestamps) == 0:
            raise ValidationError("Volume profile cannot be empty")
        if np.any(self.volumes < 0):
            raise ValidationError("volumes must be non-negative")
        if self.total_volume <= 0:
            raise ValidationError(
                f"total_volume must be positive, got {self.total_volume}"
            )

    def get_cumulative_volume(self) -> NDArray[np.float64]:
        """Get cumulative volume array.

        Returns:
            Cumulative volume at each timestamp.
        """
        return np.cumsum(self.volumes)

    def get_volume_at_time(self, t: float) -> float:
        """Get expected volume at a specific time.

        Args:
            t: Time in seconds from start.

        Returns:
            Expected volume at time t.
        """
        if t <= 0:
            return 0.0
        if t >= self.timestamps[-1]:
            return self.volumes[-1]

        idx = np.searchsorted(self.timestamps, t, side="right") - 1
        if idx < 0:
            return 0.0
        return float(self.volumes[idx])


# ---------------------------------------------------------------------------
# Base Execution Algorithm
# ---------------------------------------------------------------------------

class ExecutionAlgorithm(ABC):
    """Abstract base class for execution algorithms."""

    def __init__(self, instrument: Instrument) -> None:
        """Initialize the execution algorithm.

        Args:
            instrument: Trading instrument.
        """
        self.instrument = instrument
        self._status = ExecutionStatus.PENDING
        self._results: list[ExecutionResult] = []

    @property
    def status(self) -> ExecutionStatus:
        """Current execution status."""
        return self._status

    @abstractmethod
    def generate_schedule(
        self, total_quantity: float, duration_seconds: float, **kwargs: float
    ) -> ExecutionSchedule:
        """Generate an execution schedule.

        Args:
            total_quantity: Total quantity to execute.
            duration_seconds: Total duration in seconds.
            **kwargs: Algorithm-specific parameters.

        Returns:
            ExecutionSchedule.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Get algorithm name.

        Returns:
            Algorithm name.
        """
        ...

    def record_result(self, result: ExecutionResult) -> None:
        """Record an execution result.

        Args:
            result: The execution result.
        """
        self._results.append(result)

    def get_analytics(self, arrival_price: float) -> ExecutionAnalytics:
        """Compute execution analytics.

        Args:
            arrival_price: Price at decision time.

        Returns:
            ExecutionAnalytics.
        """
        if not self._results:
            return ExecutionAnalytics(
                total_filled=0.0,
                vwap=0.0,
                arrival_price=arrival_price,
                implementation_shortfall_bps=0.0,
                total_fees=0.0,
                total_slippage_bps=0.0,
                completion_rate=0.0,
                avg_slice_size=0.0,
                max_slice_size=0.0,
            )

        total_filled = sum(r.quantity_filled for r in self._results)
        total_fees = sum(r.fees for r in self._results)

        if total_filled > 0:
            vwap = sum(r.avg_price * r.quantity_filled for r in self._results) / total_filled
        else:
            vwap = 0.0

        # Implementation shortfall
        if arrival_price > 0 and total_filled > 0:
            is_bps = ((vwap - arrival_price) / arrival_price) * 10_000
        else:
            is_bps = 0.0

        total_slippage = sum(r.slippage_bps for r in self._results) / len(self._results)

        slice_sizes = [r.quantity_filled for r in self._results]
        avg_slice = np.mean(slice_sizes) if slice_sizes else 0.0
        max_slice = max(slice_sizes) if slice_sizes else 0.0

        return ExecutionAnalytics(
            total_filled=total_filled,
            vwap=vwap,
            arrival_price=arrival_price,
            implementation_shortfall_bps=is_bps,
            total_fees=total_fees,
            total_slippage_bps=total_slippage,
            completion_rate=total_filled / self._results[0].quantity_filled if self._results else 0.0,
            avg_slice_size=avg_slice,
            max_slice_size=max_slice,
        )

    def reset(self) -> None:
        """Reset the algorithm state."""
        self._status = ExecutionStatus.PENDING
        self._results.clear()


# ---------------------------------------------------------------------------
# TWAP Algorithm
# ---------------------------------------------------------------------------

class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price execution algorithm.

    Slices orders evenly across the execution horizon to minimize
    market impact and timing risk. Each slice is of equal size and
    executed at regular intervals.
    """

    def __init__(self, instrument: Instrument) -> None:
        """Initialize TWAP.

        Args:
            instrument: Trading instrument.
        """
        super().__init__(instrument)

    def get_name(self) -> str:
        """Get algorithm name."""
        return "TWAP"

    def generate_schedule(
        self,
        total_quantity: float,
        duration_seconds: float,
        num_slices: int = 10,
        price_limit: Optional[float] = None,
        start_time: float = 0.0,
        **kwargs: float,
    ) -> ExecutionSchedule:
        """Generate a TWAP schedule.

        Args:
            total_quantity: Total quantity to execute.
            duration_seconds: Total duration in seconds.
            num_slices: Number of slices.
            price_limit: Optional price limit.
            start_time: Start time offset in seconds.
            **kwargs: Additional parameters (ignored).

        Returns:
            ExecutionSchedule.
        """
        if total_quantity <= 0:
            raise ScheduleError(
                f"total_quantity must be positive, got {total_quantity}"
            )
        if duration_seconds <= 0:
            raise ScheduleError(
                f"duration_seconds must be positive, got {duration_seconds}"
            )
        if num_slices <= 0:
            raise ScheduleError(f"num_slices must be positive, got {num_slices}")

        slice_qty = self.instrument.round_quantity(total_quantity / num_slices)
        interval = duration_seconds / num_slices

        slices: list[OrderSlice] = []
        for i in range(num_slices):
            t_start = start_time + i * interval
            t_end = t_start + interval
            slices.append(
                OrderSlice(
                    slice_id=i,
                    start_time=t_start,
                    end_time=t_end,
                    quantity=slice_qty,
                    price_limit=price_limit,
                    order_type=OrderType.LIMIT if price_limit else OrderType.MARKET,
                )
            )

        return ExecutionSchedule(
            slices=slices,
            total_quantity=slice_qty * num_slices,
            total_duration=duration_seconds,
            strategy=self.get_name(),
        )

    def generate_adaptive_schedule(
        self,
        total_quantity: float,
        duration_seconds: float,
        num_slices: int = 10,
        volatility: float = 0.2,
        **kwargs: float,
    ) -> ExecutionSchedule:
        """Generate an adaptive TWAP schedule.

        Adjusts slice sizes based on volatility — larger slices during
        low-volatility periods.

        Args:
            total_quantity: Total quantity to execute.
            duration_seconds: Total duration in seconds.
            num_slices: Number of slices.
            volatility: Current volatility estimate.
            **kwargs: Additional parameters.

        Returns:
            ExecutionSchedule.
        """
        if volatility <= 0:
            raise ScheduleError(f"volatility must be positive, got {volatility}")

        # Higher volatility -> more slices (smaller size)
        adaptive_slices = max(1, int(num_slices * (1.0 + volatility)))
        return self.generate_schedule(
            total_quantity, duration_seconds, num_slices=adaptive_slices, **kwargs
        )


# ---------------------------------------------------------------------------
# VWAP Algorithm
# ---------------------------------------------------------------------------

class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution algorithm.

    Slices orders proportional to expected volume distribution,
    aiming to track the market's VWAP over the execution horizon.
    """

    def __init__(self, instrument: Instrument) -> None:
        """Initialize VWAP.

        Args:
            instrument: Trading instrument.
        """
        super().__init__(instrument)

    def get_name(self) -> str:
        """Get algorithm name."""
        return "VWAP"

    def generate_schedule(
        self,
        total_quantity: float,
        duration_seconds: float,
        volume_profile: Optional[VolumeProfile] = None,
        num_slices: int = 10,
        price_limit: Optional[float] = None,
        **kwargs: float,
    ) -> ExecutionSchedule:
        """Generate a VWAP schedule.

        Args:
            total_quantity: Total quantity to execute.
            duration_seconds: Total duration in seconds.
            volume_profile: Expected volume profile. Uses uniform if None.
            num_slices: Number of slices.
            price_limit: Optional price limit.
            **kwargs: Additional parameters.

        Returns:
            ExecutionSchedule.
        """
        if total_quantity <= 0:
            raise ScheduleError(
                f"total_quantity must be positive, got {total_quantity}"
            )
        if duration_seconds <= 0:
            raise ScheduleError(
                f"duration_seconds must be positive, got {duration_seconds}"
            )

        if volume_profile is None:
            # Uniform volume profile
            timestamps = np.linspace(0, duration_seconds, num_slices + 1)
            volumes = np.ones(num_slices) * (1.0 / num_slices)
            volume_profile = VolumeProfile(
                timestamps=timestamps,
                volumes=volumes,
                total_volume=1.0,
            )

        # Compute slice quantities proportional to volume
        cum_vol = volume_profile.get_cumulative_volume()
        total_vol = cum_vol[-1]

        slices: list[OrderSlice] = []
        interval = duration_seconds / num_slices

        for i in range(num_slices):
            t_start = i * interval
            t_end = t_start + interval

            # Volume fraction for this slice
            vol_start = cum_vol[i] if i < len(cum_vol) else 0.0
            vol_end = cum_vol[i + 1] if i + 1 < len(cum_vol) else total_vol
            vol_fraction = (vol_end - vol_start) / total_vol if total_vol > 0 else 1.0 / num_slices

            slice_qty = self.instrument.round_quantity(total_quantity * vol_fraction)
            # Ensure minimum quantity of 1 lot
            min_qty = self.instrument.lot_size
            if slice_qty < min_qty:
                slice_qty = min_qty

            slices.append(
                OrderSlice(
                    slice_id=i,
                    start_time=t_start,
                    end_time=t_end,
                    quantity=slice_qty,
                    price_limit=price_limit,
                    order_type=OrderType.LIMIT if price_limit else OrderType.MARKET,
                )
            )

        return ExecutionSchedule(
            slices=slices,
            total_quantity=sum(s.quantity for s in slices),
            total_duration=duration_seconds,
            strategy=self.get_name(),
        )

    def generate_from_historical(
        self,
        total_quantity: float,
        duration_seconds: float,
        historical_volumes: NDArray[np.float64],
        num_slices: int = 10,
        **kwargs: float,
    ) -> ExecutionSchedule:
        """Generate VWAP schedule from historical volume data.

        Args:
            total_quantity: Total quantity to execute.
            duration_seconds: Total duration in seconds.
            historical_volumes: Historical volume distribution.
            num_slices: Number of slices.
            **kwargs: Additional parameters.

        Returns:
            ExecutionSchedule.
        """
        if len(historical_volumes) == 0:
            raise ScheduleError("historical_volumes cannot be empty")

        timestamps = np.linspace(0, duration_seconds, len(historical_volumes) + 1)
        volume_profile = VolumeProfile(
            timestamps=timestamps,
            volumes=historical_volumes,
            total_volume=float(np.sum(historical_volumes)),
        )
        return self.generate_schedule(
            total_quantity, duration_seconds, volume_profile, num_slices, **kwargs
        )


# ---------------------------------------------------------------------------
# POV Algorithm
# ---------------------------------------------------------------------------

class POVAlgorithm(ExecutionAlgorithm):
    """Percentage of Volume execution algorithm.

    Participates at a fixed percentage of market volume, adjusting
    slice sizes in real-time based on observed volume to maintain
    the target participation rate.
    """

    def __init__(self, instrument: Instrument) -> None:
        """Initialize POV.

        Args:
            instrument: Trading instrument.
        """
        super().__init__(instrument)
        self._participation_rate: float = 0.1
        self._observed_volumes: list[float] = []

    def get_name(self) -> str:
        """Get algorithm name."""
        return "POV"

    @property
    def participation_rate(self) -> float:
        """Current participation rate."""
        return self._participation_rate

    def generate_schedule(
        self,
        total_quantity: float,
        duration_seconds: float,
        participation_rate: float = 0.1,
        num_slices: int = 10,
        price_limit: Optional[float] = None,
        **kwargs: float,
    ) -> ExecutionSchedule:
        """Generate a POV schedule.

        Args:
            total_quantity: Total quantity to execute.
            duration_seconds: Total duration in seconds.
            participation_rate: Target participation rate (0-1).
            num_slices: Number of slices.
            price_limit: Optional price limit.
            **kwargs: Additional parameters.

        Returns:
            ExecutionSchedule.
        """
        if total_quantity <= 0:
            raise ScheduleError(
                f"total_quantity must be positive, got {total_quantity}"
            )
        if duration_seconds <= 0:
            raise ScheduleError(
                f"duration_seconds must be positive, got {duration_seconds}"
            )
        if not 0 < participation_rate <= 1:
            raise ScheduleError(
                f"participation_rate must be in (0,1], got {participation_rate}"
            )

        self._participation_rate = participation_rate

        # Initial schedule: uniform slices
        # Actual quantities will be adjusted in real-time based on volume
        slice_qty = self.instrument.round_quantity(total_quantity / num_slices)
        interval = duration_seconds / num_slices

        slices: list[OrderSlice] = []
        for i in range(num_slices):
            t_start = i * interval
            t_end = t_start + interval
            slices.append(
                OrderSlice(
                    slice_id=i,
                    start_time=t_start,
                    end_time=t_end,
                    quantity=slice_qty,
                    price_limit=price_limit,
                    order_type=OrderType.LIMIT if price_limit else OrderType.MARKET,
                )
            )

        return ExecutionSchedule(
            slices=slices,
            total_quantity=slice_qty * num_slices,
            total_duration=duration_seconds,
            strategy=self.get_name(),
        )

    def compute_slice_quantity(
        self, observed_volume: float, remaining_quantity: float, remaining_slices: int
    ) -> float:
        """Compute slice quantity based on observed volume.

        Args:
            observed_volume: Current market volume.
            remaining_quantity: Remaining quantity to execute.
            remaining_slices: Number of remaining slices.

        Returns:
            Target quantity for this slice.
        """
        if observed_volume <= 0:
            return self.instrument.round_quantity(remaining_quantity / remaining_slices)

        target_qty = observed_volume * self._participation_rate
        max_qty = remaining_quantity / remaining_slices * 2.0  # Cap at 2x average
        qty = min(target_qty, max_qty, remaining_quantity)

        return self.instrument.round_quantity(qty)

    def update_participation_rate(self, new_rate: float) -> None:
        """Update the participation rate.

        Args:
            new_rate: New participation rate (0-1).
        """
        if not 0 < new_rate <= 1:
            raise ValidationError(
                f"participation_rate must be in (0,1], got {new_rate}"
            )
        self._participation_rate = new_rate

    def record_observed_volume(self, volume: float) -> None:
        """Record observed market volume.

        Args:
            volume: Observed volume.
        """
        if volume < 0:
            raise ValidationError(f"volume must be >= 0, got {volume}")
        self._observed_volumes.append(volume)

    def get_average_observed_volume(self) -> float:
        """Get average observed volume.

        Returns:
            Average observed volume.
        """
        if not self._observed_volumes:
            return 0.0
        return float(np.mean(self._observed_volumes))


# ---------------------------------------------------------------------------
# Smart Order Router
# ---------------------------------------------------------------------------

class SmartOrderRouter:
    """Smart Order Router for best execution across venues.

    Routes orders to the best available venue based on cost, liquidity,
    latency, and other factors.
    """

    def __init__(
        self,
        instrument: Instrument,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    ) -> None:
        """Initialize the SOR.

        Args:
            instrument: Trading instrument.
            strategy: Routing strategy.
        """
        self.instrument = instrument
        self.strategy = strategy
        self._venues: dict[str, Venue] = {}
        self._snapshots: dict[str, MarketSnapshot] = {}

    def add_venue(self, venue: Venue) -> None:
        """Add a venue.

        Args:
            venue: Venue to add.
        """
        self._venues[venue.name] = venue

    def remove_venue(self, venue_name: str) -> None:
        """Remove a venue.

        Args:
            venue_name: Venue to remove.
        """
        self._venues.pop(venue_name, None)
        self._snapshots.pop(venue_name, None)

    def update_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Update market snapshot for a venue.

        Args:
            snapshot: Market snapshot.
        """
        self._snapshots[snapshot.venue] = snapshot

    def route_order(
        self,
        quantity: float,
        side: OrderSide,
        order_type: OrderType = OrderType.LIMIT,
        price_limit: Optional[float] = None,
    ) -> list[tuple[str, float]]:
        """Route an order across venues.

        Args:
            quantity: Total quantity to route.
            side: Order side.
            order_type: Order type.
            price_limit: Optional price limit.

        Returns:
            List of (venue_name, quantity) tuples.
        """
        if quantity <= 0:
            raise RoutingError(f"quantity must be positive, got {quantity}")

        available = self._get_available_venues(side, price_limit)
        if not available:
            raise RoutingError("No available venues for routing")

        # Score venues
        scores = self._score_venues(available, side, quantity)

        # Allocate quantity proportional to scores
        total_score = sum(scores.values())
        if total_score <= 0:
            # Equal allocation
            allocations = [(name, quantity / len(available)) for name in available]
        else:
            allocations = [
                (name, quantity * scores[name] / total_score) for name in available
            ]

        # Round to lot size
        allocations = [
            (name, self.instrument.round_quantity(qty)) for name, qty in allocations
        ]

        return allocations

    def _get_available_venues(
        self, side: OrderSide, price_limit: Optional[float]
    ) -> list[str]:
        """Get available venues for routing.

        Args:
            side: Order side.
            price_limit: Optional price limit.

        Returns:
            List of available venue names.
        """
        available = []
        for name, venue in self._venues.items():
            if not venue.is_available:
                continue
            snapshot = self._snapshots.get(name)
            if snapshot is None:
                continue
            if price_limit is not None:
                if side == OrderSide.BUY and snapshot.ask > price_limit:
                    continue
                if side == OrderSide.SELL and snapshot.bid < price_limit:
                    continue
            available.append(name)
        return available

    def _score_venues(
        self, venue_names: list[str], side: OrderSide, quantity: float
    ) -> dict[str, float]:
        """Score venues for routing.

        Args:
            venue_names: Venue names to score.
            side: Order side.
            quantity: Order quantity.

        Returns:
            Dictionary of venue scores.
        """
        scores: dict[str, float] = {}

        for name in venue_names:
            venue = self._venues[name]
            snapshot = self._snapshots[name]

            if self.strategy == RoutingStrategy.COST_BASED:
                score = self._cost_score(venue, snapshot, side)
            elif self.strategy == RoutingStrategy.LIQUIDITY_BASED:
                score = self._liquidity_score(snapshot, side, quantity)
            elif self.strategy == RoutingStrategy.LATENCY_BASED:
                score = self._latency_score(venue)
            else:  # BALANCED
                score = (
                    self._cost_score(venue, snapshot, side)
                    + self._liquidity_score(snapshot, side, quantity)
                    + self._latency_score(venue)
                ) / 3.0

            scores[name] = max(score, 0.01)  # Minimum score

        return scores

    def _cost_score(self, venue: Venue, snapshot: MarketSnapshot, side: OrderSide) -> float:
        """Compute cost-based score.

        Args:
            venue: Venue.
            snapshot: Market snapshot.
            side: Order side.

        Returns:
            Cost score (higher = better).
        """
        # Lower cost = higher score
        cost = venue.net_cost_bps + snapshot.spread_bps
        return 1.0 / (1.0 + cost / 100.0)

    def _liquidity_score(
        self, snapshot: MarketSnapshot, side: OrderSide, quantity: float
    ) -> float:
        """Compute liquidity-based score.

        Args:
            snapshot: Market snapshot.
            side: Order side.
            quantity: Order quantity.

        Returns:
            Liquidity score (higher = better).
        """
        available = snapshot.bid_size if side == OrderSide.SELL else snapshot.ask_size
        if available <= 0:
            return 0.0
        return min(1.0, available / quantity)

    def _latency_score(self, venue: Venue) -> float:
        """Compute latency-based score.

        Args:
            venue: Venue.

        Returns:
            Latency score (higher = better).
        """
        return 1.0 / (1.0 + venue.latency_ms / 10.0)

    def get_best_venue(self, side: OrderSide) -> Optional[str]:
        """Get the best venue for a given side.

        Args:
            side: Order side.

        Returns:
            Best venue name, or None if no venues available.
        """
        available = self._get_available_venues(side, None)
        if not available:
            return None

        scores = self._score_venues(available, side, 1.0)
        return max(scores, key=scores.get)

    def get_venue_count(self) -> int:
        """Get number of configured venues.

        Returns:
            Number of venues.
        """
        return len(self._venues)


# ---------------------------------------------------------------------------
# Execution Manager
# ---------------------------------------------------------------------------

class ExecutionManager:
    """High-level execution manager combining all algorithms.

    Provides a unified interface for:
    - Schedule generation (TWAP, VWAP, POV)
    - Order routing (SOR)
    - Execution monitoring and analytics
    """

    def __init__(self, instrument: Instrument) -> None:
        """Initialize the execution manager.

        Args:
            instrument: Trading instrument.
        """
        self.instrument = instrument
        self.twap = TWAPAlgorithm(instrument)
        self.vwap = VWAPAlgorithm(instrument)
        self.pov = POVAlgorithm(instrument)
        self.sor = SmartOrderRouter(instrument)

    def generate_twap_schedule(
        self,
        total_quantity: float,
        duration_seconds: float,
        num_slices: int = 10,
        **kwargs: float,
    ) -> ExecutionSchedule:
        """Generate a TWAP schedule.

        Args:
            total_quantity: Total quantity to execute.
            duration_seconds: Total duration in seconds.
            num_slices: Number of slices.
            **kwargs: Additional parameters.

        Returns:
            ExecutionSchedule.
        """
        return self.twap.generate_schedule(
            total_quantity, duration_seconds, num_slices=num_slices, **kwargs
        )

    def generate_vwap_schedule(
        self,
        total_quantity: float,
        duration_seconds: float,
        volume_profile: Optional[VolumeProfile] = None,
        num_slices: int = 10,
        **kwargs: float,
    ) -> ExecutionSchedule:
        """Generate a VWAP schedule.

        Args:
            total_quantity: Total quantity to execute.
            duration_seconds: Total duration in seconds.
            volume_profile: Expected volume profile.
            num_slices: Number of slices.
            **kwargs: Additional parameters.

        Returns:
            ExecutionSchedule.
        """
        return self.vwap.generate_schedule(
            total_quantity, duration_seconds, volume_profile, num_slices, **kwargs
        )

    def generate_pov_schedule(
        self,
        total_quantity: float,
        duration_seconds: float,
        participation_rate: float = 0.1,
        num_slices: int = 10,
        **kwargs: float,
    ) -> ExecutionSchedule:
        """Generate a POV schedule.

        Args:
            total_quantity: Total quantity to execute.
            duration_seconds: Total duration in seconds.
            participation_rate: Target participation rate.
            num_slices: Number of slices.
            **kwargs: Additional parameters.

        Returns:
            ExecutionSchedule.
        """
        return self.pov.generate_schedule(
            total_quantity,
            duration_seconds,
            participation_rate=participation_rate,
            num_slices=num_slices,
            **kwargs,
        )

    def route_order(
        self,
        quantity: float,
        side: OrderSide,
        order_type: OrderType = OrderType.LIMIT,
        price_limit: Optional[float] = None,
    ) -> list[tuple[str, float]]:
        """Route an order using SOR.

        Args:
            quantity: Total quantity to route.
            side: Order side.
            order_type: Order type.
            price_limit: Optional price limit.

        Returns:
            List of (venue_name, quantity) tuples.
        """
        return self.sor.route_order(quantity, side, order_type, price_limit)

    def get_analytics(self, algorithm_name: str, arrival_price: float) -> ExecutionAnalytics:
        """Get analytics for an algorithm.

        Args:
            algorithm_name: Name of the algorithm.
            arrival_price: Arrival price.

        Returns:
            ExecutionAnalytics.
        """
        algo = {
            "TWAP": self.twap,
            "VWAP": self.vwap,
            "POV": self.pov,
        }.get(algorithm_name)

        if algo is None:
            raise ExecutionError(f"Unknown algorithm: {algorithm_name}")

        return algo.get_analytics(arrival_price)

    def reset(self) -> None:
        """Reset all algorithms."""
        self.twap.reset()
        self.vwap.reset()
        self.pov.reset()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def compute_implementation_shortfall(
    arrival_price: float,
    execution_prices: list[float],
    quantities: list[float],
) -> float:
    """Compute implementation shortfall.

    Args:
        arrival_price: Price at decision time.
        execution_prices: List of execution prices.
        quantities: List of execution quantities.

    Returns:
        Implementation shortfall in basis points.
    """
    if len(execution_prices) != len(quantities):
        raise ValidationError("execution_prices and quantities must have same length")
    if not quantities:
        return 0.0

    total_qty = sum(quantities)
    if total_qty <= 0:
        return 0.0

    vwap = sum(p * q for p, q in zip(execution_prices, quantities)) / total_qty
    return ((vwap - arrival_price) / arrival_price) * 10_000 if arrival_price > 0 else 0.0


def compute_arrival_price_shortfall(
    arrival_price: float,
    avg_execution_price: float,
) -> float:
    """Compute arrival price shortfall.

    Args:
        arrival_price: Price at decision time.
        avg_execution_price: Average execution price.

    Returns:
        Shortfall in basis points.
    """
    if arrival_price <= 0:
        return 0.0
    return ((avg_execution_price - arrival_price) / arrival_price) * 10_000


def compute_market_impact(
    pre_trade_mid: float,
    post_trade_mid: float,
) -> float:
    """Compute market impact.

    Args:
        pre_trade_mid: Mid price before trade.
        post_trade_mid: Mid price after trade.

    Returns:
        Market impact in basis points.
    """
    if pre_trade_mid <= 0:
        return 0.0
    return ((post_trade_mid - pre_trade_mid) / pre_trade_mid) * 10_000


def create_default_volume_profile(
    duration_seconds: float,
    num_points: int = 24,
    peak_ratio: float = 2.0,
) -> VolumeProfile:
    """Create a default volume profile with U-shape.

    Args:
        duration_seconds: Total duration.
        num_points: Number of profile points.
        peak_ratio: Peak-to-average ratio.

    Returns:
        VolumeProfile.
    """
    timestamps = np.linspace(0, duration_seconds, num_points)
    # U-shape: higher volume at open and close
    base = np.ones(num_points)
    # Add U-shape component
    x = np.linspace(0, 1, num_points)
    u_shape = peak_ratio * (4 * (x - 0.5) ** 2)
    volumes = base + u_shape

    return VolumeProfile(
        timestamps=timestamps,
        volumes=volumes,
        total_volume=float(np.sum(volumes)),
    )


def create_instrument(
    symbol: str,
    exchange: str = "NYSE",
    lot_size: float = 1.0,
    tick_size: float = 0.01,
) -> Instrument:
    """Create a trading instrument.

    Args:
        symbol: Instrument symbol.
        exchange: Primary exchange.
        lot_size: Minimum tradable quantity.
        tick_size: Minimum price increment.

    Returns:
        Instrument.
    """
    return Instrument(
        symbol=symbol,
        exchange=exchange,
        lot_size=lot_size,
        tick_size=tick_size,
    )


__all__ = [
    "ExecutionError",
    "ScheduleError",
    "RoutingError",
    "ValidationError",
    "OrderSide",
    "OrderType",
    "ExecutionStatus",
    "RoutingStrategy",
    "VenueType",
    "Instrument",
    "Venue",
    "MarketSnapshot",
    "OrderSlice",
    "ExecutionResult",
    "ExecutionSchedule",
    "ExecutionAnalytics",
    "VolumeProfile",
    "ExecutionAlgorithm",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "POVAlgorithm",
    "SmartOrderRouter",
    "ExecutionManager",
    "compute_implementation_shortfall",
    "compute_arrival_price_shortfall",
    "compute_market_impact",
    "create_default_volume_profile",
    "create_instrument",
]
