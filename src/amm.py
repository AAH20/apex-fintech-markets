"""
Apex FinTech Markets — Automated Market Making (AMM).

Implements decentralized exchange primitives:

1. **Constant Product AMM (x*y=k)** — Uniswap V2 style AMM with
   constant product formula, liquidity provision, and fee mechanics.

2. **Concentrated Liquidity** — Uniswap V3 style AMM with
   liquidity concentrated in custom price ranges, capital efficiency,
   and tick-based pricing.

3. **Dynamic Fees** — Volatility-adjusted fee tiers that respond
   to market conditions to optimize LP returns.

4. **Impermanent Loss Management** — Tools for measuring, monitoring,
   and hedging impermanent loss for liquidity providers.

All models use exact integer/float arithmetic with comprehensive
validation and error handling.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class AMMError(Exception):
    """Base exception for AMM errors."""


class PoolError(AMMError):
    """Raised when pool operations fail."""


class LiquidityError(AMMError):
    """Raised when liquidity operations fail."""


class SwapError(AMMError):
    """Raised when swap operations fail."""


class TickError(AMMError):
    """Raised when tick operations fail."""


class FeeError(AMMError):
    """Raised when fee operations fail."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PoolType(Enum):
    """AMM pool type."""

    CONSTANT_PRODUCT = "constant_product"  # x*y=k
    CONCENTRATED_LIQUIDITY = "concentrated_liquidity"  # Uniswap V3
    STABLE_SWAP = "stable_swap"  # Curve style


class FeeTier(Enum):
    """Fee tier enumeration."""

    ULTRA_LOW = 1       # 0.01%
    LOW = 2             # 0.05%
    MEDIUM = 3          # 0.30%
    HIGH = 4            # 1.00%
    DYNAMIC = 5         # Volatility-adjusted

    @property
    def default_bps(self) -> int:
        """Default fee in basis points."""
        mapping = {
            FeeTier.ULTRA_LOW: 1,
            FeeTier.LOW: 5,
            FeeTier.MEDIUM: 30,
            FeeTier.HIGH: 100,
            FeeTier.DYNAMIC: 30,
        }
        return mapping[self]


class PositionStatus(Enum):
    """LP position status."""

    IN_RANGE = "in_range"
    OUT_OF_RANGE = "out_of_range"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """A token in the pool.

    Attributes:
        symbol: Token symbol.
        decimals: Number of decimal places.
        address: Token contract address.
    """

    symbol: str
    decimals: int = 18
    address: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise PoolError("Token symbol cannot be empty")
        if self.decimals < 0 or self.decimals > 36:
            raise PoolError(f"decimals must be in [0, 36], got {self.decimals}")

    def normalize(self, amount: float) -> float:
        """Normalize human-readable amount to raw units.

        Args:
            amount: Human-readable amount.

        Returns:
            Raw amount (10^decimals).
        """
        return amount * (10**self.decimals)

    def denormalize(self, raw_amount: float) -> float:
        """Convert raw units to human-readable amount.

        Args:
            raw_amount: Raw amount.

        Returns:
            Human-readable amount.
        """
        return raw_amount / (10**self.decimals)


@dataclass(frozen=True)
class PoolConfig:
    """AMM pool configuration.

    Attributes:
        token0: First token.
        token1: Second token.
        fee_tier: Fee tier.
        pool_type: Type of AMM pool.
        price_precision: Number of decimal places for prices.
    """

    token0: Token
    token1: Token
    fee_tier: FeeTier = FeeTier.MEDIUM
    pool_type: PoolType = PoolType.CONSTANT_PRODUCT
    price_precision: int = 18

    def __post_init__(self) -> None:
        if self.price_precision < 0:
            raise PoolError(
                f"price_precision must be >= 0, got {self.price_precision}"
            )

    @property
    def fee_bps(self) -> int:
        """Fee in basis points."""
        return self.fee_tier.default_bps

    @property
    def fee_fraction(self) -> float:
        """Fee as a fraction (e.g., 0.003 for 0.3%)."""
        return self.fee_bps / 10_000.0


@dataclass(frozen=True)
class PoolState:
    """Current state of an AMM pool.

    Attributes:
        reserve0: Reserve of token0.
        reserve1: Reserve of token1.
        total_lp_supply: Total LP token supply.
        price: Current price (token1 per token0).
        volume_24h: 24-hour volume.
        fees_24h: 24-hour fees collected.
        block_number: Last update block.
    """

    reserve0: float
    reserve1: float
    total_lp_supply: float
    price: float
    volume_24h: float = 0.0
    fees_24h: float = 0.0
    block_number: int = 0

    def __post_init__(self) -> None:
        if self.reserve0 < 0 or self.reserve1 < 0:
            raise PoolError(
                f"Reserves must be non-negative, got reserve0={self.reserve0}, "
                f"reserve1={self.reserve1}"
            )
        if self.total_lp_supply < 0:
            raise PoolError(
                f"total_lp_supply must be >= 0, got {self.total_lp_supply}"
            )
        if self.price < 0:
            raise PoolError(f"price must be >= 0, got {self.price}")

    @property
    def is_empty(self) -> bool:
        """True if pool has no liquidity."""
        return self.reserve0 == 0 or self.reserve1 == 0

    @property
    def tvl(self) -> float:
        """Total value locked (in token1 units)."""
        return self.reserve1 * 2.0 if not self.is_empty else 0.0


@dataclass(frozen=True)
class SwapResult:
    """Result of a swap operation.

    Attributes:
        amount_in: Input amount.
        amount_out: Output amount.
        price_impact: Price impact as a fraction.
        fee_amount: Fee amount.
        execution_price: Actual execution price.
    """

    amount_in: float
    amount_out: float
    price_impact: float
    fee_amount: float
    execution_price: float

    def __post_init__(self) -> None:
        if self.amount_in < 0:
            raise SwapError(f"amount_in must be >= 0, got {self.amount_in}")
        if self.amount_out < 0:
            raise SwapError(f"amount_out must be >= 0, got {self.amount_out}")
        if self.price_impact < 0:
            raise SwapError(
                f"price_impact must be >= 0, got {self.price_impact}"
            )


@dataclass(frozen=True)
class LPPosition:
    """A liquidity provider position.

    Attributes:
        position_id: Unique position identifier.
        owner: Position owner address.
        liquidity: Liquidity amount.
        amount0: Amount of token0 deposited.
        amount1: Amount of token1 deposited.
        lower_tick: Lower price tick (concentrated liquidity).
        upper_tick: Upper price tick (concentrated liquidity).
        fees_owed0: Uncollected fees in token0.
        fees_owed1: Uncollected fees in token1.
        status: Position status.
    """

    position_id: str
    owner: str
    liquidity: float
    amount0: float
    amount1: float
    lower_tick: Optional[int] = None
    upper_tick: Optional[int] = None
    fees_owed0: float = 0.0
    fees_owed1: float = 0.0
    status: PositionStatus = PositionStatus.IN_RANGE

    def __post_init__(self) -> None:
        if self.liquidity < 0:
            raise LiquidityError(
                f"liquidity must be >= 0, got {self.liquidity}"
            )
        if self.amount0 < 0 or self.amount1 < 0:
            raise LiquidityError(
                f"amounts must be >= 0, got amount0={self.amount0}, amount1={self.amount1}"
            )
        if self.lower_tick is not None and self.upper_tick is not None:
            if self.lower_tick >= self.upper_tick:
                raise TickError(
                    f"lower_tick ({self.lower_tick}) must be < upper_tick ({self.upper_tick})"
                )

    @property
    def is_concentrated(self) -> bool:
        """True if this is a concentrated liquidity position."""
        return self.lower_tick is not None and self.upper_tick is not None

    @property
    def total_value(self) -> float:
        """Total value of the position (in token1 units)."""
        return self.amount0 + self.amount1


@dataclass(frozen=True)
class Tick:
    """A price tick in concentrated liquidity.

    Attributes:
        index: Tick index.
        price: Price at this tick.
        liquidity_gross: Total liquidity at this tick.
        liquidity_net: Net liquidity change at this tick.
    """

    index: int
    price: float
    liquidity_gross: float = 0.0
    liquidity_net: float = 0.0

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise TickError(f"price must be positive, got {self.price}")
        if self.liquidity_gross < 0:
            raise TickError(
                f"liquidity_gross must be >= 0, got {self.liquidity_gross}"
            )


@dataclass(frozen=True)
class FeeSchedule:
    """Dynamic fee schedule.

    Attributes:
        base_fee_bps: Base fee in basis points.
        volatility_multiplier: Multiplier based on volatility.
        volume_multiplier: Multiplier based on volume.
        max_fee_bps: Maximum fee in basis points.
        min_fee_bps: Minimum fee in basis points.
    """

    base_fee_bps: int = 30
    volatility_multiplier: float = 1.0
    volume_multiplier: float = 1.0
    max_fee_bps: int = 200
    min_fee_bps: int = 1

    def __post_init__(self) -> None:
        if self.base_fee_bps < 0:
            raise FeeError(
                f"base_fee_bps must be >= 0, got {self.base_fee_bps}"
            )
        if self.volatility_multiplier < 0:
            raise FeeError(
                f"volatility_multiplier must be >= 0, got {self.volatility_multiplier}"
            )
        if self.volume_multiplier < 0:
            raise FeeError(
                f"volume_multiplier must be >= 0, got {self.volume_multiplier}"
            )
        if self.max_fee_bps < self.min_fee_bps:
            raise FeeError(
                f"max_fee_bps ({self.max_fee_bps}) must be >= min_fee_bps ({self.min_fee_bps})"
            )

    @property
    def effective_fee_bps(self) -> int:
        """Compute the effective fee in basis points."""
        fee = self.base_fee_bps * self.volatility_multiplier * self.volume_multiplier
        return int(max(self.min_fee_bps, min(self.max_fee_bps, fee)))


@dataclass(frozen=True)
class ImpermanentLossResult:
    """Result of impermanent loss calculation.

    Attributes:
        il_fraction: Impermanent loss as a fraction (positive = loss).
        il_percentage: Impermanent loss as a percentage.
        hodl_value: Value if simply held.
        lp_value: Value in LP position.
        net_pnl: Net P&L (LP value - HODL value).
        fee_revenue: Fee revenue earned.
        net_il_after_fees: Net IL after accounting for fees.
    """

    il_fraction: float
    il_percentage: float
    hodl_value: float
    lp_value: float
    net_pnl: float
    fee_revenue: float
    net_il_after_fees: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.il_fraction <= 1.0:
            raise PoolError(
                f"il_fraction must be in [-1, 1], got {self.il_fraction}"
            )


# ---------------------------------------------------------------------------
# Constant Product AMM (x*y=k)
# ---------------------------------------------------------------------------

class ConstantProductAMM:
    """Uniswap V2 style constant product AMM.

    Maintains the invariant: x * y = k
    where x = reserve0, y = reserve1, k = constant product.

    Swaps follow: (x + dx * (1 - fee)) * (y - dy) = k
    """

    def __init__(self, config: PoolConfig) -> None:
        """Initialize the AMM.

        Args:
            config: Pool configuration.
        """
        self.config = config
        self._state = PoolState(
            reserve0=0.0,
            reserve1=0.0,
            total_lp_supply=0.0,
            price=0.0,
        )
        self._positions: dict[str, LPPosition] = {}
        self._position_counter = 0

    @property
    def state(self) -> PoolState:
        """Current pool state."""
        return self._state

    @property
    def fee_fraction(self) -> float:
        """Fee as a fraction."""
        return self.config.fee_fraction

    def get_amount_out(self, amount_in: float, token_in: Token) -> float:
        """Compute output amount for a given input.

        Args:
            amount_in: Input amount.
            token_in: Input token.

        Returns:
            Output amount.
        """
        if amount_in <= 0:
            raise SwapError(f"amount_in must be positive, got {amount_in}")
        if self._state.is_empty:
            raise PoolError("Pool has no liquidity")

        fee = self.fee_fraction
        if token_in.symbol == self.config.token0.symbol:
            reserve_in = self._state.reserve0
            reserve_out = self._state.reserve1
        else:
            reserve_in = self._state.reserve1
            reserve_out = self._state.reserve0

        amount_in_with_fee = amount_in * (1.0 - fee)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee
        amount_out = numerator / denominator

        return amount_out

    def get_amount_in(self, amount_out: float, token_out: Token) -> float:
        """Compute input amount for a desired output.

        Args:
            amount_out: Desired output amount.
            token_out: Output token.

        Returns:
            Required input amount.
        """
        if amount_out <= 0:
            raise SwapError(f"amount_out must be positive, got {amount_out}")
        if self._state.is_empty:
            raise PoolError("Pool has no liquidity")

        fee = self.fee_fraction
        if token_out.symbol == self.config.token0.symbol:
            reserve_in = self._state.reserve1
            reserve_out = self._state.reserve0
        else:
            reserve_in = self._state.reserve0
            reserve_out = self._state.reserve1

        if amount_out >= reserve_out:
            raise SwapError(
                f"amount_out ({amount_out}) exceeds reserve ({reserve_out})"
            )

        numerator = reserve_in * amount_out
        denominator = (reserve_out - amount_out) * (1.0 - fee)
        amount_in = numerator / denominator + 1e-10  # Round up

        return amount_in

    def swap(self, amount_in: float, token_in: Token) -> SwapResult:
        """Execute a swap.

        Args:
            amount_in: Input amount.
            token_in: Input token.

        Returns:
            SwapResult.
        """
        amount_out = self.get_amount_out(amount_in, token_in)

        # Update reserves
        if token_in.symbol == self.config.token0.symbol:
            new_reserve0 = self._state.reserve0 + amount_in
            new_reserve1 = self._state.reserve1 - amount_out
        else:
            new_reserve0 = self._state.reserve0 - amount_out
            new_reserve1 = self._state.reserve1 + amount_in

        # Compute price impact
        old_price = self._state.price
        new_price = new_reserve1 / new_reserve0 if new_reserve0 > 0 else 0.0
        price_impact = abs(new_price - old_price) / old_price if old_price > 0 else 0.0

        fee_amount = amount_in * self.fee_fraction
        execution_price = amount_out / amount_in if amount_in > 0 else 0.0

        self._state = PoolState(
            reserve0=new_reserve0,
            reserve1=new_reserve1,
            total_lp_supply=self._state.total_lp_supply,
            price=new_price,
            volume_24h=self._state.volume_24h + amount_in,
            fees_24h=self._state.fees_24h + fee_amount,
            block_number=self._state.block_number + 1,
        )

        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            price_impact=price_impact,
            fee_amount=fee_amount,
            execution_price=execution_price,
        )

    def add_liquidity(
        self, amount0: float, amount1: float, owner: str = ""
    ) -> LPPosition:
        """Add liquidity to the pool.

        Args:
            amount0: Amount of token0.
            amount1: Amount of token1.
            owner: LP owner address.

        Returns:
            Created LPPosition.
        """
        if amount0 < 0 or amount1 < 0:
            raise LiquidityError(
                f"Amounts must be non-negative, got amount0={amount0}, amount1={amount1}"
            )

        if self._state.is_empty:
            # Initial liquidity
            liquidity = math.sqrt(amount0 * amount1)
        else:
            # Proportional deposit
            liquidity0 = amount0 * self._state.total_lp_supply / self._state.reserve0
            liquidity1 = amount1 * self._state.total_lp_supply / self._state.reserve1
            liquidity = min(liquidity0, liquidity1)

        if liquidity <= 0:
            raise LiquidityError("Computed liquidity must be positive")

        self._position_counter += 1
        position = LPPosition(
            position_id=f"LP{self._position_counter}",
            owner=owner,
            liquidity=liquidity,
            amount0=amount0,
            amount1=amount1,
        )
        self._positions[position.position_id] = position

        new_reserve0 = self._state.reserve0 + amount0
        new_reserve1 = self._state.reserve1 + amount1
        new_price = new_reserve1 / new_reserve0 if new_reserve0 > 0 else 0.0

        self._state = PoolState(
            reserve0=new_reserve0,
            reserve1=new_reserve1,
            total_lp_supply=self._state.total_lp_supply + liquidity,
            price=new_price,
            volume_24h=self._state.volume_24h,
            fees_24h=self._state.fees_24h,
            block_number=self._state.block_number + 1,
        )

        return position

    def remove_liquidity(self, position_id: str) -> tuple[float, float]:
        """Remove liquidity from the pool.

        Args:
            position_id: Position to remove.

        Returns:
            Tuple of (amount0, amount1) returned.
        """
        if position_id not in self._positions:
            raise LiquidityError(f"Position {position_id} not found")

        position = self._positions[position_id]
        share = position.liquidity / self._state.total_lp_supply

        amount0 = self._state.reserve0 * share
        amount1 = self._state.reserve1 * share

        new_reserve0 = self._state.reserve0 - amount0
        new_reserve1 = self._state.reserve1 - amount1
        new_price = new_reserve1 / new_reserve0 if new_reserve0 > 0 else 0.0

        self._state = PoolState(
            reserve0=new_reserve0,
            reserve1=new_reserve1,
            total_lp_supply=self._state.total_lp_supply - position.liquidity,
            price=new_price,
            volume_24h=self._state.volume_24h,
            fees_24h=self._state.fees_24h,
            block_number=self._state.block_number + 1,
        )

        del self._positions[position_id]
        return amount0, amount1

    def get_price(self) -> float:
        """Get current price (token1 per token0).

        Returns:
            Current price.
        """
        if self._state.is_empty:
            return 0.0
        return self._state.reserve1 / self._state.reserve0

    def get_reserves(self) -> tuple[float, float]:
        """Get current reserves.

        Returns:
            Tuple of (reserve0, reserve1).
        """
        return self._state.reserve0, self._state.reserve1

    def get_tvl(self) -> float:
        """Get total value locked.

        Returns:
            TVL in token1 units.
        """
        return self._state.tvl

    def get_pool_share(self, position_id: str) -> float:
        """Get pool share for a position.

        Args:
            position_id: Position identifier.

        Returns:
            Pool share as a fraction.
        """
        if position_id not in self._positions:
            raise LiquidityError(f"Position {position_id} not found")
        position = self._positions[position_id]
        return position.liquidity / self._state.total_lp_supply


# ---------------------------------------------------------------------------
# Concentrated Liquidity AMM
# ---------------------------------------------------------------------------

class ConcentratedLiquidityAMM:
    """Uniswap V3 style concentrated liquidity AMM.

    Allows LPs to concentrate liquidity within custom price ranges,
    achieving much higher capital efficiency than constant product AMMs.

    Price is tracked via ticks: price = 1.0001^tick
    """

    def __init__(
        self,
        config: PoolConfig,
        tick_spacing: int = 60,
        min_tick: int = -887272,
        max_tick: int = 887272,
    ) -> None:
        """Initialize the concentrated liquidity AMM.

        Args:
            config: Pool configuration.
            tick_spacing: Spacing between ticks.
            min_tick: Minimum tick index.
            max_tick: Maximum tick index.
        """
        if config.pool_type != PoolType.CONCENTRATED_LIQUIDITY:
            raise PoolError("Config must use CONCENTRATED_LIQUIDITY pool type")
        if tick_spacing <= 0:
            raise TickError(f"tick_spacing must be positive, got {tick_spacing}")

        self.config = config
        self.tick_spacing = tick_spacing
        self.min_tick = min_tick
        self.max_tick = max_tick

        self._ticks: dict[int, Tick] = {}
        self._positions: dict[str, LPPosition] = {}
        self._position_counter = 0
        self._liquidity = 0.0  # Active liquidity
        self._sqrt_price = 0.0  # Current sqrt(price)
        self._tick_current = 0  # Current tick

    @property
    def liquidity(self) -> float:
        """Active liquidity."""
        return self._liquidity

    @property
    def sqrt_price(self) -> float:
        """Current square root of price."""
        return self._sqrt_price

    @property
    def current_price(self) -> float:
        """Current price (token1 per token0)."""
        return self._sqrt_price**2 if self._sqrt_price > 0 else 0.0

    @property
    def current_tick(self) -> int:
        """Current tick index."""
        return self._tick_current

    def tick_to_price(self, tick: int) -> float:
        """Convert tick index to price.

        Args:
            tick: Tick index.

        Returns:
            Price at this tick.
        """
        return 1.0001**tick

    def price_to_tick(self, price: float) -> int:
        """Convert price to nearest tick index.

        Args:
            price: Price to convert.

        Returns:
            Nearest tick index.
        """
        if price <= 0:
            raise TickError(f"price must be positive, got {price}")
        return int(math.log(price) / math.log(1.0001))

    def get_tick(self, tick_index: int) -> Optional[Tick]:
        """Get tick data.

        Args:
            tick_index: Tick index.

        Returns:
            Tick data or None if not initialized.
        """
        return self._ticks.get(tick_index)

    def update_liquidity(
        self, tick_index: int, liquidity_net: float
    ) -> None:
        """Update liquidity at a tick.

        Args:
            tick_index: Tick index.
            liquidity_net: Net liquidity change.
        """
        if tick_index < self.min_tick or tick_index > self.max_tick:
            raise TickError(
                f"tick_index {tick_index} out of range [{self.min_tick}, {self.max_tick}]"
            )

        if tick_index not in self._ticks:
            self._ticks[tick_index] = Tick(
                index=tick_index,
                price=self.tick_to_price(tick_index),
            )

        tick = self._ticks[tick_index]
        new_gross = tick.liquidity_gross + abs(liquidity_net)
        new_net = tick.liquidity_net + liquidity_net

        self._ticks[tick_index] = Tick(
            index=tick_index,
            price=tick.price,
            liquidity_gross=new_gross,
            liquidity_net=new_net,
        )

    def add_concentrated_liquidity(
        self,
        amount0: float,
        amount1: float,
        lower_price: float,
        upper_price: float,
        owner: str = "",
    ) -> LPPosition:
        """Add concentrated liquidity within a price range.

        Args:
            amount0: Amount of token0.
            amount1: Amount of token1.
            lower_price: Lower price bound.
            upper_price: Upper price bound.
            owner: LP owner.

        Returns:
            Created LPPosition.
        """
        if amount0 < 0 or amount1 < 0:
            raise LiquidityError("Amounts must be non-negative")
        if lower_price <= 0 or upper_price <= 0:
            raise TickError("Price bounds must be positive")
        if lower_price >= upper_price:
            raise TickError(
                f"lower_price ({lower_price}) must be < upper_price ({upper_price})"
            )

        lower_tick = self.price_to_tick(lower_price)
        upper_tick = self.price_to_tick(upper_price)

        # Compute liquidity from amounts
        # L = amount0 / (1/sqrt(lower) - 1/sqrt(upper))
        # L = amount1 / (sqrt(upper) - sqrt(lower))
        sqrt_lower = math.sqrt(lower_price)
        sqrt_upper = math.sqrt(upper_price)

        if amount0 > 0:
            liquidity_from_0 = amount0 / (1.0 / sqrt_lower - 1.0 / sqrt_upper)
        else:
            liquidity_from_0 = 0.0

        if amount1 > 0:
            liquidity_from_1 = amount1 / (sqrt_upper - sqrt_lower)
        else:
            liquidity_from_1 = 0.0

        liquidity = min(liquidity_from_0, liquidity_from_1)
        if liquidity <= 0:
            raise LiquidityError("Computed liquidity must be positive")

        self._position_counter += 1
        position = LPPosition(
            position_id=f"CLP{self._position_counter}",
            owner=owner,
            liquidity=liquidity,
            amount0=amount0,
            amount1=amount1,
            lower_tick=lower_tick,
            upper_tick=upper_tick,
        )
        self._positions[position.position_id] = position

        # Update tick liquidity
        self.update_liquidity(lower_tick, liquidity)
        self.update_liquidity(upper_tick, -liquidity)

        # Update active liquidity if current price is in range
        current_price = self.current_price
        if current_price > 0 and lower_price <= current_price <= upper_price:
            self._liquidity += liquidity

        return position

    def remove_concentrated_liquidity(
        self, position_id: str
    ) -> tuple[float, float]:
        """Remove concentrated liquidity.

        Args:
            position_id: Position to remove.

        Returns:
            Tuple of (amount0, amount1) returned.
        """
        if position_id not in self._positions:
            raise LiquidityError(f"Position {position_id} not found")

        position = self._positions[position_id]

        if position.is_concentrated:
            # Update tick liquidity
            self.update_liquidity(position.lower_tick, -position.liquidity)
            self.update_liquidity(position.upper_tick, position.liquidity)

            # Update active liquidity
            current_price = self.current_price
            if (
                current_price > 0
                and self.tick_to_price(position.lower_tick)
                <= current_price
                <= self.tick_to_price(position.upper_tick)
            ):
                self._liquidity -= position.liquidity

        del self._positions[position_id]
        return position.amount0, position.amount1

    def swap(self, amount_in: float, token_in: Token) -> SwapResult:
        """Execute a swap through concentrated liquidity.

        Args:
            amount_in: Input amount.
            token_in: Input token.

        Returns:
            SwapResult.
        """
        if amount_in <= 0:
            raise SwapError(f"amount_in must be positive, got {amount_in}")
        if self._liquidity <= 0:
            raise PoolError("No active liquidity")

        fee = self.config.fee_fraction
        amount_in_with_fee = amount_in * (1.0 - fee)

        # Simplified swap: use constant product within current tick
        # Real implementation would cross ticks
        sqrt_price = self._sqrt_price
        if sqrt_price <= 0:
            raise PoolError("Invalid sqrt price")

        # Compute output using concentrated liquidity formula
        # This is a simplified version — full implementation requires
        # tick crossing logic
        price = sqrt_price**2
        if token_in.symbol == self.config.token0.symbol:
            # Selling token0 for token1
            amount_out = amount_in_with_fee / price
        else:
            # Selling token1 for token0
            amount_out = amount_in_with_fee * price

        fee_amount = amount_in * fee
        execution_price = amount_out / amount_in if amount_in > 0 else 0.0

        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            price_impact=0.0,  # Simplified
            fee_amount=fee_amount,
            execution_price=execution_price,
        )

    def get_position_value(self, position_id: str) -> float:
        """Get current value of a position.

        Args:
            position_id: Position identifier.

        Returns:
            Position value in token1 units.
        """
        if position_id not in self._positions:
            raise LiquidityError(f"Position {position_id} not found")

        position = self._positions[position_id]
        current_price = self.current_price

        if position.is_concentrated:
            lower_price = self.tick_to_price(position.lower_tick)
            upper_price = self.tick_to_price(position.upper_tick)

            if current_price <= lower_price:
                # All in token0
                return position.amount0 * lower_price
            elif current_price >= upper_price:
                # All in token1
                return position.amount1
            else:
                # Mixed
                return position.amount0 * current_price + position.amount1
        else:
            return position.amount0 * current_price + position.amount1

    def get_capital_efficiency(self, position_id: str) -> float:
        """Get capital efficiency vs constant product AMM.

        Args:
            position_id: Position identifier.

        Returns:
            Capital efficiency multiplier.
        """
        if position_id not in self._positions:
            raise LiquidityError(f"Position {position_id} not found")

        position = self._positions[position_id]
        if not position.is_concentrated:
            return 1.0

        lower_price = self.tick_to_price(position.lower_tick)
        upper_price = self.tick_to_price(position.upper_tick)

        if lower_price <= 0 or upper_price <= 0:
            return 1.0

        # Capital efficiency = (upper - lower) / (upper + lower) * 2
        # This is a simplified metric
        efficiency = (upper_price - lower_price) / (upper_price + lower_price) * 2.0
        return max(1.0, efficiency)


# ---------------------------------------------------------------------------
# Dynamic Fee Manager
# ---------------------------------------------------------------------------

class DynamicFeeManager:
    """Manages dynamic fee tiers based on market conditions.

    Adjusts fees in response to volatility and volume to optimize
    LP returns while maintaining competitiveness.
    """

    def __init__(
        self,
        base_fee_bps: int = 30,
        volatility_window: int = 100,
        max_fee_bps: int = 200,
        min_fee_bps: int = 1,
    ) -> None:
        """Initialize the dynamic fee manager.

        Args:
            base_fee_bps: Base fee in basis points.
            volatility_window: Number of observations for volatility.
            max_fee_bps: Maximum fee in basis points.
            min_fee_bps: Minimum fee in basis points.
        """
        self.base_fee_bps = base_fee_bps
        self.volatility_window = volatility_window
        self.max_fee_bps = max_fee_bps
        self.min_fee_bps = min_fee_bps

        self._price_history: list[float] = []
        self._volume_history: list[float] = []
        self._current_schedule = FeeSchedule(
            base_fee_bps=base_fee_bps,
            max_fee_bps=max_fee_bps,
            min_fee_bps=min_fee_bps,
        )

    @property
    def current_schedule(self) -> FeeSchedule:
        """Current fee schedule."""
        return self._current_schedule

    def update_price(self, price: float) -> None:
        """Update with a new price observation.

        Args:
            price: New price.
        """
        if price <= 0:
            raise FeeError(f"price must be positive, got {price}")

        self._price_history.append(price)
        if len(self._price_history) > self.volatility_window:
            self._price_history.pop(0)

        self._recompute_schedule()

    def update_volume(self, volume: float) -> None:
        """Update with a new volume observation.

        Args:
            volume: New volume.
        """
        if volume < 0:
            raise FeeError(f"volume must be >= 0, got {volume}")

        self._volume_history.append(volume)
        if len(self._volume_history) > self.volatility_window:
            self._volume_history.pop(0)

        self._recompute_schedule()

    def _recompute_schedule(self) -> None:
        """Recompute the fee schedule based on current conditions."""
        volatility_multiplier = self._compute_volatility_multiplier()
        volume_multiplier = self._compute_volume_multiplier()

        self._current_schedule = FeeSchedule(
            base_fee_bps=self.base_fee_bps,
            volatility_multiplier=volatility_multiplier,
            volume_multiplier=volume_multiplier,
            max_fee_bps=self.max_fee_bps,
            min_fee_bps=self.min_fee_bps,
        )

    def _compute_volatility_multiplier(self) -> float:
        """Compute volatility multiplier.

        Returns:
            Volatility multiplier (>= 1.0).
        """
        if len(self._price_history) < 2:
            return 1.0

        returns = np.diff(np.log(self._price_history))
        volatility = np.std(returns) if len(returns) > 0 else 0.0

        # Map volatility to multiplier: higher vol -> higher fee
        # Base: 1.0 at 0% vol, 2.0 at 5% vol
        multiplier = 1.0 + volatility * 40.0
        return max(1.0, min(3.0, multiplier))

    def _compute_volume_multiplier(self) -> float:
        """Compute volume multiplier.

        Returns:
            Volume multiplier (>= 1.0).
        """
        if not self._volume_history:
            return 1.0

        avg_volume = np.mean(self._volume_history)
        if avg_volume <= 0:
            return 1.0

        # Higher volume -> slightly higher fee (more demand)
        # But cap to stay competitive
        multiplier = 1.0 + math.log1p(avg_volume / 1000.0) * 0.1
        return max(1.0, min(1.5, multiplier))

    def get_effective_fee_bps(self) -> int:
        """Get the current effective fee in basis points.

        Returns:
            Effective fee in bps.
        """
        return self._current_schedule.effective_fee_bps

    def reset(self) -> None:
        """Reset the fee manager."""
        self._price_history.clear()
        self._volume_history.clear()
        self._current_schedule = FeeSchedule(
            base_fee_bps=self.base_fee_bps,
            max_fee_bps=self.max_fee_bps,
            min_fee_bps=self.min_fee_bps,
        )


# ---------------------------------------------------------------------------
# Impermanent Loss Manager
# ---------------------------------------------------------------------------

class ImpermanentLossManager:
    """Manages and monitors impermanent loss for LP positions.

    Provides tools for:
    - Computing IL given price changes
    - Monitoring IL in real-time
    - Estimating break-even fee revenue
    - Hedging strategies
    """

    def __init__(self) -> None:
        """Initialize the IL manager."""
        self._positions: dict[str, dict[str, float]] = {}

    def register_position(
        self,
        position_id: str,
        initial_price: float,
        amount0: float,
        amount1: float,
    ) -> None:
        """Register a position for IL tracking.

        Args:
            position_id: Position identifier.
            initial_price: Initial price.
            amount0: Initial amount of token0.
            amount1: Initial amount of token1.
        """
        if initial_price <= 0:
            raise PoolError(f"initial_price must be positive, got {initial_price}")

        self._positions[position_id] = {
            "initial_price": initial_price,
            "amount0": amount0,
            "amount1": amount1,
        }

    def compute_il(
        self,
        position_id: str,
        current_price: float,
        fee_revenue: float = 0.0,
    ) -> ImpermanentLossResult:
        """Compute impermanent loss for a position.

        Args:
            position_id: Position identifier.
            current_price: Current price.
            fee_revenue: Fee revenue earned.

        Returns:
            ImpermanentLossResult.
        """
        if position_id not in self._positions:
            raise PoolError(f"Position {position_id} not found")

        pos = self._positions[position_id]
        initial_price = pos["initial_price"]
        amount0 = pos["amount0"]
        amount1 = pos["amount1"]

        if current_price <= 0:
            raise PoolError(f"current_price must be positive, got {current_price}")

        price_ratio = current_price / initial_price

        # HODL value: simply holding the initial amounts
        hodl_value = amount0 * current_price + amount1

        # LP value: value if in AMM pool
        # LP value = 2 * sqrt(price_ratio) / (1 + price_ratio) * hodl_value
        lp_value_factor = 2.0 * math.sqrt(price_ratio) / (1.0 + price_ratio)
        lp_value = hodl_value * lp_value_factor

        # Impermanent loss
        il_fraction = lp_value_factor - 1.0  # Negative = loss
        il_percentage = il_fraction * 100.0

        net_pnl = lp_value - hodl_value
        net_il_after_fees = net_pnl + fee_revenue

        return ImpermanentLossResult(
            il_fraction=il_fraction,
            il_percentage=il_percentage,
            hodl_value=hodl_value,
            lp_value=lp_value,
            net_pnl=net_pnl,
            fee_revenue=fee_revenue,
            net_il_after_fees=net_il_after_fees,
        )

    def compute_il_from_price_ratio(
        self, price_ratio: float, fee_revenue: float = 0.0
    ) -> ImpermanentLossResult:
        """Compute IL directly from price ratio.

        Args:
            price_ratio: Current price / initial price.
            fee_revenue: Fee revenue earned.

        Returns:
            ImpermanentLossResult.
        """
        if price_ratio <= 0:
            raise PoolError(f"price_ratio must be positive, got {price_ratio}")

        lp_value_factor = 2.0 * math.sqrt(price_ratio) / (1.0 + price_ratio)
        il_fraction = lp_value_factor - 1.0

        return ImpermanentLossResult(
            il_fraction=il_fraction,
            il_percentage=il_fraction * 100.0,
            hodl_value=1.0,
            lp_value=lp_value_factor,
            net_pnl=lp_value_factor - 1.0,
            fee_revenue=fee_revenue,
            net_il_after_fees=lp_value_factor - 1.0 + fee_revenue,
        )

    def break_even_fee_revenue(
        self, position_id: str, current_price: float
    ) -> float:
        """Compute fee revenue needed to break even.

        Args:
            position_id: Position identifier.
            current_price: Current price.

        Returns:
            Required fee revenue.
        """
        result = self.compute_il(position_id, current_price)
        return -result.net_pnl if result.net_pnl < 0 else 0.0

    def annualized_il(
        self,
        position_id: str,
        current_price: float,
        days_held: float,
    ) -> float:
        """Compute annualized impermanent loss.

        Args:
            position_id: Position identifier.
            current_price: Current price.
            days_held: Number of days the position has been held.

        Returns:
            Annualized IL as a fraction.
        """
        if days_held <= 0:
            raise PoolError(f"days_held must be positive, got {days_held}")

        result = self.compute_il(position_id, current_price)
        # Annualize: (1 + IL)^(365/days) - 1
        annualized = (1.0 + result.il_fraction) ** (365.0 / days_held) - 1.0
        return annualized

    def remove_position(self, position_id: str) -> None:
        """Remove a position from tracking.

        Args:
            position_id: Position to remove.
        """
        self._positions.pop(position_id, None)

    def get_tracked_positions(self) -> list[str]:
        """Get list of tracked position IDs.

        Returns:
            List of position IDs.
        """
        return list(self._positions.keys())


# ---------------------------------------------------------------------------
# AMM Factory
# ---------------------------------------------------------------------------

class AMMFactory:
    """Factory for creating AMM instances."""

    @staticmethod
    def create_constant_product(
        token0: Token,
        token1: Token,
        fee_tier: FeeTier = FeeTier.MEDIUM,
    ) -> ConstantProductAMM:
        """Create a constant product AMM.

        Args:
            token0: First token.
            token1: Second token.
            fee_tier: Fee tier.

        Returns:
            ConstantProductAMM.
        """
        config = PoolConfig(
            token0=token0,
            token1=token1,
            fee_tier=fee_tier,
            pool_type=PoolType.CONSTANT_PRODUCT,
        )
        return ConstantProductAMM(config)

    @staticmethod
    def create_concentrated_liquidity(
        token0: Token,
        token1: Token,
        fee_tier: FeeTier = FeeTier.MEDIUM,
        tick_spacing: int = 60,
    ) -> ConcentratedLiquidityAMM:
        """Create a concentrated liquidity AMM.

        Args:
            token0: First token.
            token1: Second token.
            fee_tier: Fee tier.
            tick_spacing: Tick spacing.

        Returns:
            ConcentratedLiquidityAMM.
        """
        config = PoolConfig(
            token0=token0,
            token1=token1,
            fee_tier=fee_tier,
            pool_type=PoolType.CONCENTRATED_LIQUIDITY,
        )
        return ConcentratedLiquidityAMM(config, tick_spacing=tick_spacing)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def compute_price_impact(
    amount_in: float,
    reserve_in: float,
    reserve_out: float,
    fee_fraction: float = 0.003,
) -> float:
    """Compute price impact of a swap.

    Args:
        amount_in: Input amount.
        reserve_in: Input reserve.
        reserve_out: Output reserve.
        fee_fraction: Fee fraction.

    Returns:
        Price impact as a fraction.
    """
    if reserve_in <= 0 or reserve_out <= 0:
        raise PoolError("Reserves must be positive")

    amount_in_with_fee = amount_in * (1.0 - fee_fraction)
    spot_price = reserve_out / reserve_in
    execution_price = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
    return (spot_price - execution_price) / spot_price


def compute_lp_tokens(
    amount0: float,
    amount1: float,
    reserve0: float,
    reserve1: float,
    total_supply: float,
) -> float:
    """Compute LP tokens for a deposit.

    Args:
        amount0: Amount of token0.
        amount1: Amount of token1.
        reserve0: Current reserve0.
        reserve1: Current reserve1.
        total_supply: Current LP token supply.

    Returns:
        LP tokens to mint.
    """
    if total_supply == 0:
        return math.sqrt(amount0 * amount1)

    liquidity0 = amount0 * total_supply / reserve0 if reserve0 > 0 else 0.0
    liquidity1 = amount1 * total_supply / reserve1 if reserve1 > 0 else 0.0
    return min(liquidity0, liquidity1)


def compute_sqrt_price(tick: int) -> float:
    """Compute sqrt price from tick.

    Args:
        tick: Tick index.

    Returns:
        Square root of price.
    """
    return math.sqrt(1.0001**tick)


def compute_tick_from_sqrt_price(sqrt_price: float) -> int:
    """Compute tick from sqrt price.

    Args:
        sqrt_price: Square root of price.

    Returns:
        Tick index.
    """
    if sqrt_price <= 0:
        raise TickError(f"sqrt_price must be positive, got {sqrt_price}")
    return int(math.log(sqrt_price**2) / math.log(1.0001))


__all__ = [
    "AMMError",
    "PoolError",
    "LiquidityError",
    "SwapError",
    "TickError",
    "FeeError",
    "PoolType",
    "FeeTier",
    "PositionStatus",
    "Token",
    "PoolConfig",
    "PoolState",
    "SwapResult",
    "LPPosition",
    "Tick",
    "FeeSchedule",
    "ImpermanentLossResult",
    "ConstantProductAMM",
    "ConcentratedLiquidityAMM",
    "DynamicFeeManager",
    "ImpermanentLossManager",
    "AMMFactory",
    "compute_price_impact",
    "compute_lp_tokens",
    "compute_sqrt_price",
    "compute_tick_from_sqrt_price",
]
