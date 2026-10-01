"""
Apex FinTech Markets — HFT Infrastructure.

High-frequency trading infrastructure components:

1. **Feed Handler** — Market data feed integration with normalization,
   validation, and dispatch to subscribers.
2. **Matching Engine** — Price-time priority matching with order
   modification, cancellation, and trade execution.
3. **Order Book** — Full limit order book with price-level aggregation,
   best bid/ask tracking, and market depth management.

All components are designed for low-latency operation with lock-free
data structures where possible and comprehensive error handling.
"""

from __future__ import annotations

import heapq
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class HFTError(Exception):
    """Base exception for HFT infrastructure errors."""


class FeedError(HFTError):
    """Raised when feed handler encounters an error."""


class MatchingEngineError(HFTError):
    """Raised when matching engine encounters an error."""


class OrderBookError(HFTError):
    """Raised when order book encounters an error."""


class OrderValidationError(HFTError):
    """Raised when an order fails validation."""


class FeedDisconnectedError(FeedError):
    """Raised when a feed disconnects unexpectedly."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    LIMIT = "limit"
    MARKET = "market"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    POST_ONLY = "post_only"


class OrderStatus(Enum):
    """Order status."""

    PENDING = auto()
    OPEN = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class TimeInForce(Enum):
    """Time in force."""

    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date


class FeedEventType(Enum):
    """Feed event type."""

    TRADE = "trade"
    QUOTE = "quote"
    BOOK_UPDATE = "book_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderID:
    """Unique order identifier."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise OrderValidationError("OrderID value cannot be empty")

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass
class Order:
    """A limit order.

    Attributes:
        order_id: Unique order identifier.
        side: Buy or sell.
        price: Limit price.
        quantity: Order quantity.
        filled_quantity: Quantity already filled.
        order_type: Type of order.
        time_in_force: Time in force.
        timestamp: Order creation timestamp.
        client_id: Client identifier.
        status: Current order status.
    """

    order_id: OrderID
    side: Side
    price: float
    quantity: float
    filled_quantity: float = 0.0
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC
    timestamp: float = 0.0
    client_id: str = ""
    status: OrderStatus = OrderStatus.PENDING

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise OrderValidationError(f"price must be positive, got {self.price}")
        if self.quantity <= 0:
            raise OrderValidationError(
                f"quantity must be positive, got {self.quantity}"
            )
        if self.filled_quantity < 0:
            raise OrderValidationError(
                f"filled_quantity must be >= 0, got {self.filled_quantity}"
            )
        if self.filled_quantity > self.quantity:
            raise OrderValidationError(
                f"filled_quantity ({self.filled_quantity}) exceeds quantity ({self.quantity})"
            )
        if self.timestamp == 0.0:
            self.timestamp = time.time_ns()

    @property
    def remaining_quantity(self) -> float:
        """Remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    @property
    def is_filled(self) -> bool:
        """True if fully filled."""
        return self.filled_quantity >= self.quantity

    @property
    def is_buy(self) -> bool:
        """True if buy order."""
        return self.side == Side.BUY

    @property
    def is_sell(self) -> bool:
        """True if sell order."""
        return self.side == Side.SELL

    def fill(self, qty: float) -> None:
        """Record a fill.

        Args:
            qty: Quantity filled.
        """
        if qty <= 0:
            raise OrderValidationError(f"fill qty must be positive, got {qty}")
        if self.filled_quantity + qty > self.quantity:
            raise OrderValidationError(
                f"fill would exceed quantity: {self.filled_quantity} + {qty} > {self.quantity}"
            )
        self.filled_quantity += qty
        if self.is_filled:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass(frozen=True)
class Trade:
    """A executed trade.

    Attributes:
        trade_id: Unique trade identifier.
        buy_order_id: Buy order identifier.
        sell_order_id: Sell order identifier.
        price: Execution price.
        quantity: Executed quantity.
        timestamp: Execution timestamp.
        taker_side: Side of the taker (aggressor).
    """

    trade_id: str
    buy_order_id: OrderID
    sell_order_id: OrderID
    price: float
    quantity: float
    timestamp: float
    taker_side: Side

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise MatchingEngineError(f"trade price must be positive, got {self.price}")
        if self.quantity <= 0:
            raise MatchingEngineError(
                f"trade quantity must be positive, got {self.quantity}"
            )


@dataclass
class PriceLevel:
    """A price level in the order book.

    Attributes:
        price: Price of this level.
        total_quantity: Total quantity at this level.
        orders: List of orders at this level.
    """

    price: float
    total_quantity: float
    orders: list[Order] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise OrderBookError(f"price must be positive, got {self.price}")
        if self.total_quantity < 0:
            raise OrderBookError(
                f"total_quantity must be >= 0, got {self.total_quantity}"
            )


@dataclass(frozen=True)
class BookSnapshot:
    """Snapshot of the order book.

    Attributes:
        bids: List of bid price levels (sorted descending).
        asks: List of ask price levels (sorted ascending).
        timestamp: Snapshot timestamp.
    """

    bids: list[PriceLevel]
    asks: list[PriceLevel]
    timestamp: float

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


@dataclass(frozen=True)
class FeedEvent:
    """A market data event.

    Attributes:
        event_type: Type of event.
        symbol: Instrument symbol.
        data: Event payload.
        timestamp: Event timestamp.
        sequence: Sequence number for ordering.
    """

    event_type: FeedEventType
    symbol: str
    data: dict[str, Any]
    timestamp: float
    sequence: int = 0


@dataclass(frozen=True)
class MarketDataTick:
    """A market data tick.

    Attributes:
        symbol: Instrument symbol.
        bid: Best bid price.
        ask: Best ask price.
        bid_size: Best bid size.
        ask_size: Best ask size.
        last_price: Last trade price.
        last_size: Last trade size.
        volume: Cumulative volume.
        timestamp: Tick timestamp.
    """

    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    last_size: float
    volume: float
    timestamp: float

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise FeedError(f"bid/ask must be positive, got bid={self.bid}, ask={self.ask}")
        if self.bid > self.ask:
            raise FeedError(f"Crossed market: bid={self.bid} > ask={self.ask}")

    @property
    def mid_price(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid


# ---------------------------------------------------------------------------
# Feed Handler
# ---------------------------------------------------------------------------

T = TypeVar("T")


class FeedHandler(ABC, Generic[T]):
    """Abstract base class for market data feed handlers.

    Provides a normalized interface for consuming market data from
    various sources (WebSocket, FIX, REST polling, etc.).
    """

    def __init__(self, symbols: list[str]) -> None:
        """Initialize the feed handler.

        Args:
            symbols: List of instrument symbols to subscribe to.
        """
        if not symbols:
            raise FeedError("At least one symbol must be provided")
        self.symbols = symbols
        self._subscribers: list[Callable[[FeedEvent], None]] = []
        self._running = False
        self._sequence = 0
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """True if the feed is running."""
        return self._running

    def subscribe(self, callback: Callable[[FeedEvent], None]) -> None:
        """Subscribe to feed events.

        Args:
            callback: Callback function for feed events.
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[FeedEvent], None]) -> None:
        """Unsubscribe from feed events.

        Args:
            callback: Callback to remove.
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _dispatch(self, event: FeedEvent) -> None:
        """Dispatch an event to all subscribers.

        Args:
            event: The feed event to dispatch.
        """
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                # Log but don't crash the feed
                print(f"Feed subscriber error: {e}")

    def _next_sequence(self) -> int:
        """Get the next sequence number.

        Returns:
            Next sequence number.
        """
        with self._lock:
            self._sequence += 1
            return self._sequence

    @abstractmethod
    def connect(self) -> None:
        """Connect to the feed."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the feed."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Start processing feed events."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop processing feed events."""
        ...


class SimulatedFeedHandler(FeedHandler[MarketDataTick]):
    """Simulated feed handler for testing and backtesting.

    Generates synthetic market data with configurable parameters.
    """

    def __init__(
        self,
        symbols: list[str],
        base_price: float = 100.0,
        volatility: float = 0.001,
        tick_interval: float = 0.1,
    ) -> None:
        """Initialize the simulated feed handler.

        Args:
            symbols: List of symbols.
            base_price: Base price for simulation.
            volatility: Per-tick volatility.
            tick_interval: Seconds between ticks.
        """
        super().__init__(symbols)
        self.base_price = base_price
        self.volatility = volatility
        self.tick_interval = tick_interval
        self._prices: dict[str, float] = {s: base_price for s in symbols}
        self._thread: Optional[threading.Thread] = None

    def connect(self) -> None:
        """No-op for simulated feed."""
        pass

    def disconnect(self) -> None:
        """No-op for simulated feed."""
        pass

    def start(self) -> None:
        """Start generating simulated data."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop generating simulated data."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        """Main simulation loop."""
        while self._running:
            for symbol in self.symbols:
                self._generate_tick(symbol)
            time.sleep(self.tick_interval)

    def _generate_tick(self, symbol: str) -> None:
        """Generate a single tick.

        Args:
            symbol: Symbol to generate tick for.
        """
        current = self._prices[symbol]
        change = np.random.normal(0, self.volatility * current)
        new_price = max(current + change, 0.01)
        self._prices[symbol] = new_price

        spread = new_price * 0.0001
        bid = new_price - spread / 2
        ask = new_price + spread / 2

        tick = MarketDataTick(
            symbol=symbol,
            bid=round(bid, 4),
            ask=round(ask, 4),
            bid_size=round(np.random.exponential(100), 2),
            ask_size=round(np.random.exponential(100), 2),
            last_price=round(new_price, 4),
            last_size=round(np.random.exponential(10), 2),
            volume=round(np.random.exponential(1000), 2),
            timestamp=time.time(),
        )

        event = FeedEvent(
            event_type=FeedEventType.QUOTE,
            symbol=symbol,
            data={"tick": tick},
            timestamp=tick.timestamp,
            sequence=self._next_sequence(),
        )
        self._dispatch(event)


# ---------------------------------------------------------------------------
# Order Book
# ---------------------------------------------------------------------------

class OrderBook:
    """Full limit order book with price-time priority.

    Maintains separate bid and ask sides with O(1) best quote access
    and O(log n) order insertion/removal.
    """

    def __init__(self, symbol: str, max_depth: int = 100) -> None:
        """Initialize the order book.

        Args:
            symbol: Instrument symbol.
            max_depth: Maximum price levels to track.
        """
        self.symbol = symbol
        self.max_depth = max_depth

        # Price levels: price -> PriceLevel
        self._bid_levels: dict[float, PriceLevel] = {}
        self._ask_levels: dict[float, PriceLevel] = {}

        # Order lookup: order_id -> (side, price, order)
        self._orders: dict[OrderID, tuple[Side, float, Order]] = {}

        # Sorted price lists for best quote access
        self._bid_prices: list[float] = []  # Max-heap (stored as negatives)
        self._ask_prices: list[float] = []  # Min-heap

        self._lock = threading.RLock()
        self._last_trade_price: Optional[float] = None
        self._volume = 0.0

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        with self._lock:
            while self._bid_prices:
                price = -self._bid_prices[0]
                if price in self._bid_levels and self._bid_levels[price].total_quantity > 0:
                    return price
                heapq.heappop(self._bid_prices)
            return None

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        with self._lock:
            while self._ask_prices:
                price = self._ask_prices[0]
                if price in self._ask_levels and self._ask_levels[price].total_quantity > 0:
                    return price
                heapq.heappop(self._ask_prices)
            return None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price."""
        with self._lock:
            bb = self.best_bid
            ba = self.best_ask
            if bb is not None and ba is not None:
                return (bb + ba) / 2.0
            return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        with self._lock:
            bb = self.best_bid
            ba = self.best_ask
            if bb is not None and ba is not None:
                return ba - bb
            return None

    @property
    def last_trade_price(self) -> Optional[float]:
        """Last trade price."""
        return self._last_trade_price

    @property
    def volume(self) -> float:
        """Total volume."""
        return self._volume

    def add_order(self, order: Order) -> None:
        """Add an order to the book.

        Args:
            order: The order to add.
        """
        with self._lock:
            if order.order_id in self._orders:
                raise OrderBookError(f"Duplicate order ID: {order.order_id}")

            levels = self._bid_levels if order.is_buy else self._ask_levels
            prices = self._bid_prices if order.is_buy else self._ask_prices

            if order.price not in levels:
                levels[order.price] = PriceLevel(price=order.price, total_quantity=0.0)
                if order.is_buy:
                    heapq.heappush(prices, -order.price)
                else:
                    heapq.heappush(prices, order.price)

            level = levels[order.price]
            level.orders.append(order)
            level.total_quantity += order.remaining_quantity
            self._orders[order.order_id] = (order.side, order.price, order)

    def cancel_order(self, order_id: OrderID) -> Optional[Order]:
        """Cancel an order.

        Args:
            order_id: Order to cancel.

        Returns:
            The cancelled order, or None if not found.
        """
        with self._lock:
            if order_id not in self._orders:
                return None

            side, price, order = self._orders.pop(order_id)
            order.status = OrderStatus.CANCELLED

            levels = self._bid_levels if side == Side.BUY else self._ask_levels
            if price in levels:
                level = levels[price]
                level.orders = [o for o in level.orders if o.order_id != order_id]
                level.total_quantity -= order.remaining_quantity
                if level.total_quantity <= 0:
                    del levels[price]

            return order

    def modify_order(
        self, order_id: OrderID, new_price: Optional[float] = None, new_quantity: Optional[float] = None
    ) -> Optional[Order]:
        """Modify an order.

        Args:
            order_id: Order to modify.
            new_price: New price (None to keep current).
            new_quantity: New quantity (None to keep current).

        Returns:
            The modified order, or None if not found.
        """
        with self._lock:
            if order_id not in self._orders:
                return None

            side, price, order = self._orders[order_id]

            # Cancel old order
            self.cancel_order(order_id)

            # Create new order with modified parameters
            updated_price = new_price if new_price is not None else price
            updated_qty = new_quantity if new_quantity is not None else order.quantity

            new_order = Order(
                order_id=order_id,
                side=side,
                price=updated_price,
                quantity=updated_qty,
                filled_quantity=order.filled_quantity,
                order_type=order.order_type,
                time_in_force=order.time_in_force,
                client_id=order.client_id,
            )

            self.add_order(new_order)
            return new_order

    def get_order(self, order_id: OrderID) -> Optional[Order]:
        """Get an order by ID.

        Args:
            order_id: Order to look up.

        Returns:
            The order, or None if not found.
        """
        with self._lock:
            entry = self._orders.get(order_id)
            return entry[2] if entry else None

    def get_depth(self, levels: int = 10) -> BookSnapshot:
        """Get a snapshot of the order book.

        Args:
            levels: Number of price levels to include.

        Returns:
            BookSnapshot.
        """
        with self._lock:
            bid_prices = sorted(self._bid_levels.keys(), reverse=True)[:levels]
            ask_prices = sorted(self._ask_levels.keys())[:levels]

            bids = [
                self._bid_levels[p] for p in bid_prices if self._bid_levels[p].total_quantity > 0
            ]
            asks = [
                self._ask_levels[p] for p in ask_prices if self._ask_levels[p].total_quantity > 0
            ]

            return BookSnapshot(
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )

    def record_trade(self, price: float, quantity: float) -> None:
        """Record a trade.

        Args:
            price: Trade price.
            quantity: Trade quantity.
        """
        with self._lock:
            self._last_trade_price = price
            self._volume += quantity

    def clear(self) -> None:
        """Clear the entire order book."""
        with self._lock:
            self._bid_levels.clear()
            self._ask_levels.clear()
            self._orders.clear()
            self._bid_prices.clear()
            self._ask_prices.clear()
            self._last_trade_price = None
            self._volume = 0.0


# ---------------------------------------------------------------------------
# Matching Engine
# ---------------------------------------------------------------------------

class MatchingEngine:
    """Price-time priority matching engine.

    Matches incoming orders against the resting book, generating trades
    and updating order statuses.
    """

    def __init__(self, symbol: str) -> None:
        """Initialize the matching engine.

        Args:
            symbol: Instrument symbol.
        """
        self.symbol = symbol
        self.order_book = OrderBook(symbol)
        self._trade_listeners: list[Callable[[Trade], None]] = []
        self._order_listeners: list[Callable[[Order], None]] = []
        self._trade_counter = 0
        self._lock = threading.Lock()

    def add_trade_listener(self, callback: Callable[[Trade], None]) -> None:
        """Add a trade listener.

        Args:
            callback: Callback for trade events.
        """
        self._trade_listeners.append(callback)

    def add_order_listener(self, callback: Callable[[Order], None]) -> None:
        """Add an order listener.

        Args:
            callback: Callback for order events.
        """
        self._order_listeners.append(callback)

    def submit_order(self, order: Order) -> list[Trade]:
        """Submit an order to the matching engine.

        Args:
            order: The order to submit.

        Returns:
            List of generated trades.
        """
        with self._lock:
            if order.order_type == OrderType.MARKET:
                return self._match_market_order(order)
            elif order.order_type == OrderType.IOC:
                return self._match_ioc_order(order)
            elif order.order_type == OrderType.FOK:
                return self._match_fok_order(order)
            elif order.order_type == OrderType.POST_ONLY:
                return self._match_post_only_order(order)
            else:
                return self._match_limit_order(order)

    def cancel_order(self, order_id: OrderID) -> Optional[Order]:
        """Cancel an order.

        Args:
            order_id: Order to cancel.

        Returns:
            The cancelled order, or None if not found.
        """
        order = self.order_book.cancel_order(order_id)
        if order:
            self._notify_order_listeners(order)
        return order

    def _match_limit_order(self, order: Order) -> list[Trade]:
        """Match a limit order against the book.

        Args:
            order: The limit order.

        Returns:
            List of generated trades.
        """
        trades: list[Trade] = []
        remaining = order.remaining_quantity

        if order.is_buy:
            while remaining > 0:
                best_ask = self.order_book.best_ask
                if best_ask is None or best_ask > order.price:
                    break
                trades.extend(self._execute_against_book(order, Side.SELL, remaining))
                remaining = order.remaining_quantity
        else:
            while remaining > 0:
                best_bid = self.order_book.best_bid
                if best_bid is None or best_bid < order.price:
                    break
                trades.extend(self._execute_against_book(order, Side.BUY, remaining))
                remaining = order.remaining_quantity

        # Add remaining to book
        if order.remaining_quantity > 0 and order.time_in_force == TimeInForce.GTC:
            self.order_book.add_order(order)
            order.status = OrderStatus.OPEN
            self._notify_order_listeners(order)

        return trades

    def _match_market_order(self, order: Order) -> list[Trade]:
        """Match a market order.

        Args:
            order: The market order.

        Returns:
            List of generated trades.
        """
        trades: list[Trade] = []
        remaining = order.remaining_quantity

        if order.is_buy:
            while remaining > 0:
                best_ask = self.order_book.best_ask
                if best_ask is None:
                    break
                trades.extend(self._execute_against_book(order, Side.SELL, remaining))
                remaining = order.remaining_quantity
        else:
            while remaining > 0:
                best_bid = self.order_book.best_bid
                if best_bid is None:
                    break
                trades.extend(self._execute_against_book(order, Side.BUY, remaining))
                remaining = order.remaining_quantity

        if order.remaining_quantity > 0:
            order.status = OrderStatus.EXPIRED
            self._notify_order_listeners(order)

        return trades

    def _match_ioc_order(self, order: Order) -> list[Trade]:
        """Match an Immediate-or-Cancel order.

        Args:
            order: The IOC order.

        Returns:
            List of generated trades.
        """
        trades = self._match_market_order(order)
        if order.remaining_quantity > 0:
            order.status = OrderStatus.CANCELLED
            self._notify_order_listeners(order)
        return trades

    def _match_fok_order(self, order: Order) -> list[Trade]:
        """Match a Fill-or-Kill order.

        Args:
            order: The FOK order.

        Returns:
            List of generated trades (empty if not fully fillable).
        """
        # Check if fully fillable
        if not self._can_fill(order):
            order.status = OrderStatus.REJECTED
            self._notify_order_listeners(order)
            return []

        return self._match_market_order(order)

    def _match_post_only_order(self, order: Order) -> list[Trade]:
        """Match a Post-Only order.

        Args:
            order: The post-only order.

        Returns:
            Empty list (post-only orders never take liquidity).
        """
        # Check if would cross
        if self._would_cross(order):
            order.status = OrderStatus.REJECTED
            self._notify_order_listeners(order)
            return []

        self.order_book.add_order(order)
        order.status = OrderStatus.OPEN
        self._notify_order_listeners(order)
        return []

    def _can_fill(self, order: Order) -> bool:
        """Check if an order can be fully filled.

        Args:
            order: The order to check.

        Returns:
            True if fully fillable.
        """
        remaining = order.remaining_quantity
        if order.is_buy:
            for level in sorted(self.order_book._ask_levels.values(), key=lambda x: x.price):
                if level.price > order.price:
                    break
                remaining -= level.total_quantity
                if remaining <= 0:
                    return True
        else:
            for level in sorted(self.order_book._bid_levels.values(), key=lambda x: -x.price):
                if level.price < order.price:
                    break
                remaining -= level.total_quantity
                if remaining <= 0:
                    return True
        return False

    def _would_cross(self, order: Order) -> bool:
        """Check if an order would cross the spread.

        Args:
            order: The order to check.

        Returns:
            True if the order would cross.
        """
        if order.is_bid:
            best_ask = self.order_book.best_ask
            return best_ask is not None and order.price >= best_ask
        else:
            best_bid = self.order_book.best_bid
            return best_bid is not None and order.price <= best_bid

    def _execute_against_book(
        self, taker: Order, maker_side: Side, max_qty: float
    ) -> list[Trade]:
        """Execute a taker order against resting book orders.

        Args:
            taker: The taker order.
            maker_side: Side of the resting orders.
            max_qty: Maximum quantity to execute.

        Returns:
            List of generated trades.
        """
        trades: list[Trade] = []
        remaining = max_qty

        levels = self.order_book._ask_levels if maker_side == Side.SELL else self.order_book._bid_levels

        while remaining > 0:
            if maker_side == Side.SELL:
                best_price = self.order_book.best_ask
            else:
                best_price = self.order_book.best_bid

            if best_price is None:
                break

            level = levels.get(best_price)
            if not level or level.total_quantity <= 0:
                break

            for maker_order in list(level.orders):
                if remaining <= 0:
                    break

                fill_qty = min(remaining, maker_order.remaining_quantity)
                if fill_qty <= 0:
                    continue

                # Execute
                self._trade_counter += 1
                trade = Trade(
                    trade_id=f"T{self._trade_counter}",
                    buy_order_id=taker.order_id if taker.is_buy else maker_order.order_id,
                    sell_order_id=taker.order_id if taker.is_sell else maker_order.order_id,
                    price=best_price,
                    quantity=fill_qty,
                    timestamp=time.time_ns(),
                    taker_side=taker.side,
                )
                trades.append(trade)

                # Update orders
                taker.fill(fill_qty)
                maker_order.fill(fill_qty)
                level.total_quantity -= fill_qty
                remaining -= fill_qty

                self.order_book.record_trade(best_price, fill_qty)
                self._notify_trade_listeners(trade)
                self._notify_order_listeners(maker_order)

                if maker_order.is_filled:
                    level.orders.remove(maker_order)
                    if maker_order.order_id in self.order_book._orders:
                        del self.order_book._orders[maker_order.order_id]

            if level.total_quantity <= 0:
                del levels[best_price]

        return trades

    def _notify_trade_listeners(self, trade: Trade) -> None:
        """Notify trade listeners.

        Args:
            trade: The trade to notify.
        """
        for listener in self._trade_listeners:
            try:
                listener(trade)
            except Exception as e:
                print(f"Trade listener error: {e}")

    def _notify_order_listeners(self, order: Order) -> None:
        """Notify order listeners.

        Args:
            order: The order to notify.
        """
        for listener in self._order_listeners:
            try:
                listener(order)
            except Exception as e:
                print(f"Order listener error: {e}")

    def get_order_book(self) -> OrderBook:
        """Get the underlying order book.

        Returns:
            The order book.
        """
        return self.order_book


# ---------------------------------------------------------------------------
# Feed-to-Engine Bridge
# ---------------------------------------------------------------------------

class FeedToEngineBridge:
    """Bridges market data feeds to the matching engine.

    Normalizes feed events and updates the matching engine's order book.
    """

    def __init__(self, engine: MatchingEngine) -> None:
        """Initialize the bridge.

        Args:
            engine: The matching engine to bridge to.
        """
        self.engine = engine
        self._feed_handler: Optional[FeedHandler[Any]] = None

    def connect_feed(self, feed_handler: FeedHandler[Any]) -> None:
        """Connect a feed handler.

        Args:
            feed_handler: The feed handler to connect.
        """
        self._feed_handler = feed_handler
        feed_handler.subscribe(self._on_feed_event)

    def disconnect_feed(self) -> None:
        """Disconnect the feed handler."""
        if self._feed_handler:
            self._feed_handler.unsubscribe(self._on_feed_event)
            self._feed_handler = None

    def _on_feed_event(self, event: FeedEvent) -> None:
        """Handle a feed event.

        Args:
            event: The feed event.
        """
        if event.event_type == FeedEventType.QUOTE:
            self._update_book_from_quote(event)
        elif event.event_type == FeedEventType.TRADE:
            self._update_book_from_trade(event)

    def _update_book_from_quote(self, event: FeedEvent) -> None:
        """Update the book from a quote event.

        Args:
            event: The quote event.
        """
        tick = event.data.get("tick")
        if tick is None:
            return

        # Update best bid/ask in the book
        book = self.engine.get_order_book()
        # This is a simplified update — real implementations would
        # maintain a separate top-of-book cache
        pass

    def _update_book_from_trade(self, event: FeedEvent) -> None:
        """Update the book from a trade event.

        Args:
            event: The trade event.
        """
        price = event.data.get("price")
        size = event.data.get("size")
        if price and size:
            self.engine.get_order_book().record_trade(price, size)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def create_matching_engine(symbol: str) -> MatchingEngine:
    """Create a matching engine for a symbol.

    Args:
        symbol: Instrument symbol.

    Returns:
        Configured MatchingEngine.
    """
    return MatchingEngine(symbol)


def create_simulated_feed(
    symbols: list[str], base_price: float = 100.0
) -> SimulatedFeedHandler:
    """Create a simulated feed handler.

    Args:
        symbols: List of symbols.
        base_price: Base price for simulation.

    Returns:
        Configured SimulatedFeedHandler.
    """
    return SimulatedFeedHandler(symbols, base_price=base_price)


def compute_book_imbalance(book: OrderBook, levels: int = 5) -> float:
    """Compute order book imbalance.

    Args:
        book: The order book.
        levels: Number of levels to consider.

    Returns:
        Imbalance ratio in [-1, 1]. Positive = more bids.
    """
    snapshot = book.get_depth(levels)
    bid_qty = sum(level.total_quantity for level in snapshot.bids)
    ask_qty = sum(level.total_quantity for level in snapshot.asks)

    total = bid_qty + ask_qty
    if total == 0:
        return 0.0
    return (bid_qty - ask_qty) / total


def compute_weighted_mid(book: OrderBook, levels: int = 3) -> Optional[float]:
    """Compute volume-weighted mid price.

    Args:
        book: The order book.
        levels: Number of levels to consider.

    Returns:
        Volume-weighted mid price, or None if book is empty.
    """
    snapshot = book.get_depth(levels)
    if not snapshot.bids or not snapshot.asks:
        return None

    bid_qty = sum(level.total_quantity for level in snapshot.bids)
    ask_qty = sum(level.total_quantity for level in snapshot.asks)

    if bid_qty == 0 or ask_qty == 0:
        return None

    bid_vwap = sum(level.price * level.total_quantity for level in snapshot.bids) / bid_qty
    ask_vwap = sum(level.price * level.total_quantity for level in snapshot.asks) / ask_qty

    return (bid_vwap + ask_vwap) / 2.0


__all__ = [
    "HFTError",
    "FeedError",
    "MatchingEngineError",
    "OrderBookError",
    "OrderValidationError",
    "FeedDisconnectedError",
    "Side",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "FeedEventType",
    "OrderID",
    "Order",
    "Trade",
    "PriceLevel",
    "BookSnapshot",
    "FeedEvent",
    "MarketDataTick",
    "FeedHandler",
    "SimulatedFeedHandler",
    "OrderBook",
    "MatchingEngine",
    "FeedToEngineBridge",
    "create_matching_engine",
    "create_simulated_feed",
    "compute_book_imbalance",
    "compute_weighted_mid",
]
