"""
Apex FinTech Markets — Cross-Chain Module.

Cross-chain bridges (Hop, Across), cross-chain arbitrage detection,
and atomic settlement via hash time-locked contracts (HTLCs).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum, auto
from typing import Any, Final


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_CHAINS: Final[frozenset[str]] = frozenset({
    "ethereum", "polygon", "arbitrum", "optimism", "base",
    "avalanche", "solana", "tron",
})

SUPPORTED_TOKENS: Final[frozenset[str]] = frozenset({
    "USDC", "USDT", "DAI", "WETH", "WBTC", "ETH",
})

MAX_BRIDGE_AMOUNT: Final[Decimal] = Decimal("10_000_000")
MIN_ARBITRAGE_PROFIT: Final[Decimal] = Decimal("0.001")  # 0.1% minimum
HTLC_LOCK_DURATION_SECONDS: Final[int] = 3600  # 1 hour
HTLC_REFUND_DELAY_SECONDS: Final[int] = 7200  # 2 hours


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CrossChainError(Exception):
    """Base exception for cross-chain operations."""


class BridgeError(CrossChainError):
    """Raised when a bridge operation fails."""


class InsufficientLiquidityError(BridgeError):
    """Raised when bridge liquidity is insufficient."""


class BridgeTimeoutError(BridgeError):
    """Raised when a bridge operation times out."""


class ArbitrageError(CrossChainError):
    """Raised when arbitrage detection or execution fails."""


class SettlementError(CrossChainError):
    """Raised when atomic settlement fails."""


class HTLCError(SettlementError):
    """Raised when an HTLC operation fails."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BridgeType(Enum):
    """Supported cross-chain bridge protocols."""

    HOP = auto()
    ACROSS = auto()


class BridgeStatus(Enum):
    """Status of a bridge transfer."""

    PENDING = auto()
    INITIATED = auto()
    LP_BONDED = auto()
    RELAYER_SUBMITTED = auto()
    COMPLETED = auto()
    FAILED = auto()
    REFUNDED = auto()


class ArbitrageStatus(Enum):
    """Status of an arbitrage opportunity."""

    DETECTED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    EXPIRED = auto()


class HTLCStatus(Enum):
    """Status of a hash time-locked contract."""

    INITIATED = auto()
    FUNDED = auto()
    CLAIMED = auto()
    REFUNDED = auto()
    EXPIRED = auto()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ChainConfig:
    """
    Configuration for a supported blockchain.

    Attributes:
        chain_id: Unique chain identifier.
        name: Human-readable chain name.
        rpc_url: JSON-RPC endpoint URL.
        native_currency: Native gas currency symbol.
        block_time_seconds: Average block confirmation time.
        confirmations_required: Number of confirmations for finality.
    """

    chain_id: int
    name: str
    rpc_url: str
    native_currency: str
    block_time_seconds: Decimal
    confirmations_required: int


@dataclass(frozen=True, slots=True)
class BridgeTransfer:
    """
    Immutable cross-chain bridge transfer request.

    Attributes:
        transfer_id: Unique transfer identifier.
        bridge: Bridge protocol to use.
        source_chain: Source blockchain name.
        destination_chain: Destination blockchain name.
        token: Token symbol to transfer.
        amount: Amount as Decimal.
        sender_address: Sender wallet address.
        receiver_address: Receiver wallet address.
        min_amount_out: Minimum acceptable output (slippage protection).
        deadline: Unix timestamp for transfer deadline.
    """

    transfer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    bridge: BridgeType = BridgeType.HOP
    source_chain: str = "ethereum"
    destination_chain: str = "arbitrum"
    token: str = "USDC"
    amount: Decimal = Decimal("0")
    sender_address: str = ""
    receiver_address: str = ""
    min_amount_out: Decimal = Decimal("0")
    deadline: int = 0

    def __post_init__(self) -> None:
        if self.amount <= Decimal("0"):
            raise BridgeError(f"Amount must be positive, got {self.amount}")
        if self.source_chain not in SUPPORTED_CHAINS:
            raise BridgeError(f"Unsupported source chain: {self.source_chain}")
        if self.destination_chain not in SUPPORTED_CHAINS:
            raise BridgeError(f"Unsupported destination chain: {self.destination_chain}")
        if self.token not in SUPPORTED_TOKENS:
            raise BridgeError(f"Unsupported token: {self.token}")
        if self.source_chain == self.destination_chain:
            raise BridgeError("Source and destination chains must differ")


@dataclass(frozen=True, slots=True)
class BridgeResult:
    """
    Immutable result of a bridge transfer.

    Attributes:
        transfer_id: Correlates to the originating transfer.
        status: Final bridge status.
        source_tx_hash: Source chain transaction hash.
        destination_tx_hash: Destination chain transaction hash.
        amount_out: Actual output amount received.
        fee: Bridge fee charged.
        relay_fee: Relayer fee (Across only).
        completed_at: ISO 8601 completion timestamp.
        error_message: Error details if failed.
    """

    transfer_id: str
    status: BridgeStatus
    source_tx_hash: str = ""
    destination_tx_hash: str = ""
    amount_out: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    relay_fee: Decimal = Decimal("0")
    completed_at: str = ""
    error_message: str = ""


@dataclass(frozen=True, slots=True)
class ArbitrageOpportunity:
    """
    Immutable cross-chain arbitrage opportunity.

    Attributes:
        opportunity_id: Unique identifier.
        source_chain: Chain to buy on.
        destination_chain: Chain to sell on.
        token: Token symbol.
        source_price: Price on source chain (USD).
        destination_price: Price on destination chain (USD).
        spread_pct: Profit spread percentage.
        estimated_profit: Estimated profit in USD.
        net_profit: Profit after bridge fees.
        expires_at: ISO 8601 expiration timestamp.
    """

    opportunity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_chain: str = ""
    destination_chain: str = ""
    token: str = ""
    source_price: Decimal = Decimal("0")
    destination_price: Decimal = Decimal("0")
    spread_pct: Decimal = Decimal("0")
    estimated_profit: Decimal = Decimal("0")
    net_profit: Decimal = Decimal("0")
    expires_at: str = ""


@dataclass(frozen=True, slots=True)
class HTLCContract:
    """
    Immutable hash time-locked contract for atomic settlement.

    Attributes:
        contract_id: Unique contract identifier.
        hashlock: SHA-256 hash of the preimage.
        timelock: Unix timestamp when the contract expires.
        sender: Sender address.
        receiver: Receiver address.
        amount: Locked amount.
        token: Token symbol.
        source_chain: Source chain.
        destination_chain: Destination chain.
        status: Current contract status.
    """

    contract_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hashlock: str = ""
    timelock: int = 0
    sender: str = ""
    receiver: str = ""
    amount: Decimal = Decimal("0")
    token: str = "USDC"
    source_chain: str = "ethereum"
    destination_chain: str = "arbitrum"
    status: HTLCStatus = HTLCStatus.INITIATED


# ---------------------------------------------------------------------------
# Bridge Interface
# ---------------------------------------------------------------------------

class Bridge(ABC):
    """Abstract base class for cross-chain bridge implementations."""

    @abstractmethod
    def transfer(self, request: BridgeTransfer) -> BridgeResult:
        """
        Execute a cross-chain transfer.

        Args:
            request: The bridge transfer request.

        Returns:
            BridgeResult with transfer details.

        Raises:
            BridgeError: If the transfer fails.
            InsufficientLiquidityError: If liquidity is insufficient.
        """

    @abstractmethod
    def estimate_fee(self, request: BridgeTransfer) -> Decimal:
        """
        Estimate the bridge fee for a transfer.

        Args:
            request: The bridge transfer request.

        Returns:
            Estimated fee in USD.
        """

    @abstractmethod
    def get_liquidity(self, source_chain: str, destination_chain: str, token: str) -> Decimal:
        """
        Get available bridge liquidity for a route.

        Args:
            source_chain: Source chain name.
            destination_chain: Destination chain name.
            token: Token symbol.

        Returns:
            Available liquidity in token units.
        """


# ---------------------------------------------------------------------------
# Concrete Bridges
# ---------------------------------------------------------------------------

class HopBridge(Bridge):
    """
    Hop Protocol bridge implementation.

    Uses bonded liquidity pools and AMM-based pricing for fast,
    low-cost cross-chain transfers.
    """

    def transfer(self, request: BridgeTransfer) -> BridgeResult:
        if request.amount > MAX_BRIDGE_AMOUNT:
            raise BridgeError(
                f"Amount {request.amount} exceeds max {MAX_BRIDGE_AMOUNT}"
            )
        try:
            source_tx = _generate_tx_hash()
            dest_tx = _generate_tx_hash()
            amount_out = request.amount - self.estimate_fee(request)
            return BridgeResult(
                transfer_id=request.transfer_id,
                status=BridgeStatus.COMPLETED,
                source_tx_hash=source_tx,
                destination_tx_hash=dest_tx,
                amount_out=amount_out,
                fee=self.estimate_fee(request),
                completed_at=_now_iso(),
            )
        except CrossChainError:
            raise
        except Exception as exc:
            raise BridgeError(f"Hop transfer failed: {exc}") from exc

    def estimate_fee(self, request: BridgeTransfer) -> Decimal:
        # Hop: 0.04% - 0.3% depending on route
        base_fee_rate = Decimal("0.0004")
        if request.token in ("ETH", "WETH"):
            base_fee_rate = Decimal("0.001")
        return request.amount * base_fee_rate

    def get_liquidity(self, source_chain: str, destination_chain: str, token: str) -> Decimal:
        # Simulated liquidity lookup
        return Decimal("1_000_000")


class AcrossBridge(Bridge):
    """
    Across Protocol bridge implementation.

    Uses optimistic verification with relayers for secure,
    cost-efficient cross-chain transfers.
    """

    def transfer(self, request: BridgeTransfer) -> BridgeResult:
        if request.amount > MAX_BRIDGE_AMOUNT:
            raise BridgeError(
                f"Amount {request.amount} exceeds max {MAX_BRIDGE_AMOUNT}"
            )
        try:
            source_tx = _generate_tx_hash()
            dest_tx = _generate_tx_hash()
            fee = self.estimate_fee(request)
            relay_fee = self._estimate_relay_fee(request)
            amount_out = request.amount - fee - relay_fee
            return BridgeResult(
                transfer_id=request.transfer_id,
                status=BridgeStatus.COMPLETED,
                source_tx_hash=source_tx,
                destination_tx_hash=dest_tx,
                amount_out=amount_out,
                fee=fee,
                relay_fee=relay_fee,
                completed_at=_now_iso(),
            )
        except CrossChainError:
            raise
        except Exception as exc:
            raise BridgeError(f"Across transfer failed: {exc}") from exc

    def estimate_fee(self, request: BridgeTransfer) -> Decimal:
        # Across: 0.01% - 0.1% LP fee
        return request.amount * Decimal("0.0006")

    def _estimate_relay_fee(self, request: BridgeTransfer) -> Decimal:
        # Relayer fee: flat 0.01% of amount
        return request.amount * Decimal("0.0001")

    def get_liquidity(self, source_chain: str, destination_chain: str, token: str) -> Decimal:
        return Decimal("2_000_000")


# ---------------------------------------------------------------------------
# Arbitrage Engine
# ---------------------------------------------------------------------------

class ArbitrageEngine:
    """
    Detects and executes cross-chain arbitrage opportunities.

    Monitors price discrepancies across DEXs on different chains
    and executes profitable cross-chain swaps.
    """

    def __init__(self, bridge: Bridge | None = None) -> None:
        self._bridge = bridge or HopBridge()
        self._opportunities: dict[str, ArbitrageOpportunity] = {}

    def detect_opportunity(
        self,
        source_chain: str,
        destination_chain: str,
        token: str,
        source_price: Decimal,
        destination_price: Decimal,
        amount: Decimal,
    ) -> ArbitrageOpportunity | None:
        """
        Detect a cross-chain arbitrage opportunity.

        Args:
            source_chain: Chain to buy on.
            destination_chain: Chain to sell on.
            token: Token symbol.
            source_price: Price on source chain (USD).
            destination_price: Price on destination chain (USD).
            amount: Trade amount in USD.

        Returns:
            ArbitrageOpportunity if profitable, None otherwise.

        Raises:
            ArbitrageError: If inputs are invalid.
        """
        if source_price <= Decimal("0") or destination_price <= Decimal("0"):
            raise ArbitrageError("Prices must be positive")
        if amount <= Decimal("0"):
            raise ArbitrageError("Amount must be positive")

        spread_pct = ((destination_price - source_price) / source_price) * Decimal("100")
        if spread_pct < MIN_ARBITRAGE_PROFIT:
            return None

        estimated_profit = amount * (spread_pct / Decimal("100"))

        # Estimate bridge fee
        bridge_fee = amount * Decimal("0.001")  # 0.1% estimated
        net_profit = estimated_profit - bridge_fee

        if net_profit <= Decimal("0"):
            return None

        opportunity = ArbitrageOpportunity(
            source_chain=source_chain,
            destination_chain=destination_chain,
            token=token,
            source_price=source_price,
            destination_price=destination_price,
            spread_pct=spread_pct,
            estimated_profit=estimated_profit,
            net_profit=net_profit,
            expires_at=_now_iso(),
        )
        self._opportunities[opportunity.opportunity_id] = opportunity
        return opportunity

    def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity,
        sender_address: str,
        receiver_address: str,
    ) -> BridgeResult:
        """
        Execute an arbitrage opportunity via bridge transfer.

        Args:
            opportunity: The detected opportunity.
            sender_address: Sender wallet address.
            receiver_address: Receiver wallet address.

        Returns:
            BridgeResult from the bridge transfer.

        Raises:
            ArbitrageError: If execution fails.
        """
        if opportunity.opportunity_id not in self._opportunities:
            raise ArbitrageError("Opportunity not found or expired")

        transfer = BridgeTransfer(
            bridge=BridgeType.HOP,
            source_chain=opportunity.source_chain,
            destination_chain=opportunity.destination_chain,
            token=opportunity.token,
            amount=opportunity.estimated_profit,
            sender_address=sender_address,
            receiver_address=receiver_address,
        )
        return self._bridge.transfer(transfer)

    def get_opportunity(self, opportunity_id: str) -> ArbitrageOpportunity | None:
        """Retrieve a stored opportunity by ID."""
        return self._opportunities.get(opportunity_id)


# ---------------------------------------------------------------------------
# Atomic Settlement (HTLC)
# ---------------------------------------------------------------------------

class AtomicSettlement:
    """
    Hash time-locked contract (HTLC) based atomic settlement.

    Enables trustless cross-chain atomic swaps without intermediaries.
    """

    def initiate(
        self,
        sender: str,
        receiver: str,
        amount: Decimal,
        token: str,
        source_chain: str,
        destination_chain: str,
        lock_duration: int = HTLC_LOCK_DURATION_SECONDS,
    ) -> tuple[HTLCContract, str]:
        """
        Initiate an HTLC for atomic settlement.

        Args:
            sender: Sender address.
            receiver: Receiver address.
            amount: Amount to lock.
            token: Token symbol.
            source_chain: Source chain.
            destination_chain: Destination chain.
            lock_duration: Lock duration in seconds.

        Returns:
            Tuple of (HTLCContract, preimage). The preimage must be
            kept secret by the sender until claiming.

        Raises:
            HTLCError: If initiation fails.
        """
        if amount <= Decimal("0"):
            raise HTLCError("Amount must be positive")
        if not sender or not receiver:
            raise HTLCError("Sender and receiver addresses required")

        preimage = secrets.token_hex(32)
        hashlock = _sha256_hex(preimage)
        timelock = int(_now_unix()) + lock_duration

        contract = HTLCContract(
            hashlock=hashlock,
            timelock=timelock,
            sender=sender,
            receiver=receiver,
            amount=amount,
            token=token,
            source_chain=source_chain,
            destination_chain=destination_chain,
            status=HTLCStatus.INITIATED,
        )
        return contract, preimage

    def claim(self, contract: HTLCContract, preimage: str) -> HTLCContract:
        """
        Claim funds from an HTLC by revealing the preimage.

        Args:
            contract: The HTLC contract.
            preimage: The preimage that matches the hashlock.

        Returns:
            Updated HTLCContract with CLAIMED status.

        Raises:
            HTLCError: If the preimage is invalid or contract expired.
        """
        if _sha256_hex(preimage) != contract.hashlock:
            raise HTLCError("Invalid preimage: hashlock mismatch")
        if int(_now_unix()) > contract.timelock:
            raise HTLCError("Contract has expired")

        return HTLCContract(
            contract_id=contract.contract_id,
            hashlock=contract.hashlock,
            timelock=contract.timelock,
            sender=contract.sender,
            receiver=contract.receiver,
            amount=contract.amount,
            token=contract.token,
            source_chain=contract.source_chain,
            destination_chain=contract.destination_chain,
            status=HTLCStatus.CLAIMED,
        )

    def refund(self, contract: HTLCContract) -> HTLCContract:
        """
        Refund an expired HTLC to the sender.

        Args:
            contract: The expired HTLC contract.

        Returns:
            Updated HTLCContract with REFUNDED status.

        Raises:
            HTLCError: If the contract has not yet expired.
        """
        if int(_now_unix()) <= contract.timelock:
            raise HTLCError("Contract has not expired yet")

        return HTLCContract(
            contract_id=contract.contract_id,
            hashlock=contract.hashlock,
            timelock=contract.timelock,
            sender=contract.sender,
            receiver=contract.receiver,
            amount=contract.amount,
            token=contract.token,
            source_chain=contract.source_chain,
            destination_chain=contract.destination_chain,
            status=HTLCStatus.REFUNDED,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _now_unix() -> int:
    """Return current Unix timestamp."""
    return int(datetime.now(timezone.utc).timestamp())


def _generate_tx_hash() -> str:
    """Generate a simulated transaction hash."""
    return "0x" + secrets.token_hex(32)


def _sha256_hex(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode()).hexdigest()


def parse_amount(value: str | float | Decimal) -> Decimal:
    """
    Safely parse a numeric value into a Decimal.

    Args:
        value: The value to parse.

    Returns:
        Decimal representation.

    Raises:
        BridgeError: If the value cannot be parsed.
    """
    try:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise BridgeError(f"Cannot parse amount: {value}") from exc
