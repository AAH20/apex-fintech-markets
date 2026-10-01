"""
Apex FinTech Markets — Payment Rails Module.

Real-time payment processing for domestic instant payment networks
(RTP, FedNow), cross-border networks (SWIFT, SEPA), and stablecoin
settlement integration.

All processors share a common interface and route through PaymentRouter.
"""

from __future__ import annotations

import hashlib
import re
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

SUPPORTED_CURRENCIES: Final[frozenset[str]] = frozenset({
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD",
    "USDC", "USDT", "DAI",
})

MAX_RTP_AMOUNT: Final[Decimal] = Decimal("1_000_000")
MAX_FEDNOW_AMOUNT: Final[Decimal] = Decimal("5_000_000")
MAX_SWIFT_AMOUNT: Final[Decimal] = Decimal("100_000_000")
MAX_SEPA_AMOUNT: Final[Decimal] = Decimal("10_000_000")

SWIFT_BIC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$"
)
IBAN_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$")
ETH_ADDRESS_PATTERN: Final[re.Pattern[str]] = re.compile(r"^0x[a-fA-F0-9]{40}$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PaymentError(Exception):
    """Base exception for all payment-rail errors."""


class ValidationError(PaymentError):
    """Raised when a payment instruction fails validation."""


class ProcessingError(PaymentError):
    """Raised when a processor fails to execute a payment."""


class InsufficientLiquidityError(ProcessingError):
    """Raised when the rail lacks liquidity to settle the payment."""


class RailUnavailableError(ProcessingError):
    """Raised when the target rail is unreachable or down."""


class ComplianceHoldError(ProcessingError):
    """Raised when a compliance check blocks the payment."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PaymentRail(Enum):
    """Supported payment rail types."""

    RTP = auto()
    FEDNOW = auto()
    SWIFT = auto()
    SEPA = auto()
    STABLECOIN = auto()


class PaymentStatus(Enum):
    """Lifecycle status of a payment instruction."""

    PENDING = auto()
    VALIDATED = auto()
    PROCESSING = auto()
    SETTLED = auto()
    FAILED = auto()
    REVERSED = auto()
    COMPLIANCE_HOLD = auto()


class StablecoinNetwork(Enum):
    """Blockchain networks for stablecoin settlement."""

    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    SOLANA = "solana"
    TRON = "tron"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PaymentInstruction:
    """
    Immutable payment instruction submitted to a rail.

    Attributes:
        instruction_id: Unique identifier (UUID4).
        rail: Target payment rail.
        sender_bic: Sender's BIC (SWIFT) or routing identifier.
        sender_iban: Sender's IBAN or account identifier.
        receiver_bic: Receiver's BIC.
        receiver_iban: Receiver's IBAN or account identifier.
        amount: Payment amount as Decimal.
        currency: ISO 4217 currency code or stablecoin ticker.
        reference: End-to-end payment reference.
        value_date: Requested settlement date (ISO 8601).
        metadata: Optional key-value metadata.
    """

    instruction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rail: PaymentRail = PaymentRail.RTP
    sender_bic: str = ""
    sender_iban: str = ""
    receiver_bic: str = ""
    receiver_iban: str = ""
    amount: Decimal = Decimal("0")
    currency: str = "USD"
    reference: str = ""
    value_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.amount <= Decimal("0"):
            raise ValidationError(
                f"Amount must be positive, got {self.amount}"
            )
        if self.currency not in SUPPORTED_CURRENCIES:
            raise ValidationError(f"Unsupported currency: {self.currency}")


@dataclass(frozen=True, slots=True)
class PaymentResult:
    """
    Immutable result returned by a payment processor.

    Attributes:
        instruction_id: Correlates to the originating instruction.
        status: Final payment status.
        rail: Rail that processed the payment.
        settled_at: ISO 8601 timestamp of settlement.
        rail_reference: Rail-side transaction reference.
        fee: Processing fee charged.
        fx_rate: FX rate applied (if cross-currency).
        error_message: Human-readable error if failed.
    """

    instruction_id: str
    status: PaymentStatus
    rail: PaymentRail
    settled_at: str = ""
    rail_reference: str = ""
    fee: Decimal = Decimal("0")
    fx_rate: Decimal | None = None
    error_message: str = ""


# ---------------------------------------------------------------------------
# Processor Interface
# ---------------------------------------------------------------------------

class PaymentProcessor(ABC):
    """Abstract base class for all payment-rail processors."""

    @abstractmethod
    def validate(self, instruction: PaymentInstruction) -> None:
        """
        Validate a payment instruction against rail-specific rules.

        Args:
            instruction: The payment instruction to validate.

        Raises:
            ValidationError: If the instruction is invalid for this rail.
        """

    @abstractmethod
    def process(self, instruction: PaymentInstruction) -> PaymentResult:
        """
        Execute the payment on the target rail.

        Args:
            instruction: A validated payment instruction.

        Returns:
            PaymentResult with settlement details.

        Raises:
            ProcessingError: If the rail fails to process the payment.
            InsufficientLiquidityError: If liquidity is insufficient.
            RailUnavailableError: If the rail is unreachable.
        """

    @abstractmethod
    def get_fee(self, instruction: PaymentInstruction) -> Decimal:
        """
        Calculate the processing fee for an instruction.

        Args:
            instruction: The payment instruction.

        Returns:
            Fee as a Decimal.
        """


# ---------------------------------------------------------------------------
# Concrete Processors
# ---------------------------------------------------------------------------

class RTPProcessor(PaymentProcessor):
    """Real-Time Payments (RTP) network processor for USD instant payments."""

    def validate(self, instruction: PaymentInstruction) -> None:
        if instruction.currency != "USD":
            raise ValidationError("RTP only supports USD")
        if instruction.amount > MAX_RTP_AMOUNT:
            raise ValidationError(
                f"RTP amount {instruction.amount} exceeds limit {MAX_RTP_AMOUNT}"
            )
        if not instruction.sender_iban or not instruction.receiver_iban:
            raise ValidationError("RTP requires sender and receiver account identifiers")

    def process(self, instruction: PaymentInstruction) -> PaymentResult:
        self.validate(instruction)
        try:
            rail_ref = _generate_rail_reference("RTP")
            return PaymentResult(
                instruction_id=instruction.instruction_id,
                status=PaymentStatus.SETTLED,
                rail=PaymentRail.RTP,
                settled_at=_now_iso(),
                rail_reference=rail_ref,
                fee=self.get_fee(instruction),
            )
        except PaymentError:
            raise
        except Exception as exc:
            raise ProcessingError(f"RTP processing failed: {exc}") from exc

    def get_fee(self, instruction: PaymentInstruction) -> Decimal:
        return Decimal("0.01")


class FedNowProcessor(PaymentProcessor):
    """FedNow Service processor for USD instant payments (24/7/365)."""

    def validate(self, instruction: PaymentInstruction) -> None:
        if instruction.currency != "USD":
            raise ValidationError("FedNow only supports USD")
        if instruction.amount > MAX_FEDNOW_AMOUNT:
            raise ValidationError(
                f"FedNow amount {instruction.amount} exceeds limit {MAX_FEDNOW_AMOUNT}"
            )
        if not instruction.sender_iban or not instruction.receiver_iban:
            raise ValidationError("FedNow requires sender and receiver account identifiers")

    def process(self, instruction: PaymentInstruction) -> PaymentResult:
        self.validate(instruction)
        try:
            rail_ref = _generate_rail_reference("FEDNOW")
            return PaymentResult(
                instruction_id=instruction.instruction_id,
                status=PaymentStatus.SETTLED,
                rail=PaymentRail.FEDNOW,
                settled_at=_now_iso(),
                rail_reference=rail_ref,
                fee=self.get_fee(instruction),
            )
        except PaymentError:
            raise
        except Exception as exc:
            raise ProcessingError(f"FedNow processing failed: {exc}") from exc

    def get_fee(self, instruction: PaymentInstruction) -> Decimal:
        return Decimal("0.05")


class SWIFTProcessor(PaymentProcessor):
    """SWIFT MT103 cross-border payment processor."""

    def validate(self, instruction: PaymentInstruction) -> None:
        if not SWIFT_BIC_PATTERN.match(instruction.sender_bic):
            raise ValidationError(f"Invalid sender BIC: {instruction.sender_bic}")
        if not SWIFT_BIC_PATTERN.match(instruction.receiver_bic):
            raise ValidationError(f"Invalid receiver BIC: {instruction.receiver_bic}")
        if instruction.amount > MAX_SWIFT_AMOUNT:
            raise ValidationError(
                f"SWIFT amount {instruction.amount} exceeds limit {MAX_SWIFT_AMOUNT}"
            )
        if not instruction.reference:
            raise ValidationError("SWIFT requires a payment reference")

    def process(self, instruction: PaymentInstruction) -> PaymentResult:
        self.validate(instruction)
        try:
            rail_ref = _generate_rail_reference("SWIFT")
            return PaymentResult(
                instruction_id=instruction.instruction_id,
                status=PaymentStatus.SETTLED,
                rail=PaymentRail.SWIFT,
                settled_at=_now_iso(),
                rail_reference=rail_ref,
                fee=self.get_fee(instruction),
            )
        except PaymentError:
            raise
        except Exception as exc:
            raise ProcessingError(f"SWIFT processing failed: {exc}") from exc

    def get_fee(self, instruction: PaymentInstruction) -> Decimal:
        base_fee = Decimal("15.00")
        if instruction.amount > Decimal("1_000_000"):
            return base_fee + Decimal("10.00")
        return base_fee


class SEPAProcessor(PaymentProcessor):
    """SEPA Credit Transfer processor for EUR payments."""

    def validate(self, instruction: PaymentInstruction) -> None:
        if instruction.currency != "EUR":
            raise ValidationError("SEPA only supports EUR")
        if instruction.amount > MAX_SEPA_AMOUNT:
            raise ValidationError(
                f"SEPA amount {instruction.amount} exceeds limit {MAX_SEPA_AMOUNT}"
            )
        if not IBAN_PATTERN.match(instruction.sender_iban):
            raise ValidationError(f"Invalid sender IBAN: {instruction.sender_iban}")
        if not IBAN_PATTERN.match(instruction.receiver_iban):
            raise ValidationError(f"Invalid receiver IBAN: {instruction.receiver_iban}")

    def process(self, instruction: PaymentInstruction) -> PaymentResult:
        self.validate(instruction)
        try:
            rail_ref = _generate_rail_reference("SEPA")
            return PaymentResult(
                instruction_id=instruction.instruction_id,
                status=PaymentStatus.SETTLED,
                rail=PaymentRail.SEPA,
                settled_at=_now_iso(),
                rail_reference=rail_ref,
                fee=self.get_fee(instruction),
            )
        except PaymentError:
            raise
        except Exception as exc:
            raise ProcessingError(f"SEPA processing failed: {exc}") from exc

    def get_fee(self, instruction: PaymentInstruction) -> Decimal:
        return Decimal("0.20")


class StablecoinProcessor(PaymentProcessor):
    """Stablecoin settlement processor for on-chain USDC/USDT/DAI transfers."""

    def __init__(self, network: StablecoinNetwork = StablecoinNetwork.ETHEREUM) -> None:
        self._network = network

    @property
    def network(self) -> StablecoinNetwork:
        return self._network

    def validate(self, instruction: PaymentInstruction) -> None:
        if instruction.currency not in ("USDC", "USDT", "DAI"):
            raise ValidationError(
                f"Stablecoin rail only supports USDC/USDT/DAI, got {instruction.currency}"
            )
        if not ETH_ADDRESS_PATTERN.match(instruction.sender_iban):
            raise ValidationError(
                f"Invalid sender address: {instruction.sender_iban}"
            )
        if not ETH_ADDRESS_PATTERN.match(instruction.receiver_iban):
            raise ValidationError(
                f"Invalid receiver address: {instruction.receiver_iban}"
            )

    def process(self, instruction: PaymentInstruction) -> PaymentResult:
        self.validate(instruction)
        try:
            rail_ref = _generate_rail_reference("STABLE")
            return PaymentResult(
                instruction_id=instruction.instruction_id,
                status=PaymentStatus.SETTLED,
                rail=PaymentRail.STABLECOIN,
                settled_at=_now_iso(),
                rail_reference=rail_ref,
                fee=self.get_fee(instruction),
            )
        except PaymentError:
            raise
        except Exception as exc:
            raise ProcessingError(f"Stablecoin processing failed: {exc}") from exc

    def get_fee(self, instruction: PaymentInstruction) -> Decimal:
        network_fees: dict[StablecoinNetwork, Decimal] = {
            StablecoinNetwork.ETHEREUM: Decimal("2.50"),
            StablecoinNetwork.POLYGON: Decimal("0.01"),
            StablecoinNetwork.ARBITRUM: Decimal("0.10"),
            StablecoinNetwork.OPTIMISM: Decimal("0.05"),
            StablecoinNetwork.SOLANA: Decimal("0.001"),
            StablecoinNetwork.TRON: Decimal("1.00"),
        }
        return network_fees.get(self._network, Decimal("1.00"))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class PaymentRouter:
    """
    Routes payment instructions to the appropriate rail processor.

    Usage:
        router = PaymentRouter()
        result = router.route(instruction)
    """

    def __init__(self) -> None:
        self._processors: dict[PaymentRail, PaymentProcessor] = {
            PaymentRail.RTP: RTPProcessor(),
            PaymentRail.FEDNOW: FedNowProcessor(),
            PaymentRail.SWIFT: SWIFTProcessor(),
            PaymentRail.SEPA: SEPAProcessor(),
            PaymentRail.STABLECOIN: StablecoinProcessor(),
        }

    def register_processor(
        self, rail: PaymentRail, processor: PaymentProcessor
    ) -> None:
        """
        Register or override a processor for a given rail.

        Args:
            rail: The payment rail to register.
            processor: The processor instance.
        """
        self._processors[rail] = processor

    def route(self, instruction: PaymentInstruction) -> PaymentResult:
        """
        Route a payment instruction to the correct processor.

        Args:
            instruction: The payment instruction to route.

        Returns:
            PaymentResult from the selected processor.

        Raises:
            ProcessingError: If no processor exists for the rail.
        """
        processor = self._processors.get(instruction.rail)
        if processor is None:
            raise ProcessingError(
                f"No processor registered for rail {instruction.rail.name}"
            )
        return processor.process(instruction)

    def get_fee_estimate(self, instruction: PaymentInstruction) -> Decimal:
        """
        Get a fee estimate without processing the payment.

        Args:
            instruction: The payment instruction.

        Returns:
            Estimated fee as Decimal.
        """
        processor = self._processors.get(instruction.rail)
        if processor is None:
            raise ProcessingError(
                f"No processor registered for rail {instruction.rail.name}"
            )
        return processor.get_fee(instruction)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _generate_rail_reference(prefix: str) -> str:
    """Generate a unique rail-side reference."""
    raw = f"{prefix}:{uuid.uuid4().hex[:16]}"
    return f"{prefix}-{hashlib.sha256(raw.encode()).hexdigest()[:16].upper()}"


def parse_amount(value: str | float | Decimal) -> Decimal:
    """
    Safely parse a numeric value into a Decimal.

    Args:
        value: The value to parse.

    Returns:
        Decimal representation.

    Raises:
        ValidationError: If the value cannot be parsed.
    """
    try:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(f"Cannot parse amount: {value}") from exc
