"""
Apex FinTech Markets — Settlement Module.

Real-time settlement, bilateral and multilateral netting,
clearing house integration, and settlement finality tracking.
"""

from __future__ import annotations

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

MAX_NETTING_AMOUNT: Final[Decimal] = Decimal("100_000_000")
SETTLEMENT_WINDOW_HOURS: Final[int] = 24


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SettlementError(Exception):
    """Base exception for settlement operations."""


class NettingError(SettlementError):
    """Raised when netting operations fail."""


class ClearingError(SettlementError):
    """Raised when clearing house operations fail."""


class SettlementFinalityError(SettlementError):
    """Raised when settlement finality cannot be achieved."""


class InsufficientBalanceError(SettlementError):
    """Raised when a participant lacks sufficient balance."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SettlementType(Enum):
    """Type of settlement mechanism."""

    BILATERAL = auto()
    MULTILATERAL = auto()
    CENTRAL_COUNTERPARTY = auto()
    REAL_TIME_GROSS = auto()


class SettlementStatus(Enum):
    """Lifecycle status of a settlement instruction."""

    PENDING = auto()
    NETTED = auto()
    CLEARED = auto()
    SETTLED = auto()
    FAILED = auto()
    REVERSED = auto()


class NettingStatus(Enum):
    """Status of a netting cycle."""

    OPEN = auto()
    CALCULATING = auto()
    CONFIRMED = auto()
    SETTLED = auto()
    CANCELLED = auto()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SettlementInstruction:
    """
    Immutable settlement instruction.

    Attributes:
        instruction_id: Unique identifier.
        settlement_type: Settlement mechanism type.
        currency: ISO 4217 currency code.
        amount: Settlement amount.
        sender_id: Sender participant identifier.
        receiver_id: Receiver participant identifier.
        value_date: Requested settlement date (ISO 8601).
        priority: Settlement priority (1=highest, 5=lowest).
        metadata: Optional key-value metadata.
    """

    instruction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    settlement_type: SettlementType = SettlementType.BILATERAL
    currency: str = "USD"
    amount: Decimal = Decimal("0")
    sender_id: str = ""
    receiver_id: str = ""
    value_date: str = ""
    priority: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.amount <= Decimal("0"):
            raise SettlementError(f"Amount must be positive, got {self.amount}")
        if self.currency not in SUPPORTED_CURRENCIES:
            raise SettlementError(f"Unsupported currency: {self.currency}")
        if not self.sender_id or not self.receiver_id:
            raise SettlementError("Sender and receiver IDs are required")
        if self.priority < 1 or self.priority > 5:
            raise SettlementError("Priority must be between 1 and 5")


@dataclass(frozen=True, slots=True)
class SettlementResult:
    """
    Immutable settlement result.

    Attributes:
        instruction_id: Correlates to the originating instruction.
        status: Final settlement status.
        settled_at: ISO 8601 settlement timestamp.
        net_amount: Netted amount (if netting applied).
        clearing_house: Clearing house identifier (if CCP).
        settlement_reference: Settlement system reference.
        fee: Settlement fee charged.
        error_message: Error details if failed.
    """

    instruction_id: str
    status: SettlementStatus
    settled_at: str = ""
    net_amount: Decimal = Decimal("0")
    clearing_house: str = ""
    settlement_reference: str = ""
    fee: Decimal = Decimal("0")
    error_message: str = ""


@dataclass(frozen=True, slots=True)
class NettingPosition:
    """
    Immutable netting position for a participant pair.

    Attributes:
        netting_id: Unique netting cycle identifier.
        currency: Currency code.
        participant_a: First participant identifier.
        participant_b: Second participant identifier.
        gross_amount_a: Gross amount owed by A to B.
        gross_amount_b: Gross amount owed by B to A.
        net_amount: Net settlement amount.
        net_direction: Who owes the net amount ('A' or 'B').
        status: Netting cycle status.
    """

    netting_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    currency: str = "USD"
    participant_a: str = ""
    participant_b: str = ""
    gross_amount_a: Decimal = Decimal("0")
    gross_amount_b: Decimal = Decimal("0")
    net_amount: Decimal = Decimal("0")
    net_direction: str = ""
    status: NettingStatus = NettingStatus.OPEN


@dataclass(frozen=True, slots=True)
class ClearingHouse:
    """
    Immutable clearing house configuration.

    Attributes:
        house_id: Unique clearing house identifier.
        name: Human-readable name.
        supported_currencies: Set of supported currency codes.
        margin_rate: Initial margin rate as decimal.
        settlement_cycle_hours: Settlement cycle duration in hours.
    """

    house_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    supported_currencies: frozenset[str] = frozenset({"USD"})
    margin_rate: Decimal = Decimal("0.02")
    settlement_cycle_hours: int = 24


# ---------------------------------------------------------------------------
# Settlement Engine
# ---------------------------------------------------------------------------

class SettlementEngine:
    """
    Real-time settlement engine supporting multiple settlement modes.

    Handles bilateral, multilateral, and CCP-based settlement
    with real-time gross settlement (RTGS) capability.
    """

    def __init__(self) -> None:
        self._instructions: dict[str, SettlementInstruction] = {}
        self._results: dict[str, SettlementResult] = {}
        self._clearing_houses: dict[str, ClearingHouse] = {}

    def register_clearing_house(self, house: ClearingHouse) -> None:
        """
        Register a clearing house for CCP settlement.

        Args:
            house: The clearing house configuration.
        """
        self._clearing_houses[house.house_id] = house

    def settle(self, instruction: SettlementInstruction) -> SettlementResult:
        """
        Execute settlement for an instruction.

        Args:
            instruction: The settlement instruction.

        Returns:
            SettlementResult with settlement details.

        Raises:
            SettlementError: If settlement fails.
        """
        self._instructions[instruction.instruction_id] = instruction
        try:
            if instruction.settlement_type == SettlementType.BILATERAL:
                return self._settle_bilateral(instruction)
            elif instruction.settlement_type == SettlementType.MULTILATERAL:
                return self._settle_multilateral(instruction)
            elif instruction.settlement_type == SettlementType.CENTRAL_COUNTERPARTY:
                return self._settle_ccp(instruction)
            elif instruction.settlement_type == SettlementType.REAL_TIME_GROSS:
                return self._settle_rtgs(instruction)
            else:
                raise SettlementError(
                    f"Unknown settlement type: {instruction.settlement_type}"
                )
        except SettlementError:
            raise
        except Exception as exc:
            raise SettlementError(f"Settlement failed: {exc}") from exc

    def _settle_bilateral(self, instruction: SettlementInstruction) -> SettlementResult:
        """Execute bilateral gross settlement."""
        result = SettlementResult(
            instruction_id=instruction.instruction_id,
            status=SettlementStatus.SETTLED,
            settled_at=_now_iso(),
            net_amount=instruction.amount,
            settlement_reference=_generate_settlement_ref("BILAT"),
            fee=self._calculate_fee(instruction),
        )
        self._results[instruction.instruction_id] = result
        return result

    def _settle_multilateral(self, instruction: SettlementInstruction) -> SettlementResult:
        """Execute multilateral net settlement."""
        result = SettlementResult(
            instruction_id=instruction.instruction_id,
            status=SettlementStatus.SETTLED,
            settled_at=_now_iso(),
            net_amount=instruction.amount,
            settlement_reference=_generate_settlement_ref("MULTI"),
            fee=self._calculate_fee(instruction),
        )
        self._results[instruction.instruction_id] = result
        return result

    def _settle_ccp(self, instruction: SettlementInstruction) -> SettlementResult:
        """Execute CCP-based settlement."""
        if not self._clearing_houses:
            raise ClearingError("No clearing houses registered")

        house = next(iter(self._clearing_houses.values()))
        if instruction.currency not in house.supported_currencies:
            raise ClearingError(
                f"Currency {instruction.currency} not supported by {house.name}"
            )

        margin = instruction.amount * house.margin_rate
        result = SettlementResult(
            instruction_id=instruction.instruction_id,
            status=SettlementStatus.SETTLED,
            settled_at=_now_iso(),
            net_amount=instruction.amount - margin,
            clearing_house=house.house_id,
            settlement_reference=_generate_settlement_ref("CCP"),
            fee=self._calculate_fee(instruction) + margin,
        )
        self._results[instruction.instruction_id] = result
        return result

    def _settle_rtgs(self, instruction: SettlementInstruction) -> SettlementResult:
        """Execute real-time gross settlement."""
        result = SettlementResult(
            instruction_id=instruction.instruction_id,
            status=SettlementStatus.SETTLED,
            settled_at=_now_iso(),
            net_amount=instruction.amount,
            settlement_reference=_generate_settlement_ref("RTGS"),
            fee=self._calculate_fee(instruction),
        )
        self._results[instruction.instruction_id] = result
        return result

    def _calculate_fee(self, instruction: SettlementInstruction) -> Decimal:
        """Calculate settlement fee based on amount and priority."""
        base_fee = Decimal("0.10")
        priority_multiplier = Decimal(6 - instruction.priority) * Decimal("0.05")
        amount_fee = instruction.amount * Decimal("0.0001")
        return base_fee + priority_multiplier + amount_fee

    def get_result(self, instruction_id: str) -> SettlementResult | None:
        """Retrieve a settlement result by instruction ID."""
        return self._results.get(instruction_id)


# ---------------------------------------------------------------------------
# Netting Engine
# ---------------------------------------------------------------------------

class NettingEngine:
    """
    Bilateral and multilateral netting engine.

    Calculates net positions across participant groups to reduce
    settlement volume and liquidity requirements.
    """

    def __init__(self) -> None:
        self._cycles: dict[str, NettingPosition] = {}
        self._instructions: dict[str, list[SettlementInstruction]] = {}

    def add_instruction(self, instruction: SettlementInstruction) -> None:
        """
        Add a settlement instruction to the netting pool.

        Args:
            instruction: The settlement instruction to add.
        """
        key = _netting_key(instruction.sender_id, instruction.receiver_id, instruction.currency)
        if key not in self._instructions:
            self._instructions[key] = []
        self._instructions[key].append(instruction)

    def calculate_bilateral_netting(
        self,
        participant_a: str,
        participant_b: str,
        currency: str,
    ) -> NettingPosition:
        """
        Calculate bilateral net position between two participants.

        Args:
            participant_a: First participant identifier.
            participant_b: Second participant identifier.
            currency: Currency code.

        Returns:
            NettingPosition with net amounts.

        Raises:
            NettingError: If netting calculation fails.
        """
        key = _netting_key(participant_a, participant_b, currency)
        instructions = self._instructions.get(key, [])

        gross_a = Decimal("0")
        gross_b = Decimal("0")

        for inst in instructions:
            if inst.sender_id == participant_a and inst.receiver_id == participant_b:
                gross_a += inst.amount
            elif inst.sender_id == participant_b and inst.receiver_id == participant_a:
                gross_b += inst.amount

        if gross_a >= gross_b:
            net_amount = gross_a - gross_b
            net_direction = "A"
        else:
            net_amount = gross_b - gross_a
            net_direction = "B"

        position = NettingPosition(
            currency=currency,
            participant_a=participant_a,
            participant_b=participant_b,
            gross_amount_a=gross_a,
            gross_amount_b=gross_b,
            net_amount=net_amount,
            net_direction=net_direction,
            status=NettingStatus.CONFIRMED,
        )
        self._cycles[position.netting_id] = position
        return position

    def calculate_multilateral_netting(
        self,
        participants: list[str],
        currency: str,
    ) -> dict[str, Decimal]:
        """
        Calculate multilateral net positions for a participant group.

        Args:
            participants: List of participant identifiers.
            currency: Currency code.

        Returns:
            Dictionary mapping participant ID to net position.
            Positive = net receiver, Negative = net payer.

        Raises:
            NettingError: If netting calculation fails.
        """
        if len(participants) < 2:
            raise NettingError("At least 2 participants required for netting")

        net_positions: dict[str, Decimal] = {p: Decimal("0") for p in participants}

        for i, sender in enumerate(participants):
            for receiver in participants[i + 1:]:
                key = _netting_key(sender, receiver, currency)
                instructions = self._instructions.get(key, [])

                for inst in instructions:
                    if inst.sender_id == sender and inst.receiver_id == receiver:
                        net_positions[sender] -= inst.amount
                        net_positions[receiver] += inst.amount
                    elif inst.sender_id == receiver and inst.receiver_id == sender:
                        net_positions[receiver] -= inst.amount
                        net_positions[sender] += inst.amount

        return net_positions

    def get_cycle(self, netting_id: str) -> NettingPosition | None:
        """Retrieve a netting cycle by ID."""
        return self._cycles.get(netting_id)


# ---------------------------------------------------------------------------
# Clearing House Interface
# ---------------------------------------------------------------------------

class ClearingHouseInterface(ABC):
    """Abstract interface for clearing house integration."""

    @abstractmethod
    def submit_trade(self, instruction: SettlementInstruction) -> str:
        """
        Submit a trade to the clearing house.

        Args:
            instruction: The settlement instruction.

        Returns:
            Clearing house trade reference.
        """

    @abstractmethod
    def get_margin_requirement(self, participant_id: str, currency: str) -> Decimal:
        """
        Get margin requirement for a participant.

        Args:
            participant_id: Participant identifier.
            currency: Currency code.

        Returns:
            Margin requirement amount.
        """

    @abstractmethod
    def confirm_settlement(self, settlement_reference: str) -> bool:
        """
        Confirm settlement finality.

        Args:
            settlement_reference: Settlement reference.

        Returns:
            True if settlement is confirmed final.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _generate_settlement_ref(prefix: str) -> str:
    """Generate a unique settlement reference."""
    return f"{prefix}-{uuid.uuid4().hex[:16].upper()}"


def _netting_key(sender: str, receiver: str, currency: str) -> str:
    """Generate a canonical netting key for a participant pair."""
    sorted_pair = sorted([sender, receiver])
    return f"{sorted_pair[0]}:{sorted_pair[1]}:{currency}"


def parse_amount(value: str | float | Decimal) -> Decimal:
    """
    Safely parse a numeric value into a Decimal.

    Args:
        value: The value to parse.

    Returns:
        Decimal representation.

    Raises:
        SettlementError: If the value cannot be parsed.
    """
    try:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SettlementError(f"Cannot parse amount: {value}") from exc
