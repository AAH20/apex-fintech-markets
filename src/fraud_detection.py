"""
Apex FinTech Markets — Fraud Detection Module.

Real-time fraud detection, AML compliance, transaction monitoring,
and risk scoring using rule-based and behavioral analytics.
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

HIGH_RISK_COUNTRIES: Final[frozenset[str]] = frozenset({
    "AF", "BY", "CF", "CU", "IR", "KP", "LY", "MM", "RU", "SO", "SD", "SY", "VE", "YE", "ZW",
})

SANCTIONED_ADDRESSES: Final[frozenset[str]] = frozenset({
    # Example sanctioned addresses (lowercase)
    "0x1da5821544e25c636c1417ba96ade4cf6d2f9b5a5",
    "0x098b716b8aaf21512996dc57eb0615e2383e2f96",
})

MAX_TRANSACTION_AMOUNT: Final[Decimal] = Decimal("10_000_000")
MAX_VELOCITY_PER_HOUR: Final[int] = 50
MAX_VELOCITY_AMOUNT_PER_HOUR: Final[Decimal] = Decimal("1_000_000")
RISK_SCORE_THRESHOLD: Final[int] = 70
AML_ALERT_THRESHOLD: Final[int] = 50

ETH_ADDRESS_PATTERN: Final[re.Pattern[str]] = re.compile(r"^0x[a-fA-F0-9]{40}$")
SUSPICIOUS_PATTERNS: Final[list[str]] = [
    r"round_number",      # Round number transactions
    r"rapid_sequence",    # Rapid-fire transactions
    r"amount_structuring",  # Structuring below reporting threshold
    r"high_risk_geo",     # High-risk geography
    r"sanctioned",        # Sanctioned entity
    r"unusual_hours",     # Unusual hour activity
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FraudDetectionError(Exception):
    """Base exception for fraud detection operations."""


class ValidationError(FraudDetectionError):
    """Raised when transaction data fails validation."""


class RuleEngineError(FraudDetectionError):
    """Raised when the rule engine encounters an error."""


class AMLComplianceError(FraudDetectionError):
    """Raised when AML compliance check fails."""


class TransactionBlockedError(FraudDetectionError):
    """Raised when a transaction is blocked by fraud rules."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    """Risk classification levels."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class AlertStatus(Enum):
    """Status of a fraud/AML alert."""

    OPEN = auto()
    UNDER_REVIEW = auto()
    CONFIRMED_FRAUD = auto()
    FALSE_POSITIVE = auto()
    CLOSED = auto()


class TransactionType(Enum):
    """Types of monitored transactions."""

    DOMESTIC = auto()
    CROSS_BORDER = auto()
    CRYPTO = auto()
    ATM = auto()
    ONLINE = auto()
    WIRE = auto()


class AMLCheckType(Enum):
    """Types of AML compliance checks."""

    SANCTIONS_SCREENING = auto()
    PEP_CHECK = auto()
    ADVERSE_MEDIA = auto()
    TRANSACTION_MONITORING = auto()
    SUSPICIOUS_ACTIVITY_REPORT = auto()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Transaction:
    """
    Immutable transaction record for monitoring.

    Attributes:
        transaction_id: Unique transaction identifier.
        transaction_type: Type of transaction.
        sender_id: Sender identifier.
        receiver_id: Receiver identifier.
        amount: Transaction amount.
        currency: ISO 4217 currency code.
        timestamp: ISO 8601 transaction timestamp.
        source_country: ISO 3166-1 alpha-2 source country code.
        destination_country: ISO 3166-1 alpha-2 destination country code.
        payment_rail: Payment rail used.
        metadata: Optional key-value metadata.
    """

    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transaction_type: TransactionType = TransactionType.DOMESTIC
    sender_id: str = ""
    receiver_id: str = ""
    amount: Decimal = Decimal("0")
    currency: str = "USD"
    timestamp: str = ""
    source_country: str = ""
    destination_country: str = ""
    payment_rail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.amount <= Decimal("0"):
            raise ValidationError(f"Amount must be positive, got {self.amount}")
        if not self.sender_id or not self.receiver_id:
            raise ValidationError("Sender and receiver IDs are required")


@dataclass(frozen=True, slots=True)
class RiskScore:
    """
    Immutable risk score for a transaction.

    Attributes:
        transaction_id: Correlates to the transaction.
        score: Risk score (0-100).
        level: Risk classification level.
        factors: List of risk factors that contributed to the score.
        timestamp: ISO 8601 scoring timestamp.
    """

    transaction_id: str
    score: int = 0
    level: RiskLevel = RiskLevel.LOW
    factors: tuple[str, ...] = ()
    timestamp: str = ""


@dataclass(frozen=True, slots=True)
class FraudAlert:
    """
    Immutable fraud/AML alert.

    Attributes:
        alert_id: Unique alert identifier.
        transaction_id: Correlates to the transaction.
        alert_type: Type of alert.
        status: Current alert status.
        risk_score: Associated risk score.
        description: Human-readable alert description.
        created_at: ISO 8601 creation timestamp.
        resolved_at: ISO 8601 resolution timestamp.
        assigned_to: Analyst assigned to the alert.
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str = ""
    alert_type: str = ""
    status: AlertStatus = AlertStatus.OPEN
    risk_score: int = 0
    description: str = ""
    created_at: str = ""
    resolved_at: str = ""
    assigned_to: str = ""


@dataclass(frozen=True, slots=True)
class AMLReport:
    """
    Immutable AML compliance report.

    Attributes:
        report_id: Unique report identifier.
        transaction_id: Correlates to the transaction.
        check_type: Type of AML check performed.
        passed: Whether the check passed.
        details: Check details.
        reported_at: ISO 8601 report timestamp.
    """

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str = ""
    check_type: AMLCheckType = AMLCheckType.SANCTIONS_SCREENING
    passed: bool = True
    details: str = ""
    reported_at: str = ""


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

class FraudRule(ABC):
    """Abstract base class for fraud detection rules."""

    @abstractmethod
    def evaluate(self, transaction: Transaction, context: dict[str, Any]) -> int:
        """
        Evaluate a transaction against this rule.

        Args:
            transaction: The transaction to evaluate.
            context: Additional context (velocity, history, etc.).

        Returns:
            Risk score contribution (0-100).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable rule name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Rule description."""


class AmountThresholdRule(FraudRule):
    """Flags transactions exceeding amount thresholds."""

    def __init__(self, threshold: Decimal = Decimal("10_000")) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "amount_threshold"

    @property
    def description(self) -> str:
        return f"Flags transactions exceeding {self._threshold}"

    def evaluate(self, transaction: Transaction, context: dict[str, Any]) -> int:
        if transaction.amount > self._threshold * Decimal("10"):
            return 40
        elif transaction.amount > self._threshold * Decimal("5"):
            return 25
        elif transaction.amount > self._threshold:
            return 10
        return 0


class VelocityRule(FraudRule):
    """Flags excessive transaction velocity."""

    def __init__(self, max_per_hour: int = MAX_VELOCITY_PER_HOUR) -> None:
        self._max_per_hour = max_per_hour

    @property
    def name(self) -> str:
        return "velocity_check"

    @property
    def description(self) -> str:
        return f"Flags >{self._max_per_hour} transactions/hour"

    def evaluate(self, transaction: Transaction, context: dict[str, Any]) -> int:
        velocity = context.get("velocity_count", 0)
        if velocity > self._max_per_hour * 2:
            return 35
        elif velocity > self._max_per_hour:
            return 20
        return 0


class HighRiskCountryRule(FraudRule):
    """Flags transactions involving high-risk countries."""

    @property
    def name(self) -> str:
        return "high_risk_country"

    @property
    def description(self) -> str:
        return "Flags high-risk jurisdiction involvement"

    def evaluate(self, transaction: Transaction, context: dict[str, Any]) -> int:
        score = 0
        if transaction.source_country in HIGH_RISK_COUNTRIES:
            score += 30
        if transaction.destination_country in HIGH_RISK_COUNTRIES:
            score += 30
        return score


class RoundAmountRule(FraudRule):
    """Flags round-number transactions (potential structuring)."""

    @property
    def name(self) -> str:
        return "round_amount"

    @property
    def description(self) -> str:
        return "Flags round-number amounts"

    def evaluate(self, transaction: Transaction, context: dict[str, Any]) -> int:
        amount_str = str(transaction.amount)
        if re.match(r"^\d+\.00$", amount_str):
            return 15
        if re.match(r"^\d{4,}\.00$", amount_str):
            return 20
        return 0


class SanctionedAddressRule(FraudRule):
    """Flags transactions involving sanctioned addresses."""

    @property
    def name(self) -> str:
        return "sanctioned_address"

    @property
    def description(self) -> str:
        return "Flags sanctioned address involvement"

    def evaluate(self, transaction: Transaction, context: dict[str, Any]) -> int:
        sender = transaction.sender_id.lower()
        receiver = transaction.receiver_id.lower()
        if sender in SANCTIONED_ADDRESSES or receiver in SANCTIONED_ADDRESSES:
            return 100
        return 0


class StructuringRule(FraudRule):
    """Detects potential structuring (keeping amounts below reporting threshold)."""

    def __init__(self, threshold: Decimal = Decimal("10_000")) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "structuring"

    @property
    def description(self) -> str:
        return "Detects potential structuring behavior"

    def evaluate(self, transaction: Transaction, context: dict[str, Any]) -> int:
        recent_amounts = context.get("recent_amounts", [])
        if len(recent_amounts) < 3:
            return 0

        # Check if multiple transactions are just below threshold
        below_threshold = sum(
            1 for a in recent_amounts
            if a > self._threshold * Decimal("0.8") and a < self._threshold
        )
        if below_threshold >= 3:
            return 45
        return 0


# ---------------------------------------------------------------------------
# Fraud Detection Engine
# ---------------------------------------------------------------------------

class FraudDetectionEngine:
    """
    Real-time fraud detection engine.

    Evaluates transactions against a configurable rule set and
    produces risk scores and alerts.
    """

    def __init__(self) -> None:
        self._rules: list[FraudRule] = []
        self._alerts: dict[str, FraudAlert] = {}
        self._transaction_history: dict[str, list[Transaction]] = {}
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Register default fraud detection rules."""
        self._rules = [
            AmountThresholdRule(),
            VelocityRule(),
            HighRiskCountryRule(),
            RoundAmountRule(),
            SanctionedAddressRule(),
            StructuringRule(),
        ]

    def add_rule(self, rule: FraudRule) -> None:
        """
        Add a custom fraud detection rule.

        Args:
            rule: The rule to add.
        """
        self._rules.append(rule)

    def evaluate_transaction(self, transaction: Transaction) -> RiskScore:
        """
        Evaluate a transaction and return a risk score.

        Args:
            transaction: The transaction to evaluate.

        Returns:
            RiskScore with overall score and contributing factors.

        Raises:
            FraudDetectionError: If evaluation fails.
        """
        context = self._build_context(transaction)
        total_score = 0
        factors: list[str] = []

        for rule in self._rules:
            try:
                score = rule.evaluate(transaction, context)
                if score > 0:
                    total_score += score
                    factors.append(f"{rule.name}:+{score}")
            except Exception as exc:
                raise RuleEngineError(
                    f"Rule {rule.name} failed: {exc}"
                ) from exc

        total_score = min(total_score, 100)
        level = _score_to_level(total_score)

        risk_score = RiskScore(
            transaction_id=transaction.transaction_id,
            score=total_score,
            level=level,
            factors=tuple(factors),
            timestamp=_now_iso(),
        )

        # Store transaction in history
        self._store_transaction(transaction)

        # Generate alert if threshold exceeded
        if total_score >= RISK_SCORE_THRESHOLD:
            self._create_alert(transaction, risk_score)

        return risk_score

    def _build_context(self, transaction: Transaction) -> dict[str, Any]:
        """Build evaluation context from transaction history."""
        sender_history = self._transaction_history.get(transaction.sender_id, [])
        recent = [
            t for t in sender_history
            if _is_recent(t.timestamp, hours=1)
        ]
        return {
            "velocity_count": len(recent),
            "recent_amounts": [t.amount for t in recent],
            "sender_history": sender_history,
        }

    def _store_transaction(self, transaction: Transaction) -> None:
        """Store transaction in sender's history."""
        if transaction.sender_id not in self._transaction_history:
            self._transaction_history[transaction.sender_id] = []
        self._transaction_history[transaction.sender_id].append(transaction)

    def _create_alert(self, transaction: Transaction, risk_score: RiskScore) -> FraudAlert:
        """Create a fraud alert for a high-risk transaction."""
        alert = FraudAlert(
            transaction_id=transaction.transaction_id,
            alert_type="HIGH_RISK_TRANSACTION",
            status=AlertStatus.OPEN,
            risk_score=risk_score.score,
            description=f"Risk score {risk_score.score}: {', '.join(risk_score.factors)}",
            created_at=_now_iso(),
        )
        self._alerts[alert.alert_id] = alert
        return alert

    def get_alert(self, alert_id: str) -> FraudAlert | None:
        """Retrieve an alert by ID."""
        return self._alerts.get(alert_id)

    def get_alerts_by_status(self, status: AlertStatus) -> list[FraudAlert]:
        """Retrieve all alerts with a given status."""
        return [a for a in self._alerts.values() if a.status == status]


# ---------------------------------------------------------------------------
# AML Compliance Engine
# ---------------------------------------------------------------------------

class AMLComplianceEngine:
    """
    Anti-Money Laundering compliance engine.

    Performs sanctions screening, PEP checks, transaction monitoring,
    and suspicious activity reporting.
    """

    def __init__(self) -> None:
        self._reports: dict[str, AMLReport] = {}
        self._pep_database: set[str] = set()
        self._sanctions_list: set[str] = set(SANCTIONED_ADDRESSES)

    def screen_transaction(self, transaction: Transaction) -> list[AMLReport]:
        """
        Run all AML checks on a transaction.

        Args:
            transaction: The transaction to screen.

        Returns:
            List of AMLReport for each check performed.

        Raises:
            AMLComplianceError: If screening fails.
        """
        reports: list[AMLReport] = []

        # Sanctions screening
        sanctions_report = self._check_sanctions(transaction)
        reports.append(sanctions_report)

        # PEP check
        pep_report = self._check_pep(transaction)
        reports.append(pep_report)

        # Transaction monitoring
        monitoring_report = self._monitor_transaction(transaction)
        reports.append(monitoring_report)

        for report in reports:
            self._reports[report.report_id] = report

        return reports

    def _check_sanctions(self, transaction: Transaction) -> AMLReport:
        """Screen transaction against sanctions lists."""
        sender = transaction.sender_id.lower()
        receiver = transaction.receiver_id.lower()
        hit = sender in self._sanctions_list or receiver in self._sanctions_list

        return AMLReport(
            transaction_id=transaction.transaction_id,
            check_type=AMLCheckType.SANCTIONS_SCREENING,
            passed=not hit,
            details="Sanctions list match" if hit else "No sanctions match",
            reported_at=_now_iso(),
        )

    def _check_pep(self, transaction: Transaction) -> AMLReport:
        """Check if sender/receiver is a Politically Exposed Person."""
        sender_hit = transaction.sender_id in self._pep_database
        receiver_hit = transaction.receiver_id in self._pep_database
        hit = sender_hit or receiver_hit

        return AMLReport(
            transaction_id=transaction.transaction_id,
            check_type=AMLCheckType.PEP_CHECK,
            passed=not hit,
            details="PEP match" if hit else "No PEP match",
            reported_at=_now_iso(),
        )

    def _monitor_transaction(self, transaction: Transaction) -> AMLReport:
        """Monitor transaction for suspicious patterns."""
        alerts: list[str] = []

        if transaction.amount > MAX_TRANSACTION_AMOUNT:
            alerts.append("Amount exceeds maximum threshold")

        if transaction.source_country != transaction.destination_country:
            if transaction.source_country in HIGH_RISK_COUNTRIES:
                alerts.append("High-risk cross-border transaction")
            if transaction.destination_country in HIGH_RISK_COUNTRIES:
                alerts.append("High-risk destination country")

        passed = len(alerts) == 0
        return AMLReport(
            transaction_id=transaction.transaction_id,
            check_type=AMLCheckType.TRANSACTION_MONITORING,
            passed=passed,
            details="; ".join(alerts) if alerts else "No suspicious patterns",
            reported_at=_now_iso(),
        )

    def add_to_pep_database(self, entity_id: str) -> None:
        """Add an entity to the PEP database."""
        self._pep_database.add(entity_id)

    def add_to_sanctions_list(self, address: str) -> None:
        """Add an address to the sanctions list."""
        self._sanctions_list.add(address.lower())

    def get_report(self, report_id: str) -> AMLReport | None:
        """Retrieve an AML report by ID."""
        return self._reports.get(report_id)


# ---------------------------------------------------------------------------
# Transaction Monitor
# ---------------------------------------------------------------------------

class TransactionMonitor:
    """
    Real-time transaction monitoring pipeline.

    Orchestrates fraud detection and AML compliance checks
    for each incoming transaction.
    """

    def __init__(
        self,
        fraud_engine: FraudDetectionEngine | None = None,
        aml_engine: AMLComplianceEngine | None = None,
    ) -> None:
        self._fraud_engine = fraud_engine or FraudDetectionEngine()
        self._aml_engine = aml_engine or AMLComplianceEngine()
        self._blocked_transactions: dict[str, Transaction] = {}

    def process_transaction(self, transaction: Transaction) -> dict[str, Any]:
        """
        Process a transaction through the full monitoring pipeline.

        Args:
            transaction: The transaction to process.

        Returns:
            Dictionary with risk_score, aml_reports, and decision.

        Raises:
            TransactionBlockedError: If the transaction is blocked.
        """
        # Fraud detection
        risk_score = self._fraud_engine.evaluate_transaction(transaction)

        # AML compliance
        aml_reports = self._aml_engine.screen_transaction(transaction)

        # Decision
        blocked = risk_score.score >= RISK_SCORE_THRESHOLD
        aml_failed = any(not r.passed for r in aml_reports)

        if blocked or aml_failed:
            self._blocked_transactions[transaction.transaction_id] = transaction
            raise TransactionBlockedError(
                f"Transaction {transaction.transaction_id} blocked: "
                f"risk_score={risk_score.score}, aml_failed={aml_failed}"
            )

        return {
            "transaction_id": transaction.transaction_id,
            "risk_score": risk_score,
            "aml_reports": aml_reports,
            "decision": "APPROVED",
        }

    def get_blocked_transactions(self) -> list[Transaction]:
        """Retrieve all blocked transactions."""
        return list(self._blocked_transactions.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _score_to_level(score: int) -> RiskLevel:
    """Convert numeric score to RiskLevel."""
    if score >= 80:
        return RiskLevel.CRITICAL
    elif score >= 60:
        return RiskLevel.HIGH
    elif score >= 30:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def _is_recent(timestamp: str, hours: int = 1) -> bool:
    """Check if a timestamp is within the recent window."""
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (now - ts).total_seconds() < hours * 3600
    except (ValueError, TypeError):
        return False


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
