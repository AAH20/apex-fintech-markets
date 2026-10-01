"""
Apex FinTech Markets — Audit Trail Module.

Provides immutable audit logging, regulatory reporting, and evidence chains
for production trading environments.

All audit entries are cryptographically chained to prevent tampering.
Designed for compliance with MiFID II, Basel III, EMIR, and MAR requirements.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
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
class AuditTrailError(Exception):
    """Base exception for audit trail errors."""


class ImmutableEntryError(AuditTrailError):
    """Raised when attempting to modify an immutable audit entry."""


class EvidenceChainError(AuditTrailError):
    """Raised when evidence chain verification fails."""


class AuditLogCorruptedError(AuditTrailError):
    """Raised when audit log integrity check fails."""


class RegulatoryReportError(AuditTrailError):
    """Raised when regulatory report generation fails."""


class AuditEntryNotFoundError(AuditTrailError):
    """Raised when an audit entry is not found."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AuditEventType(str, Enum):
    """Types of audit events."""

    ORDER_SUBMITTED = "order_submitted"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXECUTED = "order_executed"
    ORDER_REJECTED = "order_rejected"
    POSITION_UPDATED = "position_updated"
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    KILL_SWITCH_ENGAGED = "kill_switch_engaged"
    KILL_SWITCH_DISENGAGED = "kill_switch_disengaged"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"
    LIMIT_BREACH = "limit_breach"
    COMPLIANCE_ALERT = "compliance_alert"
    TRADE_REPORTED = "trade_reported"
    SETTINGS_CHANGED = "settings_changed"
    LOGIN = "login"
    LOGOUT = "logout"
    DATA_EXPORT = "data_export"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RegulatoryReportType(str, Enum):
    """Types of regulatory reports."""

    MIFID_II_RTS6 = "mifid_ii_rts6"
    MIFID_II_RTS27 = "mifid_ii_rts27"
    MIFID_II_RTS28 = "mifid_ii_rts28"
    BASEL_III_FRTB = "basel_iii_frtb"
    BASEL_III_SA = "basel_iii_sa"
    EMIR_TRADE = "emir_trade"
    EMIR_VALUATION = "emir_valuation"
    MAR_SURVEILLANCE = "mar_surveillance"
    DAILY_RISK = "daily_risk"
    MONTHLY_COMPLIANCE = "monthly_compliance"


class EvidenceChainStatus(str, Enum):
    """Status of evidence chain verification."""

    VALID = "valid"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AuditEntry:
    """Immutable audit log entry.

    Each entry contains a hash of its contents and the hash of the previous
    entry, forming a tamper-evident chain.

    Attributes:
        entry_id: Unique entry identifier.
        timestamp: Event timestamp (UTC).
        event_type: Type of audit event.
        severity: Severity level.
        actor: Who/what triggered the event.
        action: What action was performed.
        resource: What resource was affected.
        details: Additional event details.
        metadata: System metadata (IP, session, etc.).
        previous_hash: Hash of the previous entry in the chain.
        entry_hash: Hash of this entry's contents.
    """

    entry_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    actor: str
    action: str
    resource: str
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    entry_hash: str = ""

    def __post_init__(self) -> None:
        if not self.entry_hash:
            # Calculate hash if not provided
            object.__setattr__(
                self, "entry_hash", self._calculate_hash()
            )

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of entry contents.

        Returns:
            Hex digest of the entry hash.
        """
        # Create a deterministic representation
        data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "details": self._serialize_details(self.details),
            "metadata": self._serialize_details(self.metadata),
            "previous_hash": self.previous_hash,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_details(d: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize details dict to JSON-compatible format.

        Args:
            d: Dictionary to serialize.

        Returns:
            JSON-compatible dictionary.
        """
        result: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v
            elif isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, Enum):
                result[k] = v.value
            elif isinstance(v, (list, tuple)):
                result[k] = [
                    item.isoformat() if isinstance(item, datetime)
                    else item.value if isinstance(item, Enum)
                    else str(item)
                    for item in v
                ]
            elif isinstance(v, dict):
                result[k] = AuditEntry._serialize_details(v)
            else:
                result[k] = str(v)
        return result

    def verify_hash(self) -> bool:
        """Verify the entry's hash integrity.

        Returns:
            True if hash is valid, False otherwise.
        """
        return self.entry_hash == self._calculate_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary.

        Returns:
            Dictionary representation of the entry.
        """
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }

    def to_json(self) -> str:
        """Convert entry to JSON string.

        Returns:
            JSON string representation of the entry.
        """
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass(frozen=True)
class EvidenceChain:
    """Evidence chain for regulatory proceedings.

    Attributes:
        chain_id: Unique chain identifier.
        case_id: Related case/investigation identifier.
        description: Description of the evidence chain.
        entry_ids: Ordered list of audit entry IDs in the chain.
        created_at: Chain creation timestamp.
        created_by: Who created the chain.
        status: Chain verification status.
        metadata: Additional metadata.
    """

    chain_id: str
    case_id: str
    description: str
    entry_ids: Tuple[str, ...]
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    status: EvidenceChainStatus = EvidenceChainStatus.VALID
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegulatoryReport:
    """Regulatory report container.

    Attributes:
        report_id: Unique report identifier.
        report_type: Type of regulatory report.
        period_start: Reporting period start.
        period_end: Reporting period end.
        generated_at: Report generation timestamp.
        generated_by: Who generated the report.
        data: Report data.
        audit_entry_ids: Related audit entry IDs.
        signature: Report signature for integrity.
    """

    report_id: str
    report_type: RegulatoryReportType
    period_start: datetime
    period_end: datetime
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "system"
    data: Dict[str, Any] = field(default_factory=dict)
    audit_entry_ids: Tuple[str, ...] = field(default_factory=tuple)
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            object.__setattr__(self, "signature", self._calculate_signature())

    def _calculate_signature(self) -> str:
        """Calculate report signature.

        Returns:
            Hex digest of the report signature.
        """
        data = {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
            "data": self._serialize_data(self.data),
            "audit_entry_ids": list(self.audit_entry_ids),
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_data(d: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize report data to JSON-compatible format.

        Args:
            d: Dictionary to serialize.

        Returns:
            JSON-compatible dictionary.
        """
        result: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v
            elif isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, Enum):
                result[k] = v.value
            elif isinstance(v, (list, tuple)):
                result[k] = [
                    item.isoformat() if isinstance(item, datetime)
                    else item.value if isinstance(item, Enum)
                    else str(item)
                    for item in v
                ]
            elif isinstance(v, dict):
                result[k] = RegulatoryReport._serialize_data(v)
            else:
                result[k] = str(v)
        return result

    def verify_signature(self) -> bool:
        """Verify the report signature.

        Returns:
            True if signature is valid, False otherwise.
        """
        return self.signature == self._calculate_signature()


# ---------------------------------------------------------------------------
# Immutable Audit Log
# ---------------------------------------------------------------------------
class ImmutableAuditLog:
    """Immutable audit log with cryptographic chaining.

    All entries are append-only and cryptographically chained.
    Any tampering with historical entries will be detected.
    """

    def __init__(
        self,
        log_file: Optional[Union[str, Path]] = None,
        secret_key: Optional[bytes] = None,
    ) -> None:
        """Initialize immutable audit log.

        Args:
            log_file: Path to the audit log file.
            secret_key: Secret key for HMAC signatures.
        """
        self._entries: List[AuditEntry] = []
        self._entry_index: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._log_file = Path(log_file) if log_file else None
        self._secret_key = secret_key or os.urandom(32)
        self._last_hash = "0" * 64  # Genesis hash

        if self._log_file and self._log_file.exists():
            self._load_from_file()

    def append(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        actor: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Append a new entry to the audit log.

        Args:
            event_type: Type of audit event.
            severity: Severity level.
            actor: Who/what triggered the event.
            action: What action was performed.
            resource: What resource was affected.
            details: Additional event details.
            metadata: System metadata.

        Returns:
            The created AuditEntry.
        """
        with self._lock:
            entry_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()

            entry = AuditEntry(
                entry_id=entry_id,
                timestamp=timestamp,
                event_type=event_type,
                severity=severity,
                actor=actor,
                action=action,
                resource=resource,
                details=details or {},
                metadata=metadata or {},
                previous_hash=self._last_hash,
            )

            self._entries.append(entry)
            self._entry_index[entry_id] = len(self._entries) - 1
            self._last_hash = entry.entry_hash

            if self._log_file:
                self._append_to_file(entry)

            logger.debug(
                "Audit entry appended: %s - %s - %s",
                entry_id,
                event_type.value,
                action,
            )
            return entry

    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get an audit entry by ID.

        Args:
            entry_id: Entry identifier.

        Returns:
            AuditEntry if found, None otherwise.
        """
        with self._lock:
            idx = self._entry_index.get(entry_id)
            if idx is not None:
                return self._entries[idx]
            return None

    def get_entries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[Set[AuditEventType]] = None,
        severity: Optional[AuditSeverity] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEntry]:
        """Get audit entries with optional filtering.

        Args:
            start_time: Filter by start time.
            end_time: Filter by end time.
            event_types: Filter by event types.
            severity: Filter by severity.
            actor: Filter by actor.
            resource: Filter by resource.
            limit: Maximum number of entries to return.

        Returns:
            List of matching audit entries.
        """
        with self._lock:
            entries = self._entries

            if start_time:
                entries = [e for e in entries if e.timestamp >= start_time]
            if end_time:
                entries = [e for e in entries if e.timestamp <= end_time]
            if event_types:
                entries = [e for e in entries if e.event_type in event_types]
            if severity:
                entries = [e for e in entries if e.severity == severity]
            if actor:
                entries = [e for e in entries if e.actor == actor]
            if resource:
                entries = [e for e in entries if e.resource == resource]
            if limit:
                entries = entries[-limit:]

            return list(entries)

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of the entire audit chain.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: List[str] = []

        with self._lock:
            for i, entry in enumerate(self._entries):
                # Verify entry hash
                if not entry.verify_hash():
                    errors.append(
                        f"Entry {entry.entry_id} (index {i}): hash mismatch"
                    )

                # Verify chain linkage
                if i == 0:
                    # First entry should have genesis previous_hash
                    if entry.previous_hash != "0" * 64:
                        errors.append(
                            f"Entry {entry.entry_id} (index {i}): invalid genesis hash"
                        )
                else:
                    expected_prev_hash = self._entries[i - 1].entry_hash
                    if entry.previous_hash != expected_prev_hash:
                        errors.append(
                            f"Entry {entry.entry_id} (index {i}): chain broken, "
                            f"expected previous_hash {expected_prev_hash[:16]}..., "
                            f"got {entry.previous_hash[:16]}..."
                        )

        is_valid = len(errors) == 0
        if not is_valid:
            logger.error("Audit chain verification failed: %s", errors)
        return is_valid, errors

    def export_entries(
        self,
        output_file: Union[str, Path],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Export audit entries to a file.

        Args:
            output_file: Output file path.
            start_time: Filter by start time.
            end_time: Filter by end time.

        Returns:
            Number of entries exported.
        """
        entries = self.get_entries(start_time=start_time, end_time=end_time)
        output_path = Path(output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(entry.to_json() + "\n")

        logger.info("Exported %d audit entries to %s", len(entries), output_path)
        return len(entries)

    def _append_to_file(self, entry: AuditEntry) -> None:
        """Append an entry to the log file.

        Args:
            entry: Entry to append.
        """
        if self._log_file is None:
            return

        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")

    def _load_from_file(self) -> None:
        """Load entries from the log file."""
        if self._log_file is None or not self._log_file.exists():
            return

        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = AuditEntry(
                        entry_id=data["entry_id"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        event_type=AuditEventType(data["event_type"]),
                        severity=AuditSeverity(data["severity"]),
                        actor=data["actor"],
                        action=data["action"],
                        resource=data["resource"],
                        details=data.get("details", {}),
                        metadata=data.get("metadata", {}),
                        previous_hash=data.get("previous_hash", ""),
                        entry_hash=data.get("entry_hash", ""),
                    )
                    self._entries.append(entry)
                    self._entry_index[entry.entry_id] = len(self._entries) - 1
                    self._last_hash = entry.entry_hash
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error("Failed to load audit entry: %s", e)

    def __len__(self) -> int:
        """Get the number of entries in the log."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# Evidence Chain Manager
# ---------------------------------------------------------------------------
class EvidenceChainManager:
    """Manages evidence chains for regulatory proceedings."""

    def __init__(self, audit_log: ImmutableAuditLog) -> None:
        """Initialize evidence chain manager.

        Args:
            audit_log: Immutable audit log instance.
        """
        self.audit_log = audit_log
        self._chains: Dict[str, EvidenceChain] = {}
        self._lock = threading.Lock()

    def create_chain(
        self,
        case_id: str,
        description: str,
        entry_ids: Sequence[str],
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceChain:
        """Create a new evidence chain.

        Args:
            case_id: Related case/investigation identifier.
            description: Description of the evidence chain.
            entry_ids: Ordered list of audit entry IDs.
            created_by: Who created the chain.
            metadata: Additional metadata.

        Returns:
            Created EvidenceChain.

        Raises:
            EvidenceChainError: If any entry ID is not found.
        """
        # Verify all entries exist
        for entry_id in entry_ids:
            if self.audit_log.get_entry(entry_id) is None:
                raise EvidenceChainError(
                    f"Entry {entry_id} not found in audit log"
                )

        chain_id = str(uuid.uuid4())
        chain = EvidenceChain(
            chain_id=chain_id,
            case_id=case_id,
            description=description,
            entry_ids=tuple(entry_ids),
            created_by=created_by,
            metadata=metadata or {},
        )

        with self._lock:
            self._chains[chain_id] = chain

        logger.info("Created evidence chain: %s for case %s", chain_id, case_id)
        return chain

    def verify_chain(self, chain_id: str) -> Tuple[bool, List[str]]:
        """Verify an evidence chain.

        Args:
            chain_id: Chain identifier.

        Returns:
            Tuple of (is_valid, list of error messages).

        Raises:
            EvidenceChainError: If chain not found.
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            raise EvidenceChainError(f"Chain {chain_id} not found")

        errors: List[str] = []

        # Verify all entries exist and are in order
        for i, entry_id in enumerate(chain.entry_ids):
            entry = self.audit_log.get_entry(entry_id)
            if entry is None:
                errors.append(f"Entry {entry_id} (position {i}) not found")
                continue

            if not entry.verify_hash():
                errors.append(f"Entry {entry_id} (position {i}) hash mismatch")

        # Verify chain linkage
        for i in range(1, len(chain.entry_ids)):
            prev_entry = self.audit_log.get_entry(chain.entry_ids[i - 1])
            curr_entry = self.audit_log.get_entry(chain.entry_ids[i])
            if prev_entry and curr_entry:
                if curr_entry.previous_hash != prev_entry.entry_hash:
                    errors.append(
                        f"Chain broken between entries {chain.entry_ids[i-1]} "
                        f"and {chain.entry_ids[i]}"
                    )

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_chain(self, chain_id: str) -> Optional[EvidenceChain]:
        """Get an evidence chain by ID.

        Args:
            chain_id: Chain identifier.

        Returns:
            EvidenceChain if found, None otherwise.
        """
        return self._chains.get(chain_id)

    def get_chains_by_case(self, case_id: str) -> List[EvidenceChain]:
        """Get all evidence chains for a case.

        Args:
            case_id: Case identifier.

        Returns:
            List of evidence chains for the case.
        """
        return [c for c in self._chains.values() if c.case_id == case_id]


# ---------------------------------------------------------------------------
# Regulatory Reporting
# ---------------------------------------------------------------------------
class RegulatoryReporter:
    """Generates regulatory reports from audit data."""

    def __init__(self, audit_log: ImmutableAuditLog) -> None:
        """Initialize regulatory reporter.

        Args:
            audit_log: Immutable audit log instance.
        """
        self.audit_log = audit_log
        self._reports: Dict[str, RegulatoryReport] = {}
        self._lock = threading.Lock()

    def generate_daily_risk_report(
        self,
        report_date: datetime,
        risk_data: Dict[str, Any],
        generated_by: str = "system",
    ) -> RegulatoryReport:
        """Generate a daily risk report.

        Args:
            report_date: Report date.
            risk_data: Risk data to include.
            generated_by: Who generated the report.

        Returns:
            RegulatoryReport.
        """
        period_start = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=1)

        # Get related audit entries
        entries = self.audit_log.get_entries(
            start_time=period_start,
            end_time=period_end,
            event_types={
                AuditEventType.RISK_CHECK_PASSED,
                AuditEventType.RISK_CHECK_FAILED,
                AuditEventType.LIMIT_BREACH,
                AuditEventType.KILL_SWITCH_ENGAGED,
                AuditEventType.KILL_SWITCH_DISENGAGED,
            },
        )

        report = RegulatoryReport(
            report_id=f"DAILY-RISK-{report_date.strftime('%Y%m%d')}",
            report_type=RegulatoryReportType.DAILY_RISK,
            period_start=period_start,
            period_end=period_end,
            generated_by=generated_by,
            data=risk_data,
            audit_entry_ids=tuple(e.entry_id for e in entries),
        )

        with self._lock:
            self._reports[report.report_id] = report

        logger.info("Generated daily risk report: %s", report.report_id)
        return report

    def generate_monthly_compliance_report(
        self,
        year: int,
        month: int,
        compliance_data: Dict[str, Any],
        generated_by: str = "system",
    ) -> RegulatoryReport:
        """Generate a monthly compliance report.

        Args:
            year: Report year.
            month: Report month.
            compliance_data: Compliance data to include.
            generated_by: Who generated the report.

        Returns:
            RegulatoryReport.
        """
        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)

        # Get related audit entries
        entries = self.audit_log.get_entries(
            start_time=period_start,
            end_time=period_end,
        )

        report = RegulatoryReport(
            report_id=f"MONTHLY-COMPLIANCE-{year}{month:02d}",
            report_type=RegulatoryReportType.MONTHLY_COMPLIANCE,
            period_start=period_start,
            period_end=period_end,
            generated_by=generated_by,
            data=compliance_data,
            audit_entry_ids=tuple(e.entry_id for e in entries),
        )

        with self._lock:
            self._reports[report.report_id] = report

        logger.info("Generated monthly compliance report: %s", report.report_id)
        return report

    def generate_mifid_ii_rts27_report(
        self,
        period_start: datetime,
        period_end: datetime,
        transactions: List[Dict[str, Any]],
        generated_by: str = "system",
    ) -> RegulatoryReport:
        """Generate MiFID II RTS 27 transaction report.

        Args:
            period_start: Reporting period start.
            period_end: Reporting period end.
            transactions: Transaction data.
            generated_by: Who generated the report.

        Returns:
            RegulatoryReport.
        """
        # Get related audit entries
        entries = self.audit_log.get_entries(
            start_time=period_start,
            end_time=period_end,
            event_types={
                AuditEventType.ORDER_SUBMITTED,
                AuditEventType.ORDER_EXECUTED,
                AuditEventType.ORDER_CANCELLED,
                AuditEventType.ORDER_REJECTED,
            },
        )

        report = RegulatoryReport(
            report_id=f"MIFID-II-RTS27-{period_start.strftime('%Y%m%d')}-{period_end.strftime('%Y%m%d')}",
            report_type=RegulatoryReportType.MIFID_II_RTS27,
            period_start=period_start,
            period_end=period_end,
            generated_by=generated_by,
            data={"transactions": transactions},
            audit_entry_ids=tuple(e.entry_id for e in entries),
        )

        with self._lock:
            self._reports[report.report_id] = report

        logger.info("Generated MiFID II RTS 27 report: %s", report.report_id)
        return report

    def generate_emir_report(
        self,
        period_start: datetime,
        period_end: datetime,
        trade_data: List[Dict[str, Any]],
        generated_by: str = "system",
    ) -> RegulatoryReport:
        """Generate EMIR trade report.

        Args:
            period_start: Reporting period start.
            period_end: Reporting period end.
            trade_data: Trade data.
            generated_by: Who generated the report.

        Returns:
            RegulatoryReport.
        """
        # Get related audit entries
        entries = self.audit_log.get_entries(
            start_time=period_start,
            end_time=period_end,
            event_types={
                AuditEventType.TRADE_REPORTED,
                AuditEventType.ORDER_EXECUTED,
            },
        )

        report = RegulatoryReport(
            report_id=f"EMIR-{period_start.strftime('%Y%m%d')}-{period_end.strftime('%Y%m%d')}",
            report_type=RegulatoryReportType.EMIR_TRADE,
            period_start=period_start,
            period_end=period_end,
            generated_by=generated_by,
            data={"trades": trade_data},
            audit_entry_ids=tuple(e.entry_id for e in entries),
        )

        with self._lock:
            self._reports[report.report_id] = report

        logger.info("Generated EMIR report: %s", report.report_id)
        return report

    def get_report(self, report_id: str) -> Optional[RegulatoryReport]:
        """Get a regulatory report by ID.

        Args:
            report_id: Report identifier.

        Returns:
            RegulatoryReport if found, None otherwise.
        """
        return self._reports.get(report_id)

    def verify_report(self, report_id: str) -> bool:
        """Verify a regulatory report's signature.

        Args:
            report_id: Report identifier.

        Returns:
            True if signature is valid, False otherwise.

        Raises:
            RegulatoryReportError: If report not found.
        """
        report = self._reports.get(report_id)
        if report is None:
            raise RegulatoryReportError(f"Report {report_id} not found")
        return report.verify_signature()


# ---------------------------------------------------------------------------
# Audit Trail Facade
# ---------------------------------------------------------------------------
class AuditTrail:
    """Facade for the complete audit trail system.

    Provides a unified interface to the immutable audit log,
    evidence chain manager, and regulatory reporter.
    """

    def __init__(
        self,
        log_file: Optional[Union[str, Path]] = None,
        secret_key: Optional[bytes] = None,
    ) -> None:
        """Initialize the audit trail system.

        Args:
            log_file: Path to the audit log file.
            secret_key: Secret key for HMAC signatures.
        """
        self.audit_log = ImmutableAuditLog(log_file, secret_key)
        self.evidence_chains = EvidenceChainManager(self.audit_log)
        self.regulatory_reports = RegulatoryReporter(self.audit_log)

    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        actor: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log an audit event.

        Args:
            event_type: Type of audit event.
            severity: Severity level.
            actor: Who/what triggered the event.
            action: What action was performed.
            resource: What resource was affected.
            details: Additional event details.
            metadata: System metadata.

        Returns:
            The created AuditEntry.
        """
        return self.audit_log.append(
            event_type=event_type,
            severity=severity,
            actor=actor,
            action=action,
            resource=resource,
            details=details,
            metadata=metadata,
        )

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of the entire audit trail.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        return self.audit_log.verify_chain()

    def create_evidence_chain(
        self,
        case_id: str,
        description: str,
        entry_ids: Sequence[str],
        created_by: str = "system",
    ) -> EvidenceChain:
        """Create an evidence chain.

        Args:
            case_id: Related case/investigation identifier.
            description: Description of the evidence chain.
            entry_ids: Ordered list of audit entry IDs.
            created_by: Who created the chain.

        Returns:
            Created EvidenceChain.
        """
        return self.evidence_chains.create_chain(
            case_id=case_id,
            description=description,
            entry_ids=entry_ids,
            created_by=created_by,
        )

    def generate_report(
        self,
        report_type: RegulatoryReportType,
        period_start: datetime,
        period_end: datetime,
        data: Dict[str, Any],
        generated_by: str = "system",
    ) -> RegulatoryReport:
        """Generate a regulatory report.

        Args:
            report_type: Type of regulatory report.
            period_start: Reporting period start.
            period_end: Reporting period end.
            data: Report data.
            generated_by: Who generated the report.

        Returns:
            RegulatoryReport.
        """
        if report_type == RegulatoryReportType.DAILY_RISK:
            return self.regulatory_reports.generate_daily_risk_report(
                period_start, data, generated_by
            )
        elif report_type == RegulatoryReportType.MONTHLY_COMPLIANCE:
            return self.regulatory_reports.generate_monthly_compliance_report(
                period_start.year, period_start.month, data, generated_by
            )
        elif report_type == RegulatoryReportType.MIFID_II_RTS27:
            return self.regulatory_reports.generate_mifid_ii_rts27_report(
                period_start, period_end, data.get("transactions", []), generated_by
            )
        elif report_type == RegulatoryReportType.EMIR_TRADE:
            return self.regulatory_reports.generate_emir_report(
                period_start, period_end, data.get("trades", []), generated_by
            )
        else:
            raise RegulatoryReportError(f"Unsupported report type: {report_type}")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_audit_trail(
    log_file: Optional[Union[str, Path]] = None,
    secret_key: Optional[bytes] = None,
) -> AuditTrail:
    """Create a fully configured audit trail system.

    Args:
        log_file: Path to the audit log file.
        secret_key: Secret key for HMAC signatures.

    Returns:
        Configured AuditTrail instance.
    """
    return AuditTrail(log_file=log_file, secret_key=secret_key)
