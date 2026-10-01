"""
Apex FinTech Markets — Compliance Module.

Provides regulatory reporting and compliance checking for:
- MiFID II (RTS 6 — Algorithmic Trading, RTS 27 — Transaction Reporting,
  RTS 28 — Best Execution)
- Basel III (FRTB — Fundamental Review of the Trading Book, SA — Standardised Approach)
- EMIR (European Market Infrastructure Regulation)
- MAR (Market Abuse Regulation)

All functions are designed for production trading environments with full type safety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
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
class ComplianceError(Exception):
    """Base exception for compliance errors."""


class MiFIDIIReportError(ComplianceError):
    """Raised when MiFID II reporting fails validation."""


class BaselIIIError(ComplianceError):
    """Raised when Basel III calculations fail validation."""


class EMIRError(ComplianceError):
    """Raised when EMIR reporting fails validation."""


class MARError(ComplianceError):
    """Raised when MAR surveillance detects an issue."""


class InvalidTransactionError(ComplianceError):
    """Raised when a transaction fails compliance validation."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RTS6AlgorithmType(str, Enum):
    """RTS 6 algorithm classification."""

    MARKET_MAKING = "market_making"
    ARBITRAGE = "arbitrage"
    DIRECTIONAL = "directional"
    HIGH_FREQUENCY = "high_frequency"
    OTHER = "other"


class RTS27ReportType(str, Enum):
    """RTS 27 transaction report type."""

    TRANSACTION_REPORT = "transaction_report"
    CANCELLATION = "cancellation"
    AMENDMENT = "amendment"


class RTS28ExecutionVenue(str, Enum):
    """RTS 28 execution venue types."""

    REGULATED_MARKET = "regulated_market"
    MTF = "multilateral_trading_facility"
    OTF = "organised_trading_facility"
    SYSTEMATISER = "systematiser"
    INTERNALISER = "internaliser"


class FRTBAssetClass(str, Enum):
    """FRTB asset classes."""

    INTEREST_RATE = "interest_rate"
    CREDIT = "credit"
    EQUITY = "equity"
    COMMODITY = "commodity"
    FX = "fx"


class FRTBSensitivityClass(str, Enum):
    """FRTB sensitivity classes."""

    DELTA = "delta"
    VEGA = "vega"
    CURVATURE = "curvature"


class EMIRReportType(str, Enum):
    """EMIR report types."""

    TRADE_REPORT = "trade_report"
    VALUATION_REPORT = "valuation_report"
    COLLATERAL_REPORT = "collateral_report"
    RISK_REDUCTION = "risk_reduction"


class MARAlertType(str, Enum):
    """MAR surveillance alert types."""

    INSIDER_TRADING = "insider_trading"
    MARKET_MANIPULATION = "market_manipulation"
    SUSPICIOUS_ORDER = "suspicious_order"
    LAYERING = "layering"
    SPOOFING = "spoofing"
    WASH_TRADE = "wash_trade"


class MAROrderType(str, Enum):
    """MAR order classification."""

    BID = "bid"
    ASK = "ask"
    CANCEL = "cancel"
    AMEND = "amend"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Transaction:
    """Represents a financial transaction for regulatory reporting.

    Attributes:
        transaction_id: Unique transaction identifier.
        timestamp: Transaction timestamp (UTC).
        instrument_id: Instrument identifier (ISIN, etc.).
        instrument_type: Type of instrument (equity, bond, derivative, etc.).
        side: Buy or sell.
        quantity: Number of units.
        price: Execution price.
        currency: ISO 4217 currency code.
        venue: Execution venue.
        counterparty: Counterparty identifier.
        trader_id: Trader identifier.
        algorithm_id: Algorithm identifier (if algorithmic).
        client_id: Client identifier.
        notional: Notional value.
        maturity_date: Maturity date (for derivatives).
        asset_class: Asset class for FRTB.
    """

    transaction_id: str
    timestamp: datetime
    instrument_id: str
    instrument_type: str
    side: str
    quantity: Decimal
    price: Decimal
    currency: str
    venue: str
    counterparty: str
    trader_id: str
    algorithm_id: Optional[str] = None
    client_id: Optional[str] = None
    notional: Optional[Decimal] = None
    maturity_date: Optional[date] = None
    asset_class: Optional[FRTBAssetClass] = None

    @property
    def trade_value(self) -> Decimal:
        """Calculate trade value."""
        return self.quantity * self.price


@dataclass(frozen=True)
class RTS6AlgorithmReport:
    """RTS 6 algorithmic trading report.

    Attributes:
        algorithm_id: Unique algorithm identifier.
        algorithm_type: RTS 6 algorithm classification.
        description: Algorithm description.
        asset_classes: Asset classes traded.
        venues: Venues where algorithm operates.
        kill_switch_enabled: Whether kill switch is deployed.
        pre_trade_controls: Pre-trade risk controls in place.
        max_order_size: Maximum order size.
        max_daily_volume: Maximum daily volume.
        max_position: Maximum position limit.
        report_date: Report generation date.
    """

    algorithm_id: str
    algorithm_type: RTS6AlgorithmType
    description: str
    asset_classes: Tuple[str, ...]
    venues: Tuple[str, ...]
    kill_switch_enabled: bool
    pre_trade_controls: Tuple[str, ...]
    max_order_size: Decimal
    max_daily_volume: Decimal
    max_position: Decimal
    report_date: date = field(default_factory=date.today)


@dataclass(frozen=True)
class RTS27TransactionReport:
    """RTS 27 transaction report.

    Attributes:
        report_id: Unique report identifier.
        transaction: The transaction being reported.
        report_type: Type of report.
        execution_timestamp: Execution timestamp.
        publication_timestamp: Publication timestamp.
        is_algorithmic: Whether execution was algorithmic.
        algorithm_id: Algorithm identifier if applicable.
        report_date: Report generation date.
    """

    report_id: str
    transaction: Transaction
    report_type: RTS27ReportType
    execution_timestamp: datetime
    publication_timestamp: datetime
    is_algorithmic: bool
    algorithm_id: Optional[str] = None
    report_date: date = field(default_factory=date.today)


@dataclass(frozen=True)
class RTS28ExecutionQualityReport:
    """RTS 28 best execution quality report.

    Attributes:
        report_id: Unique report identifier.
        venue: Execution venue.
        instrument_id: Instrument identifier.
        instrument_type: Instrument type.
        execution_count: Number of executions.
        avg_execution_price: Average execution price.
        avg_market_price: Average market price at time of execution.
        slippage_bps: Average slippage in basis points.
        price_improvement_pct: Percentage of executions with price improvement.
        report_period_start: Start of reporting period.
        report_period_end: End of reporting period.
    """

    report_id: str
    venue: str
    instrument_id: str
    instrument_type: str
    execution_count: int
    avg_execution_price: Decimal
    avg_market_price: Decimal
    slippage_bps: Decimal
    price_improvement_pct: Decimal
    report_period_start: date
    report_period_end: date


@dataclass(frozen=True)
class FRTBSensitivity:
    """FRTB sensitivity measure.

    Attributes:
        asset_class: FRTB asset class.
        sensitivity_class: Sensitivity class (delta, vega, curvature).
        risk_factor: Risk factor name.
        sensitivity_value: Sensitivity value in reporting currency.
        bucket: Risk bucket (currency, tenor, etc.).
        correlation: Correlation with other risk factors.
    """

    asset_class: FRTBAssetClass
    sensitivity_class: FRTBSensitivityClass
    risk_factor: str
    sensitivity_value: Decimal
    bucket: str
    correlation: Optional[Decimal] = None


@dataclass(frozen=True)
class FRTBCapitalRequirement:
    """FRTB capital requirement result.

    Attributes:
        total_capital: Total capital requirement.
        delta_capital: Delta risk capital.
        vega_capital: Vega risk capital.
        curvature_capital: Curvature risk capital.
        default_risk_capital: Default risk capital (for non-securitisation).
        res_sec_capital: Residual securitisation capital.
        report_date: Report generation date.
    """

    total_capital: Decimal
    delta_capital: Decimal
    vega_capital: Decimal
    curvature_capital: Decimal
    default_risk_capital: Decimal
    res_sec_capital: Decimal
    report_date: date = field(default_factory=date.today)


@dataclass(frozen=True)
class BaselIIIStandardisedResult:
    """Basel III Standardised Approach result.

    Attributes:
        rwa: Risk-weighted assets.
        capital_requirement: Minimum capital requirement (8% of RWA).
        asset_class_breakdown: Breakdown by asset class.
        report_date: Report generation date.
    """

    rwa: Decimal
    capital_requirement: Decimal
    asset_class_breakdown: Dict[str, Decimal] = field(default_factory=dict)
    report_date: date = field(default_factory=date.today)


@dataclass(frozen=True)
class EMIRTradeReport:
    """EMIR trade report.

    Attributes:
        report_id: Unique report identifier.
        transaction: The transaction being reported.
        report_type: EMIR report type.
        counterparty_id: Counterparty LEI.
        clearing_obligation: Whether clearing is mandatory.
        clearing_status: Clearing status.
        collateral_amount: Collateral amount.
        valuation_amount: Valuation amount.
        report_date: Report generation date.
    """

    report_id: str
    transaction: Transaction
    report_type: EMIRReportType
    counterparty_id: str
    clearing_obligation: bool
    clearing_status: str
    collateral_amount: Optional[Decimal] = None
    valuation_amount: Optional[Decimal] = None
    report_date: date = field(default_factory=date.today)


@dataclass(frozen=True)
class MARAlert:
    """MAR surveillance alert.

    Attributes:
        alert_id: Unique alert identifier.
        alert_type: Type of MAR alert.
        severity: Severity level (1-5, 5 being highest).
        trader_id: Trader identifier.
        instrument_id: Instrument identifier.
        description: Alert description.
        timestamp: Alert generation timestamp.
        related_orders: Related order identifiers.
        status: Alert status (open, investigating, closed).
    """

    alert_id: str
    alert_type: MARAlertType
    severity: int
    trader_id: str
    instrument_id: str
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    related_orders: Tuple[str, ...] = field(default_factory=tuple)
    status: str = "open"


@dataclass(frozen=True)
class OrderEvent:
    """Order event for MAR surveillance.

    Attributes:
        order_id: Unique order identifier.
        timestamp: Event timestamp.
        trader_id: Trader identifier.
        instrument_id: Instrument identifier.
        order_type: Order type (bid, ask, cancel, amend).
        side: Buy or sell.
        quantity: Order quantity.
        price: Order price.
        venue: Order venue.
        is_cancelled: Whether the order was cancelled.
    """

    order_id: str
    timestamp: datetime
    trader_id: str
    instrument_id: str
    order_type: MAROrderType
    side: str
    quantity: Decimal
    price: Decimal
    venue: str
    is_cancelled: bool = False


# ---------------------------------------------------------------------------
# MiFID II Compliance
# ---------------------------------------------------------------------------
class MiFIDIICompliance:
    """MiFID II regulatory compliance checker and reporter.

    Covers RTS 6 (Algorithmic Trading), RTS 27 (Transaction Reporting),
    and RTS 28 (Best Execution).
    """

    def __init__(self) -> None:
        """Initialize MiFID II compliance module."""
        self._rts6_algorithms: Dict[str, RTS6AlgorithmReport] = {}
        self._rts27_reports: Dict[str, RTS27TransactionReport] = {}
        self._rts28_reports: Dict[str, RTS28ExecutionQualityReport] = {}

    # -- RTS 6: Algorithmic Trading ----------------------------------------

    def register_algorithm(self, report: RTS6AlgorithmReport) -> None:
        """Register an RTS 6 algorithm report.

        Args:
            report: RTS 6 algorithm report to register.

        Raises:
            MiFIDIIReportError: If algorithm is already registered.
        """
        if report.algorithm_id in self._rts6_algorithms:
            raise MiFIDIIReportError(
                f"Algorithm '{report.algorithm_id}' already registered."
            )
        self._validate_rts6(report)
        self._rts6_algorithms[report.algorithm_id] = report
        logger.info("Registered RTS 6 algorithm: %s", report.algorithm_id)

    def _validate_rts6(self, report: RTS6AlgorithmReport) -> None:
        """Validate RTS 6 algorithm report.

        Args:
            report: Report to validate.

        Raises:
            MiFIDIIReportError: If validation fails.
        """
        if not report.kill_switch_enabled:
            raise MiFIDIIReportError(
                f"Algorithm '{report.algorithm_id}' must have kill switch enabled."
            )
        if not report.pre_trade_controls:
            raise MiFIDIIReportError(
                f"Algorithm '{report.algorithm_id}' must have pre-trade controls."
            )
        if report.max_order_size <= 0:
            raise MiFIDIIReportError(
                f"Algorithm '{report.algorithm_id}' max_order_size must be positive."
            )

    def get_rts6_report(self, algorithm_id: str) -> Optional[RTS6AlgorithmReport]:
        """Get RTS 6 report for an algorithm.

        Args:
            algorithm_id: Algorithm identifier.

        Returns:
            RTS 6 report if found, None otherwise.
        """
        return self._rts6_algorithms.get(algorithm_id)

    # -- RTS 27: Transaction Reporting --------------------------------------

    def create_rts27_report(
        self,
        transaction: Transaction,
        report_type: RTS27ReportType = RTS27ReportType.TRANSACTION_REPORT,
        is_algorithmic: bool = False,
    ) -> RTS27TransactionReport:
        """Create an RTS 27 transaction report.

        Args:
            transaction: Transaction to report.
            report_type: Type of report.
            is_algorithmic: Whether execution was algorithmic.

        Returns:
            RTS 27 transaction report.

        Raises:
            MiFIDIIReportError: If transaction fails validation.
        """
        self._validate_transaction(transaction)
        report = RTS27TransactionReport(
            report_id=f"RTS27-{transaction.transaction_id}",
            transaction=transaction,
            report_type=report_type,
            execution_timestamp=transaction.timestamp,
            publication_timestamp=datetime.utcnow(),
            is_algorithmic=is_algorithmic,
            algorithm_id=transaction.algorithm_id,
        )
        self._rts27_reports[report.report_id] = report
        logger.info("Created RTS 27 report: %s", report.report_id)
        return report

    def _validate_transaction(self, transaction: Transaction) -> None:
        """Validate a transaction for RTS 27 reporting.

        Args:
            transaction: Transaction to validate.

        Raises:
            MiFIDIIReportError: If validation fails.
        """
        if transaction.quantity <= 0:
            raise MiFIDIIReportError(
                f"Transaction {transaction.transaction_id}: quantity must be positive."
            )
        if transaction.price <= 0:
            raise MiFIDIIReportError(
                f"Transaction {transaction.transaction_id}: price must be positive."
            )
        if not transaction.instrument_id:
            raise MiFIDIIReportError(
                f"Transaction {transaction.transaction_id}: instrument_id is required."
            )
        if not transaction.trader_id:
            raise MiFIDIIReportError(
                f"Transaction {transaction.transaction_id}: trader_id is required."
            )

    # -- RTS 28: Best Execution ---------------------------------------------

    def create_rts28_report(
        self,
        venue: str,
        instrument_id: str,
        instrument_type: str,
        executions: Sequence[Tuple[Decimal, Decimal]],
        period_start: date,
        period_end: date,
    ) -> RTS28ExecutionQualityReport:
        """Create an RTS 28 best execution quality report.

        Args:
            venue: Execution venue.
            instrument_id: Instrument identifier.
            instrument_type: Instrument type.
            executions: List of (execution_price, market_price) tuples.
            period_start: Start of reporting period.
            period_end: End of reporting period.

        Returns:
            RTS 28 execution quality report.

        Raises:
            MiFIDIIReportError: If no executions provided.
        """
        if not executions:
            raise MiFIDIIReportError("At least one execution is required for RTS 28.")

        execution_prices = [e[0] for e in executions]
        market_prices = [e[1] for e in executions]
        avg_exec = sum(execution_prices) / len(execution_prices)
        avg_market = sum(market_prices) / len(market_prices)

        slippages = []
        improvements = 0
        for exec_price, mkt_price in executions:
            slippage = (exec_price - mkt_price) / mkt_price * 10000  # bps
            slippages.append(slippage)
            if slippage < 0:
                improvements += 1

        avg_slippage = sum(slippages) / len(slippages)
        improvement_pct = Decimal(improvements) / Decimal(len(executions)) * 100

        report = RTS28ExecutionQualityReport(
            report_id=f"RTS28-{venue}-{instrument_id}-{period_start}",
            venue=venue,
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            execution_count=len(executions),
            avg_execution_price=avg_exec,
            avg_market_price=avg_market,
            slippage_bps=Decimal(str(avg_slippage)),
            price_improvement_pct=improvement_pct,
            report_period_start=period_start,
            report_period_end=period_end,
        )
        self._rts28_reports[report.report_id] = report
        logger.info("Created RTS 28 report: %s", report.report_id)
        return report


# ---------------------------------------------------------------------------
# Basel III Compliance
# ---------------------------------------------------------------------------
class BaselIIICompliance:
    """Basel III regulatory compliance calculator.

    Covers FRTB (Fundamental Review of the Trading Book) and
    SA (Standardised Approach).
    """

    # Standardised Approach risk weights by asset class
    SA_RISK_WEIGHTS: Dict[str, Decimal] = {
        "sovereign_0": Decimal("0.00"),
        "sovereign_20": Decimal("0.20"),
        "sovereign_50": Decimal("0.50"),
        "sovereign_100": Decimal("1.00"),
        "sovereign_150": Decimal("1.50"),
        "bank_20": Decimal("0.20"),
        "bank_50": Decimal("0.50"),
        "bank_100": Decimal("1.00"),
        "corporate_100": Decimal("1.00"),
        "corporate_150": Decimal("1.50"),
        "equity_100": Decimal("1.00"),
        "equity_250": Decimal("2.50"),
        "equity_400": Decimal("4.00"),
        "residential_mortgage": Decimal("0.35"),
        "commercial_real_estate": Decimal("1.00"),
        "commodities": Decimal("0.15"),
        "fx": Decimal("0.08"),
    }

    # FRTB risk weights by asset class and sensitivity class
    FRTB_RISK_WEIGHTS: Dict[FRTBAssetClass, Dict[FRTBSensitivityClass, Decimal]] = {
        FRTBAssetClass.INTEREST_RATE: {
            FRTBSensitivityClass.DELTA: Decimal("0.015"),
            FRTBSensitivityClass.VEGA: Decimal("0.55"),
            FRTBSensitivityClass.CURVATURE: Decimal("0.015"),
        },
        FRTBAssetClass.CREDIT: {
            FRTBSensitivityClass.DELTA: Decimal("0.05"),
            FRTBSensitivityClass.VEGA: Decimal("0.55"),
            FRTBSensitivityClass.CURVATURE: Decimal("0.05"),
        },
        FRTBAssetClass.EQUITY: {
            FRTBSensitivityClass.DELTA: Decimal("0.25"),
            FRTBSensitivityClass.VEGA: Decimal("0.55"),
            FRTBSensitivityClass.CURVATURE: Decimal("0.25"),
        },
        FRTBAssetClass.COMMODITY: {
            FRTBSensitivityClass.DELTA: Decimal("0.20"),
            FRTBSensitivityClass.VEGA: Decimal("0.55"),
            FRTBSensitivityClass.CURVATURE: Decimal("0.20"),
        },
        FRTBAssetClass.FX: {
            FRTBSensitivityClass.DELTA: Decimal("0.08"),
            FRTBSensitivityClass.VEGA: Decimal("0.55"),
            FRTBSensitivityClass.CURVATURE: Decimal("0.08"),
        },
    }

    def __init__(self) -> None:
        """Initialize Basel III compliance module."""
        self._frtb_sensitivities: List[FRTBSensitivity] = []

    # -- FRTB: Fundamental Review of the Trading Book ------------------------

    def add_frtb_sensitivity(self, sensitivity: FRTBSensitivity) -> None:
        """Add an FRTB sensitivity measure.

        Args:
            sensitivity: Sensitivity measure to add.
        """
        self._frtb_sensitivities.append(sensitivity)

    def calculate_frtb_capital(self) -> FRTBCapitalRequirement:
        """Calculate FRTB capital requirement.

        Returns:
            FRTBCapitalRequirement with capital breakdown.

        Raises:
            BaselIIIError: If no sensitivities are registered.
        """
        if not self._frtb_sensitivities:
            raise BaselIIIError("No FRTB sensitivities registered.")

        delta_capital = Decimal("0")
        vega_capital = Decimal("0")
        curvature_capital = Decimal("0")

        for sens in self._frtb_sensitivities:
            risk_weight = self.FRTB_RISK_WEIGHTS.get(sens.asset_class, {}).get(
                sens.sensitivity_class, Decimal("0")
            )
            capital_contribution = abs(sens.sensitivity_value) * risk_weight

            if sens.sensitivity_class == FRTBSensitivityClass.DELTA:
                delta_capital += capital_contribution
            elif sens.sensitivity_class == FRTBSensitivityClass.VEGA:
                vega_capital += capital_contribution
            elif sens.sensitivity_class == FRTBSensitivityClass.CURVATURE:
                curvature_capital += capital_contribution

        # Apply diversification benefit (simplified)
        diversification_factor = Decimal("0.75")
        total = (delta_capital + vega_capital + curvature_capital) * diversification_factor

        return FRTBCapitalRequirement(
            total_capital=total,
            delta_capital=delta_capital,
            vega_capital=vega_capital,
            curvature_capital=curvature_capital,
            default_risk_capital=Decimal("0"),
            res_sec_capital=Decimal("0"),
        )

    # -- SA: Standardised Approach -------------------------------------------

    def calculate_standardised_approach(
        self,
        exposures: Dict[str, Decimal],
    ) -> BaselIIIStandardisedResult:
        """Calculate Basel III Standardised Approach capital requirement.

        Args:
            exposures: Mapping of exposure category to exposure amount.

        Returns:
            BaselIIIStandardisedResult with RWA and capital requirement.
        """
        rwa = Decimal("0")
        breakdown: Dict[str, Decimal] = {}

        for category, amount in exposures.items():
            risk_weight = self.SA_RISK_WEIGHTS.get(category, Decimal("1.00"))
            asset_rwa = amount * risk_weight
            rwa += asset_rwa
            breakdown[category] = asset_rwa

        capital_requirement = rwa * Decimal("0.08")  # 8% minimum capital ratio

        return BaselIIIStandardisedResult(
            rwa=rwa,
            capital_requirement=capital_requirement,
            asset_class_breakdown=breakdown,
        )


# ---------------------------------------------------------------------------
# EMIR Compliance
# ---------------------------------------------------------------------------
class EMIRCompliance:
    """EMIR (European Market Infrastructure Regulation) compliance reporter."""

    def __init__(self) -> None:
        """Initialize EMIR compliance module."""
        self._reports: Dict[str, EMIRTradeReport] = {}

    def create_trade_report(
        self,
        transaction: Transaction,
        counterparty_id: str,
        clearing_obligation: bool = False,
        clearing_status: str = "pending",
        report_type: EMIRReportType = EMIRReportType.TRADE_REPORT,
    ) -> EMIRTradeReport:
        """Create an EMIR trade report.

        Args:
            transaction: Transaction to report.
            counterparty_id: Counterparty LEI.
            clearing_obligation: Whether clearing is mandatory.
            clearing_status: Clearing status.
            report_type: EMIR report type.

        Returns:
            EMIR trade report.

        Raises:
            EMIRError: If validation fails.
        """
        if not counterparty_id:
            raise EMIRError("Counterparty LEI is required for EMIR reporting.")

        report = EMIRTradeReport(
            report_id=f"EMIR-{transaction.transaction_id}",
            transaction=transaction,
            report_type=report_type,
            counterparty_id=counterparty_id,
            clearing_obligation=clearing_obligation,
            clearing_status=clearing_status,
            collateral_amount=transaction.notional,
            valuation_amount=transaction.trade_value,
        )
        self._reports[report.report_id] = report
        logger.info("Created EMIR report: %s", report.report_id)
        return report

    def get_report(self, report_id: str) -> Optional[EMIRTradeReport]:
        """Get an EMIR report by ID.

        Args:
            report_id: Report identifier.

        Returns:
            EMIR report if found, None otherwise.
        """
        return self._reports.get(report_id)


# ---------------------------------------------------------------------------
# MAR Compliance
# ---------------------------------------------------------------------------
class MARCompliance:
    """MAR (Market Abuse Regulation) surveillance and compliance module.

    Detects suspicious trading patterns including spoofing, layering,
    wash trades, and insider trading.
    """

    def __init__(self) -> None:
        """Initialize MAR compliance module."""
        self._alerts: Dict[str, MARAlert] = {}
        self._order_history: List[OrderEvent] = []
        self._alert_counter: int = 0

    def process_order_event(self, event: OrderEvent) -> Optional[MARAlert]:
        """Process an order event and check for market abuse patterns.

        Args:
            event: Order event to process.

        Returns:
            MARAlert if suspicious activity detected, None otherwise.
        """
        self._order_history.append(event)

        # Check for spoofing: large order followed by quick cancellation
        if event.is_cancelled and event.order_type in (MAROrderType.BID, MAROrderType.ASK):
            spoofing_alert = self._check_spoofing(event)
            if spoofing_alert:
                return spoofing_alert

        # Check for layering: multiple orders at different price levels
        layering_alert = self._check_layering(event)
        if layering_alert:
            return layering_alert

        # Check for wash trades: same trader on both sides
        wash_trade_alert = self._check_wash_trade(event)
        if wash_trade_alert:
            return wash_trade_alert

        return None

    def _check_spoofing(self, event: OrderEvent) -> Optional[MARAlert]:
        """Check for spoofing pattern.

        Spoofing: placing large orders with intent to cancel before execution
        to create false supply/demand impression.

        Args:
            event: Order event to check.

        Returns:
            MARAlert if spoofing detected, None otherwise.
        """
        # Look for large cancelled orders by same trader in short window
        recent_orders = [
            o for o in self._order_history
            if o.trader_id == event.trader_id
            and o.instrument_id == event.instrument_id
            and (event.timestamp - o.timestamp) < timedelta(minutes=5)
        ]

        large_cancelled = [
            o for o in recent_orders
            if o.is_cancelled and o.quantity > event.quantity * Decimal("2")
        ]

        if large_cancelled:
            self._alert_counter += 1
            alert = MARAlert(
                alert_id=f"MAR-{self._alert_counter:06d}",
                alert_type=MARAlertType.SPOOFING,
                severity=4,
                trader_id=event.trader_id,
                instrument_id=event.instrument_id,
                description=f"Potential spoofing: {len(large_cancelled)} large cancelled orders",
                related_orders=tuple(o.order_id for o in large_cancelled),
            )
            self._alerts[alert.alert_id] = alert
            logger.warning("MAR Alert: %s", alert.description)
            return alert

        return None

    def _check_layering(self, event: OrderEvent) -> Optional[MARAlert]:
        """Check for layering pattern.

        Layering: placing multiple orders at different price levels
        to create false market depth impression.

        Args:
            event: Order event to check.

        Returns:
            MARAlert if layering detected, None otherwise.
        """
        recent_orders = [
            o for o in self._order_history
            if o.trader_id == event.trader_id
            and o.instrument_id == event.instrument_id
            and (event.timestamp - o.timestamp) < timedelta(minutes=10)
            and not o.is_cancelled
        ]

        # Check for orders at 3+ distinct price levels on same side
        buy_prices = set()
        sell_prices = set()
        for o in recent_orders:
            if o.side.lower() == "buy":
                buy_prices.add(o.price)
            else:
                sell_prices.add(o.price)

        if len(buy_prices) >= 3 or len(sell_prices) >= 3:
            self._alert_counter += 1
            alert = MARAlert(
                alert_id=f"MAR-{self._alert_counter:06d}",
                alert_type=MARAlertType.LAYERING,
                severity=3,
                trader_id=event.trader_id,
                instrument_id=event.instrument_id,
                description=f"Potential layering: {len(buy_prices)} buy levels, {len(sell_prices)} sell levels",
                related_orders=tuple(o.order_id for o in recent_orders),
            )
            self._alerts[alert.alert_id] = alert
            logger.warning("MAR Alert: %s", alert.description)
            return alert

        return None

    def _check_wash_trade(self, event: OrderEvent) -> Optional[MARAlert]:
        """Check for wash trade pattern.

        Wash trade: same trader acting as both buyer and seller
        to create artificial volume.

        Args:
            event: Order event to check.

        Returns:
            MARAlert if wash trade detected, None otherwise.
        """
        recent_orders = [
            o for o in self._order_history
            if o.trader_id == event.trader_id
            and o.instrument_id == event.instrument_id
            and (event.timestamp - o.timestamp) < timedelta(minutes=30)
            and not o.is_cancelled
        ]

        has_buy = any(o.side.lower() == "buy" for o in recent_orders)
        has_sell = any(o.side.lower() == "sell" for o in recent_orders)

        if has_buy and has_sell:
            self._alert_counter += 1
            alert = MARAlert(
                alert_id=f"MAR-{self._alert_counter:06d}",
                alert_type=MARAlertType.WASH_TRADE,
                severity=5,
                trader_id=event.trader_id,
                instrument_id=event.instrument_id,
                description="Potential wash trade: same trader on both sides",
                related_orders=tuple(o.order_id for o in recent_orders),
            )
            self._alerts[alert.alert_id] = alert
            logger.warning("MAR Alert: %s", alert.description)
            return alert

        return None

    def get_alerts(
        self,
        status: Optional[str] = None,
        alert_type: Optional[MARAlertType] = None,
    ) -> List[MARAlert]:
        """Get MAR alerts with optional filtering.

        Args:
            status: Filter by status.
            alert_type: Filter by alert type.

        Returns:
            List of matching MAR alerts.
        """
        alerts = list(self._alerts.values())
        if status:
            alerts = [a for a in alerts if a.status == status]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        return alerts

    def update_alert_status(self, alert_id: str, status: str) -> None:
        """Update the status of a MAR alert.

        Args:
            alert_id: Alert identifier.
            status: New status.

        Raises:
            MARError: If alert not found.
        """
        if alert_id not in self._alerts:
            raise MARError(f"Alert '{alert_id}' not found.")
        old_alert = self._alerts[alert_id]
        self._alerts[alert_id] = MARAlert(
            alert_id=old_alert.alert_id,
            alert_type=old_alert.alert_type,
            severity=old_alert.severity,
            trader_id=old_alert.trader_id,
            instrument_id=old_alert.instrument_id,
            description=old_alert.description,
            timestamp=old_alert.timestamp,
            related_orders=old_alert.related_orders,
            status=status,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_compliance_suite() -> Dict[str, Any]:
    """Create a full compliance suite with all modules initialized.

    Returns:
        Dictionary with all compliance module instances.
    """
    return {
        "mifid_ii": MiFIDIICompliance(),
        "basel_iii": BaselIIICompliance(),
        "emir": EMIRCompliance(),
        "mar": MARCompliance(),
    }
