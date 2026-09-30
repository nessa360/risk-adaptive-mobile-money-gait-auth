"""Risk-adaptive decision engine from Assignment #1 Table 3.1."""
from __future__ import annotations
from .config import Decision, GaitConfidence, TransactionRisk

DEFAULT_POLICY = {
    GaitConfidence.HIGH: {
        TransactionRisk.LOW: Decision.ALLOW,
        TransactionRisk.MEDIUM: Decision.ALLOW,
        TransactionRisk.HIGH: Decision.STEP_UP_AUTHENTICATION,
    },
    GaitConfidence.MEDIUM: {
        TransactionRisk.LOW: Decision.ALLOW,
        TransactionRisk.MEDIUM: Decision.STEP_UP_AUTHENTICATION,
        TransactionRisk.HIGH: Decision.STEP_UP_AUTHENTICATION,
    },
    GaitConfidence.LOW: {
        TransactionRisk.LOW: Decision.STEP_UP_AUTHENTICATION,
        TransactionRisk.MEDIUM: Decision.STEP_UP_AUTHENTICATION,
        TransactionRisk.HIGH: Decision.DENY,
    },
}


def classify_transaction_risk(amount: float, recipient_is_new: bool = False, device_or_location_anomalous: bool = False, low_limit: float = 100.0, high_limit: float = 1000.0) -> TransactionRisk:
    """Classify risk from proposal-aligned transaction context inputs.

    Amount thresholds are configuration defaults for the prototype and are not
    intended as a production mobile-money risk model.
    """
    if amount < 0:
        raise ValueError("amount cannot be negative")
    risk_points = int(amount >= low_limit) + int(amount >= high_limit) + int(recipient_is_new) + int(device_or_location_anomalous)
    if risk_points >= 3 or amount >= high_limit:
        return TransactionRisk.HIGH
    if risk_points >= 1:
        return TransactionRisk.MEDIUM
    return TransactionRisk.LOW


def decide(gait_confidence: GaitConfidence | str, transaction_risk: TransactionRisk | str) -> Decision:
    """Return ALLOW, STEP_UP_AUTHENTICATION, or DENY."""
    gait = GaitConfidence(gait_confidence)
    risk = TransactionRisk(transaction_risk)
    return DEFAULT_POLICY[gait][risk]
