"""Configurable risk thresholds and presentation helpers."""
from __future__ import annotations

import os

RISK_THRESHOLDS = {
    "low": float(os.getenv("RISK_LOW_THRESHOLD", "30")),
    "medium": float(os.getenv("RISK_MEDIUM_THRESHOLD", "60")),
}
MODEL_WEIGHT = float(os.getenv("RISK_MODEL_WEIGHT", "0.8"))
EVIDENCE_WEIGHT = float(os.getenv("RISK_EVIDENCE_WEIGHT", "0.2"))
WARNING_WEIGHTS = {
    "payment or fee language detected": 30.0,
    "sensitive personal or financial information requested": 25.0,
    "high income combined with no experience requirement": 15.0,
    "unusually attractive income claims detected": 10.0,
    "social-media job makes unusually broad earnings or experience claims": 12.0,
    "urgent hiring language detected": 8.0,
    "suspicious recruiter contact method detected": 5.0,
    "company information is missing or unverifiable (verify independently)": 3.0,
}
WARNING_GROUPS = {
    "payment or fee language detected": "payment",
    "sensitive personal or financial information requested": "sensitive_information",
    "high income combined with no experience requirement": "earnings",
    "unusually attractive income claims detected": "earnings",
    "social-media job makes unusually broad earnings or experience claims": "earnings",
    "urgent hiring language detected": "urgency",
    "suspicious recruiter contact method detected": "contact",
    "company information is missing or unverifiable (verify independently)": "company",
}
RISK_COPY = {
    "low": {
        "label": "Lower Estimated Risk — Still Verify",
        "explanation": "Fewer risk indicators were detected, but verify the employer and offer independently.",
    },
    "medium": {
        "label": "Caution — Verify Before Proceeding",
        "explanation": "The combined indicators warrant caution. Verify the employer, role, and payment or information requests before proceeding.",
    },
    "high": {
        "label": "High Risk — Strong Warning Signals",
        "explanation": "Multiple or strong risk indicators were detected. Do not send money or sensitive information until the posting is independently verified.",
    },
}


def adjusted_probability(model_probability: float, flags: list[str]) -> float:
    """Blend model probability with bounded, weighted warning evidence.

    The model remains the primary signal. Warning evidence can move the score
    by at most 20 percentage points, rather than replacing the model estimate.
    """
    model_score = max(0.0, min(100.0, float(model_probability)))
    grouped_scores: dict[str, float] = {}
    for flag in flags:
        key = flag.lower()
        group = WARNING_GROUPS.get(key)
        if group is None:
            continue
        grouped_scores[group] = max(grouped_scores.get(group, 0.0), WARNING_WEIGHTS[key])
    maximum_evidence = sum(
        max(WARNING_WEIGHTS[flag] for flag, group in WARNING_GROUPS.items() if group == name)
        for name in set(WARNING_GROUPS.values())
    )
    evidence_score = (
        min(100.0, (sum(grouped_scores.values()) / maximum_evidence) * 100.0)
        if maximum_evidence
        else 0.0
    )
    total_weight = MODEL_WEIGHT + EVIDENCE_WEIGHT
    if total_weight <= 0:
        raise ValueError("Risk weights must have a positive sum.")
    return ((MODEL_WEIGHT * model_score) + (EVIDENCE_WEIGHT * evidence_score)) / total_weight


def risk_level(probability: float) -> dict[str, str]:
    value = max(0.0, min(100.0, float(probability)))
    if value <= RISK_THRESHOLDS["low"]:
        risk_class = "low"
    elif value <= RISK_THRESHOLDS["medium"]:
        risk_class = "medium"
    else:
        risk_class = "high"
    return {
        "name": f"{risk_class.upper()} RISK",
        "class": risk_class,
        "label": RISK_COPY[risk_class]["label"],
        "explanation": RISK_COPY[risk_class]["explanation"],
    }
