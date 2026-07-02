"""
core/signals.py
───────────────
Converts raw redrob_signals dict into a behavioral_multiplier float [0.05, 1.0].

The multiplier is applied MULTIPLICATIVELY to the profile score — not added.
A ghost candidate (inactive, never responds) gets multiplied by 0.05 regardless
of how good their profile looks. This is intentional: availability is a GATE.
"""
from __future__ import annotations
import math
from datetime import datetime, date
from typing import Any

TODAY = datetime(2026, 6, 27)   # Competition reference date


def _days_since_active(last_active_str: str | None) -> int:
    """Parse last_active_date and return days since then. Caps at 365."""
    if not last_active_str:
        return 90   # Unknown → treat as moderately inactive
    try:
        last = datetime.fromisoformat(str(last_active_str))
        days = max(0, (TODAY - last).days)   # Never negative
        return min(days, 365)
    except Exception:
        return 90


def compute_behavioral_multiplier(signals: dict[str, Any]) -> float:
    """
    Main function. Returns float in [0.05, 1.0].

    Components:
      open_to_work_score    : 1.0 if looking, 0.4 if passive
      recency_score         : linear decay over 180 days inactive
      response_quality      : recruiter_response_rate × interview_completion_rate
      availability_raw      : product of above three
      multiplier            : sqrt(availability_raw) to soften extremes
    """
    if not signals:
        return 0.5   # Missing signals → neutral

    # ── 1. Open to work ──────────────────────────────────────────────────────
    open_flag = signals.get("open_to_work_flag", False)
    open_score = 1.0 if open_flag else 0.4

    # ── 2. Recency ───────────────────────────────────────────────────────────
    days_inactive = _days_since_active(signals.get("last_active_date"))
    recency_score = max(0.05, 1.0 - (days_inactive / 180.0))

    # ── 3. Response quality ───────────────────────────────────────────────────
    response_rate = float(signals.get("recruiter_response_rate") or 0.5)
    response_rate = max(0.0, min(1.0, response_rate))

    interview_rate = float(signals.get("interview_completion_rate") or 0.5)
    # Edge case: 0% completion with < 3 interviews = insufficient data → neutral
    # We approximate this by checking if response_rate is also very low
    if interview_rate == 0.0 and response_rate > 0.3:
        interview_rate = 0.5   # Probably just hasn't had many interviews
    interview_rate = max(0.0, min(1.0, interview_rate))

    response_quality = (response_rate * 0.6) + (interview_rate * 0.4)

    # ── 4. Raw availability ───────────────────────────────────────────────────
    availability_raw = open_score * recency_score * response_quality

    # ── 5. Soften with sqrt ───────────────────────────────────────────────────
    multiplier = math.sqrt(max(0.0025, availability_raw))  # min sqrt = 0.05

    return round(max(0.05, min(1.0, multiplier)), 4)


def compute_signal_features(signals: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a rich feature dict used by the UI score breakdown panel
    and the reasoning generator.
    """
    if not signals:
        return {}

    days_inactive = _days_since_active(signals.get("last_active_date"))

    # Salary fit (35-38 LPA target)
    salary = signals.get("expected_salary_range_inr_lpa") or {}
    sal_min = float(salary.get("min", 0) or 0)
    sal_max = float(salary.get("max", 0) or 0)
    salary_midpoint = (sal_min + sal_max) / 2 if sal_max > 0 else 0
    salary_fit = "unknown"
    if salary_midpoint > 0:
        if 25 <= salary_midpoint <= 50:
            salary_fit = "good"
        elif salary_midpoint > 60:
            salary_fit = "too_high"
        else:
            salary_fit = "too_low"

    # GitHub score — -1 means no GitHub (neutral)
    github_raw = signals.get("github_activity_score", -1)
    github_score = 0.5 if github_raw == -1 else float(github_raw) / 100.0

    # Offer acceptance — -1 means no prior offers (neutral)
    offer_raw = signals.get("offer_acceptance_rate", -1)
    offer_score = 0.5 if offer_raw == -1 else float(offer_raw)

    # Profile completeness (0-100 → 0.0-1.0)
    completeness = float(signals.get("profile_completeness_score") or 50) / 100.0

    # Market demand signals
    saved_30d = int(signals.get("saved_by_recruiters_30d") or 0)
    views_30d = int(signals.get("profile_views_received_30d") or 0)
    demand_score = min(1.0, (saved_30d * 0.1 + views_30d * 0.01))

    return {
        "days_inactive": days_inactive,
        "open_to_work": bool(signals.get("open_to_work_flag", False)),
        "response_rate": float(signals.get("recruiter_response_rate") or 0),
        "interview_rate": float(signals.get("interview_completion_rate") or 0),
        "notice_period_days": int(signals.get("notice_period_days") or 30),
        "github_score": github_score,
        "offer_score": offer_score,
        "salary_fit": salary_fit,
        "salary_midpoint_lpa": round(salary_midpoint, 1),
        "profile_completeness": round(completeness, 2),
        "market_demand": round(demand_score, 2),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "preferred_work_mode": signals.get("preferred_work_mode", "unknown"),
        "verified": bool(signals.get("verified_email") and signals.get("verified_phone")),
        "behavioral_multiplier": compute_behavioral_multiplier(signals),
    }
