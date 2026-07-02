"""
core/honeypot.py
────────────────
Detects honeypot / fraudulent candidates through passive scoring deductions.
We do NOT hardcode specific candidate IDs. Instead we check for structural
inconsistencies that naturally push fake profiles to the bottom.
"""
from __future__ import annotations
from typing import Any


def compute_honeypot_multiplier(candidate: dict[str, Any]) -> float:
    """
    Returns a multiplier in [0.05, 1.0].
    1.0  = no suspicious signals
    0.05 = highly suspicious (almost certainly a honeypot)

    Applied multiplicatively to the structured score.
    """
    flags = 0
    severity = 0.0

    profile    = candidate.get("profile", {})
    career     = candidate.get("career_history", [])
    skills     = candidate.get("skills", [])

    yoe        = float(profile.get("years_of_experience", 0) or 0)
    total_career_months = sum(j.get("duration_months", 0) or 0 for j in career)

    # ── Check 1: Timeline impossibility ──────────────────────────────────────
    # Sum of all job durations > claimed YoE by >3 years = suspicious
    max_possible_months = yoe * 12 + 24   # allow 2-year buffer for overlaps
    if total_career_months > max_possible_months + 36:
        flags += 1
        severity += 0.4   # Strong signal

    # ── Check 2: Inflated skill claims ────────────────────────────────────────
    # "Expert" with 0 endorsements AND <6 months duration
    inflated = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("endorsements", 0) == 0
        and s.get("duration_months", 0) < 6
    )
    if inflated >= 3:
        flags += 1
        severity += 0.2

    # ── Check 3: Title ↔ description incoherence ─────────────────────────────
    # If a role title is completely non-technical but description is full of AI
    # keywords — likely shuffled/fake data
    TECH_DESC_KEYWORDS = [
        "machine learning", "neural network", "deep learning", "embedding",
        "vector", "model training", "data pipeline", "spark", "kafka",
        "recommendation", "ranking", "retrieval",
    ]
    NON_TECH_TITLES = [
        "accountant", "marketing manager", "sales", "hr manager",
        "content writer", "graphic designer", "operations manager",
        "finance manager", "procurement",
    ]
    incoherent_roles = 0
    for job in career:
        title_lower = (job.get("title") or "").lower()
        desc_lower  = (job.get("description") or "").lower()
        is_non_tech_title = any(nt in title_lower for nt in NON_TECH_TITLES)
        has_tech_desc     = sum(1 for kw in TECH_DESC_KEYWORDS if kw in desc_lower) >= 3
        if is_non_tech_title and has_tech_desc:
            incoherent_roles += 1

    if incoherent_roles >= 2:
        flags += 1
        severity += 0.3

    # ── Check 4: Company size contradiction ──────────────────────────────────
    BIG_COMPANIES = {"google", "microsoft", "amazon", "meta", "apple", "netflix"}
    for job in career:
        company_lower = (job.get("company") or "").lower()
        size          = job.get("company_size") or ""
        if any(big in company_lower for big in BIG_COMPANIES) and size in ("1-10", "11-50"):
            flags += 1
            severity += 0.15
            break

    # ── Check 5: Zero career history with high YoE ───────────────────────────
    if yoe > 8 and len(career) == 0:
        flags += 1
        severity += 0.3

    # ── Compute multiplier ────────────────────────────────────────────────────
    if flags == 0:
        return 1.0

    multiplier = max(0.05, 1.0 - severity)
    return round(multiplier, 4)


def get_honeypot_flags(candidate: dict[str, Any]) -> list[str]:
    """
    Returns human-readable list of honeypot signals found.
    Used by the reasoning generator and UI to explain flags.
    """
    flags = []
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])

    yoe = float(profile.get("years_of_experience", 0) or 0)
    total_months = sum(j.get("duration_months", 0) or 0 for j in career)

    if total_months > yoe * 12 + 60:
        flags.append(
            f"Timeline inconsistency: {total_months} months of work history "
            f"exceeds claimed {yoe} years experience"
        )

    inflated = [
        s["name"] for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("endorsements", 0) == 0
        and s.get("duration_months", 0) < 6
    ]
    if len(inflated) >= 3:
        flags.append(f"Inflated skill claims: {', '.join(inflated[:3])}")

    return flags
