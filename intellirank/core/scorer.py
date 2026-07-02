"""
core/scorer.py
──────────────
The 7-dimension structured scoring engine.

Every function takes a candidate dict + JDRequirements and returns float 0.0–1.0.
The final structured_score is a weighted composite of all 7 dimensions.

Formula:
  structured_score = (
      tech_depth     × 0.30 +
      career_traj    × 0.20 +
      company_fit    × 0.15 +
      location_fit   × 0.10 +
      yoe_fit        × 0.05 +
      education      × 0.05 +
      disq_penalty   (negative, capped at -0.30)
  )
"""
from __future__ import annotations
from typing import Any
from .jd_parser import JDRequirements, JD_REQUIREMENTS


# ── Constants ────────────────────────────────────────────────────────────────

PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}

COMPANY_SIZE_SCORE = {
    "1-10": 0.9,       # Startup — good for product ownership
    "11-50": 0.9,
    "51-200": 0.85,
    "201-500": 0.8,
    "501-1000": 0.7,
    "1001-5000": 0.6,
    "5001-10000": 0.5,
    "10001+": 0.4,     # Very large — usually more siloed
}

TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4}

TECH_FIELDS = {
    "computer science", "cs", "information technology", "software engineering",
    "machine learning", "artificial intelligence", "data science", "statistics",
    "mathematics", "electrical engineering", "electronics", "physics",
    "computational", "engineering",
}

SERVICES_AI_ONLY = [
    "langchain", "openai api", "chatgpt", "gpt api", "openai wrapper",
]


# ── Helper: build full text corpus from candidate ────────────────────────────

def _candidate_text(candidate: dict) -> str:
    """Concatenate all text fields for keyword matching."""
    parts = []
    p = candidate.get("profile", {})
    parts.append(p.get("headline") or "")
    parts.append(p.get("summary") or "")
    parts.append(p.get("current_title") or "")

    for job in candidate.get("career_history", []):
        parts.append(job.get("title") or "")
        parts.append(job.get("description") or "")

    for sk in candidate.get("skills", []):
        parts.append(sk.get("name") or "")

    for cert in candidate.get("certifications", []):
        if isinstance(cert, dict):
            parts.append(cert.get("name") or "")
        elif isinstance(cert, str):
            parts.append(cert)

    return " ".join(parts).lower()


# ── 1. Technical Depth Score ─────────────────────────────────────────────────

def score_technical_depth(
    candidate: dict[str, Any],
    jd: JDRequirements = JD_REQUIREMENTS,
) -> tuple[float, dict]:
    """
    Returns (score 0.0–1.0, breakdown_dict).

    Method:
    - For each skill category, check career text + skills array
    - Skills get credit weighted by proficiency × endorsement_boost
    - Skill assessment scores from Redrob assessments add verified bonus
    """
    full_text = _candidate_text(candidate)
    signals   = candidate.get("redrob_signals", {})
    assessed  = signals.get("skill_assessment_scores") or {}

    # Build a skills lookup for fast access
    skills_map: dict[str, dict] = {}
    for sk in candidate.get("skills", []):
        name = (sk.get("name") or "").lower().strip()
        if name:
            skills_map[name] = sk

    category_scores = {}
    for cat, keywords in jd.skill_categories.items():
        cat_weight = jd.category_weights.get(cat, 0.5)

        # Text match score (0 or 1 per keyword, max 1.0)
        text_hits = [kw for kw in keywords if kw in full_text]
        text_score = min(1.0, len(text_hits) / max(1, len(keywords) * 0.3))

        # Skill array score: proficiency × endorsement boost × duration
        skill_score = 0.0
        for kw in keywords:
            for sk_name, sk in skills_map.items():
                if kw in sk_name:
                    prof   = PROFICIENCY_WEIGHT.get(sk.get("proficiency", "beginner"), 0.2)
                    endorse_boost = min(1.3, 1.0 + sk.get("endorsements", 0) * 0.02)
                    dur_boost = min(1.2, 1.0 + sk.get("duration_months", 0) * 0.005)

                    # Edge case: expert + 0 endorsements + <6 months = suspicious
                    if (sk.get("proficiency") in ("expert", "advanced")
                            and sk.get("endorsements", 0) == 0
                            and sk.get("duration_months", 0) < 6):
                        prof *= 0.1  # Reduce weight for inflated claims

                    skill_score = max(skill_score, prof * endorse_boost * dur_boost)

        # Verified assessment bonus (Redrob platform tested)
        assess_bonus = 0.0
        for kw in keywords:
            for assess_name, assess_score in assessed.items():
                if kw in assess_name.lower() and isinstance(assess_score, (int, float)):
                    assess_bonus = max(assess_bonus, float(assess_score) / 100.0 * 0.2)

        combined = max(text_score, skill_score) + assess_bonus
        category_scores[cat] = min(1.0, combined) * cat_weight

    # Weighted average across categories
    total_weight = sum(jd.category_weights.values())
    raw_score = sum(category_scores.values()) / total_weight

    return round(min(1.0, raw_score), 4), category_scores


# ── 2. Career Trajectory Score ────────────────────────────────────────────────

def score_career_trajectory(candidate: dict[str, Any]) -> tuple[float, dict]:
    """
    Rewards: product company experience, increasing seniority, long tenures.
    Penalizes: pure services, frequent job-hopping, stagnant roles.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.2, {"reason": "no_career_history"}

    score = 0.5   # Neutral baseline

    # ── Tenure signal ─────────────────────────────────────────────────────────
    # Reward long tenures (24+ months), penalize short ones (<12 months)
    tenures = [j.get("duration_months", 0) or 0 for j in career]
    avg_tenure = sum(tenures) / len(tenures) if tenures else 0
    if avg_tenure >= 30:
        score += 0.15
    elif avg_tenure >= 20:
        score += 0.08
    elif avg_tenure < 12:
        score -= 0.10

    # ── Title seniority progression ───────────────────────────────────────────
    SENIOR_KEYWORDS = ["senior", "staff", "principal", "lead", "architect", "head", "director"]
    recent_jobs = sorted(career, key=lambda j: j.get("start_date") or "", reverse=True)[:3]
    has_senior_recent = any(
        any(kw in (j.get("title") or "").lower() for kw in SENIOR_KEYWORDS)
        for j in recent_jobs[:2]
    )
    if has_senior_recent:
        score += 0.10

    # ── Recent coding signal ──────────────────────────────────────────────────
    # Penalize if most recent roles are pure management/architecture (no coding)
    MANAGEMENT_ONLY = ["vp ", "vice president", "cto", "chief", "coo", "ceo", "evp", "svp"]
    most_recent = recent_jobs[0] if recent_jobs else {}
    recent_title = (most_recent.get("title") or "").lower()
    if any(m in recent_title for m in MANAGEMENT_ONLY):
        score -= 0.12

    # ── Production AI evidence in descriptions ────────────────────────────────
    PRODUCTION_SIGNALS = [
        "deployed", "production", "serving", "launched", "shipped",
        "real-time", "end-to-end", "users", "scale", "latency",
        "throughput", "api", "inference",
    ]
    production_hits = 0
    for job in career[:4]:  # Look at last 4 roles
        desc = (job.get("description") or "").lower()
        production_hits += sum(1 for sig in PRODUCTION_SIGNALS if sig in desc)

    if production_hits >= 8:
        score += 0.15
    elif production_hits >= 4:
        score += 0.08
    elif production_hits == 0:
        score -= 0.08

    return round(max(0.0, min(1.0, score)), 4), {
        "avg_tenure_months": round(avg_tenure, 1),
        "production_signal_hits": production_hits,
        "has_senior_recent": has_senior_recent,
    }


# ── 3. Company Type / Fit Score ───────────────────────────────────────────────

def score_company_fit(
    candidate: dict[str, Any],
    jd: JDRequirements = JD_REQUIREMENTS,
) -> tuple[float, dict]:
    """
    Best: AI/ML product companies, funded startups.
    Neutral: large diversified tech.
    Penalized: entire career at pure IT services.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.3, {}

    services_count = 0
    product_count  = 0
    ai_company_count = 0

    AI_KEYWORDS = [
        "ai", "ml", "data", "analytics", "intelligence", "tech",
        "software", "cloud", "platform", "saas",
    ]

    for job in career:
        company = (job.get("company") or "").lower()
        industry = (job.get("industry") or "").lower()
        size = job.get("company_size") or ""

        # Services company check
        is_services = any(s in company for s in jd.services_companies)
        if is_services:
            services_count += 1
        else:
            product_count += 1

        # AI company bonus
        if any(kw in company for kw in ["ai", "ml", "intelligence"]) or \
           any(kw in industry for kw in ["artificial intelligence", "machine learning", "data"]):
            ai_company_count += 1

    total = len(career)
    services_ratio = services_count / total
    product_ratio  = product_count / total

    # Base score
    score = 0.5
    if services_ratio >= 0.9:   # Entirely services
        score = 0.25
    elif services_ratio >= 0.6:  # Mostly services
        score = 0.38
    elif services_ratio <= 0.1:  # Mostly product
        score = 0.80
    elif services_ratio <= 0.3:
        score = 0.65

    # AI company bonus
    if ai_company_count >= 2:
        score = min(1.0, score + 0.15)
    elif ai_company_count == 1:
        score = min(1.0, score + 0.07)

    # Company size bonus (startups preferred for founding team role)
    current_size = candidate.get("profile", {}).get("current_company_size") or ""
    score += COMPANY_SIZE_SCORE.get(current_size, 0.5) * 0.1

    return round(max(0.0, min(1.0, score)), 4), {
        "services_ratio": round(services_ratio, 2),
        "product_ratio": round(product_ratio, 2),
        "ai_companies": ai_company_count,
    }


# ── 4. Location Fit ───────────────────────────────────────────────────────────

def score_location_fit(
    candidate: dict[str, Any],
    jd: JDRequirements = JD_REQUIREMENTS,
) -> float:
    """Returns 0.0–1.0. Uses both profile location and redrob signals."""
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})

    location = (profile.get("location") or "").lower()
    country  = (profile.get("country") or "").lower()
    relocate = bool(signals.get("willing_to_relocate", False))
    work_mode = (signals.get("preferred_work_mode") or "").lower()

    # Perfect: already in a preferred city
    if any(city in location for city in jd.preferred_locations):
        return 1.0

    # Good: in India and willing to relocate
    if "india" in country and relocate:
        return 0.85

    # Acceptable: in India but not preferred city, not stated willingness
    if "india" in country:
        return 0.65

    # Outside India but willing to relocate (strong signal)
    if relocate and work_mode in ("remote", "flexible"):
        return 0.45

    # Outside India, no relocation willingness
    return 0.15


# ── 5. YoE Fit ────────────────────────────────────────────────────────────────

def score_yoe_fit(
    candidate: dict[str, Any],
    jd: JDRequirements = JD_REQUIREMENTS,
) -> float:
    """Gaussian-like curve peaking at JD range (5-9 years)."""
    yoe = float(candidate.get("profile", {}).get("years_of_experience") or 0)

    if jd.yoe_min <= yoe <= jd.yoe_max:
        return 1.0
    elif 3 <= yoe < jd.yoe_min:
        return 0.70   # Promising, slightly junior
    elif jd.yoe_max < yoe <= 13:
        return 0.75   # Experienced, might want more senior title
    elif yoe <= 2:
        return 0.30   # Too junior
    else:
        return 0.45   # Too senior (>13 years)


# ── 6. Education Signal ───────────────────────────────────────────────────────

def score_education(candidate: dict[str, Any]) -> float:
    """Tier 1 = 1.0, Tier 4 = 0.4. CS/ML field gets bonus."""
    education = candidate.get("education", [])
    if not education:
        return 0.4   # Unknown = tier 4 default

    best_score = 0.0
    for edu in education:
        tier  = (edu.get("tier") or "tier_4").lower()
        field = (edu.get("field_of_study") or "").lower()

        tier_score  = TIER_SCORE.get(tier, 0.4)
        field_bonus = 0.0
        if any(tf in field for tf in TECH_FIELDS):
            field_bonus = 0.1

        best_score = max(best_score, min(1.0, tier_score + field_bonus))

    return round(best_score, 4)


# ── 7. Disqualifier Penalty ───────────────────────────────────────────────────

def compute_disqualifier_penalty(
    candidate: dict[str, Any],
    jd: JDRequirements = JD_REQUIREMENTS,
) -> tuple[float, list[str]]:
    """
    Returns (penalty 0.0–0.30, list_of_triggered_disqualifiers).
    Penalties are SUBTRACTED from structured_score.
    Total capped at 0.30 so score never goes negative.
    """
    penalties = []
    total_penalty = 0.0
    career  = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    full_text = _candidate_text(candidate)

    # ── D1: Entire career at services companies ───────────────────────────────
    if career:
        companies = [(j.get("company") or "").lower() for j in career]
        all_services = all(
            any(s in co for s in jd.services_companies)
            for co in companies
        )
        if all_services:
            penalties.append("Entire career at IT services companies (TCS/Wipro/Infosys etc.)")
            total_penalty += 0.25

    # ── D2: AI experience only LangChain/OpenAI wrappers ─────────────────────
    has_wrapper_only = any(kw in full_text for kw in SERVICES_AI_ONLY)
    has_pre_llm_ml   = any(kw in full_text for kw in [
        "recommendation", "ranking", "retrieval", "embedding", "faiss",
        "sklearn", "scikit", "xgboost", "feature engineering", "model training",
    ])
    if has_wrapper_only and not has_pre_llm_ml:
        penalties.append("AI experience limited to LangChain/OpenAI wrappers with no pre-LLM ML background")
        total_penalty += 0.15

    # ── D3: Title-hop disqualifier (3+ companies in 4 years) ─────────────────
    if len(career) >= 3:
        recent = sorted(career, key=lambda j: j.get("start_date") or "", reverse=True)[:4]
        recent_months = sum(j.get("duration_months", 0) or 0 for j in recent)
        if len(recent) >= 3 and recent_months <= 48:
            penalties.append(f"Frequent job-hopping: {len(recent)} roles in ~{recent_months} months")
            total_penalty += 0.10

    # ── D4: Wrong primary domain (CV/Speech/Robotics) ────────────────────────
    wrong_domain_hits = sum(1 for kw in jd.wrong_domains if kw in full_text)
    # Check if core skills/title also point to wrong domain
    title_lower = (profile.get("current_title") or "").lower()
    is_wrong_domain_title = any(kw in title_lower for kw in [
        "computer vision", "cv engineer", "speech", "robotics", "slam",
    ])
    if wrong_domain_hits >= 3 and is_wrong_domain_title:
        penalties.append("Primary expertise is CV/Speech/Robotics — not NLP/IR focus the role needs")
        total_penalty += 0.15

    # ── Cap total penalty ─────────────────────────────────────────────────────
    total_penalty = min(0.30, total_penalty)
    return round(total_penalty, 4), penalties


# ── Master: Compute Full Structured Score ─────────────────────────────────────

def compute_structured_score(
    candidate: dict[str, Any],
    jd: JDRequirements = JD_REQUIREMENTS,
) -> dict[str, Any]:
    """
    Computes all 7 dimensions and returns full breakdown dict.

    Returns:
        {
            "structured_score": float,      # Weighted composite, 0.0–1.0
            "tech_depth": float,
            "career_trajectory": float,
            "company_fit": float,
            "location_fit": float,
            "yoe_fit": float,
            "education": float,
            "disqualifier_penalty": float,
            "disqualifiers_triggered": list,
            "tech_breakdown": dict,
            "career_breakdown": dict,
            "company_breakdown": dict,
        }
    """
    tech_depth, tech_bd      = score_technical_depth(candidate, jd)
    career_traj, career_bd   = score_career_trajectory(candidate)
    company_fit, company_bd  = score_company_fit(candidate, jd)
    location_fit             = score_location_fit(candidate, jd)
    yoe_fit                  = score_yoe_fit(candidate, jd)
    education                = score_education(candidate)
    penalty, disqs           = compute_disqualifier_penalty(candidate, jd)

    # Weighted composite (weights sum to 0.85, penalty can reduce by up to 0.30)
    raw = (
        tech_depth  * 0.30 +
        career_traj * 0.20 +
        company_fit * 0.15 +
        location_fit * 0.10 +
        yoe_fit     * 0.05 +
        education   * 0.05 -
        penalty
    )

    structured_score = round(max(0.01, min(1.0, raw)), 4)

    return {
        "structured_score":        structured_score,
        "tech_depth":              tech_depth,
        "career_trajectory":       career_traj,
        "company_fit":             company_fit,
        "location_fit":            location_fit,
        "yoe_fit":                 yoe_fit,
        "education":               education,
        "disqualifier_penalty":    penalty,
        "disqualifiers_triggered": disqs,
        "tech_breakdown":          tech_bd,
        "career_breakdown":        career_bd,
        "company_breakdown":       company_bd,
    }
