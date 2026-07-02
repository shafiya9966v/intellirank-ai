"""
core/reasoning.py
─────────────────
Generates specific, non-templated reasoning strings for each ranked candidate.

Rules (from submission_spec Stage 4 checks):
  1. Only mention facts actually in the candidate's profile — NO hallucination
  2. Must reference specific skills, role titles, or companies from their history
  3. Must acknowledge concerns where they exist
  4. Reasoning must MATCH rank — rank-5 ≠ same tone as rank-90
  5. Each reasoning must be meaningfully different
  6. Honest: admit weaknesses, not just strengths
"""
from __future__ import annotations
from typing import Any

# Keywords that mark a skill as directly relevant to the JD —
# used to prioritize which skills appear in reasoning text.
_JD_RELEVANT_SKILL_KEYWORDS = [
    "embedding", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "vector", "retrieval", "rag", "semantic search", "ranking",
    "recommend", "information retrieval", "sentence-transformer",
    "sentence transformer", "bge", "bert", "transformer", "llm",
    "fine-tun", "lora", "qlora", "peft", "ndcg", "mrr", "evaluation",
    "pytorch", "tensorflow", "python", "huggingface", "hugging face",
    "mlops", "mlflow", "elasticsearch", "opensearch",
]


def _rank_skills_by_relevance(skills: list[dict]) -> list[dict]:
    """
    Sort skills so JD-relevant ones (embeddings, vector DB, retrieval, LLM)
    come first, then fall back to endorsements/duration as tiebreaker.
    Prevents irrelevant high-endorsement skills (e.g. 'Marketing') from
    crowding out the skills that actually explain the candidate's fit.
    """
    def relevance_key(s):
        name_lower = (s.get("name") or "").lower()
        is_relevant = any(kw in name_lower for kw in _JD_RELEVANT_SKILL_KEYWORDS)
        return (
            1 if is_relevant else 0,
            s.get("endorsements", 0),
            s.get("duration_months", 0),
        )
    return sorted(skills, key=relevance_key, reverse=True)


def generate_reasoning(
    candidate: dict[str, Any],
    score_breakdown: dict[str, Any],
    signal_features: dict[str, Any],
    rank: int,
    final_score: float,
) -> str:
    """
    Build a 1–2 sentence reasoning string.
    Reads ONLY from candidate dict and pre-computed score breakdown.
    Never invents information.
    """
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])
    disqs    = score_breakdown.get("disqualifiers_triggered", [])

    title    = profile.get("current_title") or "Professional"
    yoe      = profile.get("years_of_experience") or 0
    location = profile.get("location") or "Unknown location"

    # Top 3 skills — JD-relevant skills (embeddings, vector DB, retrieval, LLM)
    # are prioritized first, falling back to endorsements/duration as tiebreaker.
    # This prevents irrelevant high-endorsement skills (e.g. "Marketing") from
    # crowding out the skills that actually explain the candidate's fit.
    top_skills = _rank_skills_by_relevance(skills)[:3]
    skill_names = [s["name"] for s in top_skills if s.get("name")]

    # Most recent company
    recent_jobs = sorted(career, key=lambda j: j.get("start_date") or "", reverse=True)
    recent_company = recent_jobs[0].get("company") if recent_jobs else None
    recent_title   = recent_jobs[0].get("title")   if recent_jobs else None

    # Tech signal: which core categories matched
    tech_bd = score_breakdown.get("tech_breakdown") or {}
    matched_cats = [k for k, v in tech_bd.items() if isinstance(v, float) and v > 0.3]

    # Behavioral flags
    days_inactive   = signal_features.get("days_inactive", 0)
    response_rate   = signal_features.get("response_rate", 0)
    notice_period   = signal_features.get("notice_period_days", 30)
    open_to_work    = signal_features.get("open_to_work", False)
    behavioral_mult = signal_features.get("behavioral_multiplier", 1.0)

    # ── Rank band → tone ──────────────────────────────────────────────────────
    if rank <= 10:
        return _top10_reasoning(
            title, yoe, location, skill_names, recent_company, recent_title,
            matched_cats, disqs, days_inactive, notice_period, open_to_work,
            score_breakdown, signal_features
        )
    elif rank <= 30:
        return _strong_reasoning(
            title, yoe, skill_names, recent_company, matched_cats, disqs,
            days_inactive, notice_period, behavioral_mult
        )
    elif rank <= 60:
        return _moderate_reasoning(
            title, yoe, skill_names, matched_cats, disqs,
            days_inactive, notice_period, final_score
        )
    else:
        return _weak_reasoning(
            title, yoe, skill_names, matched_cats, disqs,
            days_inactive, score_breakdown
        )


def _top10_reasoning(
    title, yoe, location, skills, company, job_title,
    matched_cats, disqs, days_inactive, notice, open_work, breakdown, sig_ft
) -> str:
    """Detailed, specific, positive-but-honest reasoning for top 10."""
    parts = []

    # Lead with their actual role + company
    if job_title and company:
        parts.append(f"{job_title} at {company} ({yoe:.0f}yr exp)")
    else:
        parts.append(f"{title} with {yoe:.0f} years experience")

    # Skill evidence
    if skills:
        parts.append(f"with hands-on {', '.join(skills[:2])}")

    # Technical category hits
    CAT_LABELS = {
        "embedding": "embedding/retrieval",
        "vector_db": "vector DB",
        "retrieval_ranking": "ranking/search",
        "llm": "LLM",
        "eval_frameworks": "evaluation frameworks",
    }
    cat_labels = [CAT_LABELS.get(c, c) for c in matched_cats[:2] if c in CAT_LABELS]
    if cat_labels:
        parts.append(f"strong {' & '.join(cat_labels)} experience")

    base = "; ".join(parts) + "."

    # Add concern if any
    concerns = []
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if days_inactive > 30:
        concerns.append(f"last active {days_inactive} days ago")
    if disqs:
        short_disq = disqs[0][:60] + "..." if len(disqs[0]) > 60 else disqs[0]
        concerns.append(short_disq.lower())

    if concerns:
        base += f" Note: {'; '.join(concerns)}."

    return base[:300]   # Hard cap — CSV field limit


def _strong_reasoning(
    title, yoe, skills, company, matched_cats, disqs,
    days_inactive, notice, behavioral_mult
) -> str:
    """Ranks 11–30: solid fit, one notable strength, one concern."""
    skill_str = f"{', '.join(skills[:2])}" if skills else "technical"

    if matched_cats:
        CAT_LABELS = {
            "embedding": "embedding", "vector_db": "vector DB",
            "retrieval_ranking": "search/ranking", "llm": "LLM work",
            "eval_frameworks": "evaluation"
        }
        cat = CAT_LABELS.get(matched_cats[0], matched_cats[0])
        strength = f"Strong {cat} background ({skill_str})"
    else:
        strength = f"{title} with {yoe:.0f}yr exp and {skill_str} skills"

    concern = ""
    if disqs:
        concern = f" Primary concern: {disqs[0][:80].lower()}."
    elif days_inactive > 60:
        concern = f" Inactive on platform for {days_inactive} days — verify availability."
    elif notice > 90:
        concern = f" Long notice period ({notice} days) may delay start."

    return (strength + "." + concern)[:300]


def _moderate_reasoning(
    title, yoe, skills, matched_cats, disqs, days_inactive, notice, final_score
) -> str:
    """Ranks 31–60: partial fit, clear limitation."""
    skill_str = f"{skills[0]}" if skills else "general tech"

    if matched_cats:
        base = f"Partial technical match — {skill_str} background with some {matched_cats[0]} exposure"
    else:
        base = f"{title} ({yoe:.0f}yr) with tangential relevance to role requirements"

    limitation = ""
    if disqs:
        limitation = f" Key gap: {disqs[0][:90].lower()}."
    elif not matched_cats:
        limitation = " Limited evidence of production embedding/retrieval system experience."

    return (base + "." + limitation)[:300]


def _weak_reasoning(
    title, yoe, skills, matched_cats, disqs, days_inactive, breakdown
) -> str:
    """Ranks 61–100: best available but clearly limited fit."""
    skill_str = f"{skills[0]}" if skills else "various"
    penalty = breakdown.get("disqualifier_penalty", 0)

    if disqs:
        base = f"Included as best available in pool; {disqs[0][:80].lower()}"
    elif days_inactive > 120:
        base = f"{title} with {skill_str} skills; inactive {days_inactive} days significantly limits availability score"
    elif not matched_cats:
        base = f"{title} ({yoe:.0f}yr) — limited direct overlap with JD core requirements (embedding, vector DB, ranking)"
    else:
        base = f"Borderline fit — {skill_str} background with {matched_cats[0]} exposure but significant gaps remain"

    return (base + ".")[:300]
