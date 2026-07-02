"""
tests/test_scorer.py
Unit tests for all scoring modules.
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.jd_parser import JD_REQUIREMENTS
from core.scorer import (
    score_technical_depth, score_career_trajectory,
    score_company_fit, score_location_fit, score_yoe_fit,
    score_education, compute_disqualifier_penalty,
    compute_structured_score,
)
from core.signals import compute_behavioral_multiplier, compute_signal_features
from core.honeypot import compute_honeypot_multiplier


# ── Test Fixtures ─────────────────────────────────────────────────────────────

def good_candidate():
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "current_title": "Senior ML Engineer",
            "headline": "Embedding and retrieval specialist",
            "summary": "6 years building production RAG systems with FAISS, Pinecone, semantic search. "
                       "Designed NDCG evaluation frameworks. Deployed to 1M+ users.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_company": "ProductAI",
            "current_company_size": "51-200",
            "current_industry": "Artificial Intelligence",
        },
        "career_history": [{
            "company": "ProductAI",
            "title": "Senior ML Engineer",
            "start_date": "2020-01-01",
            "end_date": None,
            "duration_months": 54,
            "is_current": True,
            "industry": "AI/ML",
            "company_size": "51-200",
            "description": "Built production embedding pipelines using sentence-transformers and FAISS. "
                           "Deployed semantic search retrieval serving 1M users at 50ms p99 latency. "
                           "Designed A/B tests and offline evaluation measuring NDCG@10 and MRR. "
                           "Fine-tuned BERT with LoRA for candidate ranking. Vector DB: Pinecone, Qdrant.",
        }],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 45, "duration_months": 72},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 12, "duration_months": 36},
            {"name": "Pinecone", "proficiency": "advanced", "endorsements": 8, "duration_months": 24},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 15, "duration_months": 40},
        ],
        "education": [{
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "tier": "tier_1",
        }],
        "certifications": [],
        "redrob_signals": {
            "open_to_work_flag": True,
            "last_active_date": "2026-06-20",
            "recruiter_response_rate": 0.85,
            "interview_completion_rate": 0.90,
            "offer_acceptance_rate": 0.5,
            "github_activity_score": 72.0,
            "notice_period_days": 30,
            "willing_to_relocate": True,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 40},
            "preferred_work_mode": "hybrid",
            "profile_completeness_score": 92,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "saved_by_recruiters_30d": 8,
            "profile_views_received_30d": 120,
            "avg_response_time_hours": 3.0,
            "connection_count": 450,
            "endorsements_received": 85,
            "applications_submitted_30d": 3,
            "search_appearance_30d": 45,
            "skill_assessment_scores": {"Python": 88, "Machine Learning": 91},
            "signup_date": "2023-01-15",
        }
    }


def services_only_candidate():
    c = good_candidate()
    c["candidate_id"] = "CAND_9999998"
    c["career_history"] = [
        {
            "company": "TCS", "title": "Developer",
            "start_date": "2018-01-01", "end_date": "2020-06-01",
            "duration_months": 29, "is_current": False,
            "industry": "IT Services", "company_size": "10001+",
            "description": "Maintained SAP modules. Worked on client-facing dashboards.",
        },
        {
            "company": "Infosys", "title": "Senior Analyst",
            "start_date": "2020-06-01", "end_date": None,
            "duration_months": 36, "is_current": True,
            "industry": "IT Services", "company_size": "10001+",
            "description": "Led a team maintaining legacy Java applications for banking client.",
        },
    ]
    return c


def ghost_candidate():
    c = good_candidate()
    c["candidate_id"] = "CAND_9999997"
    c["redrob_signals"]["last_active_date"] = "2025-09-01"
    c["redrob_signals"]["open_to_work_flag"] = False
    c["redrob_signals"]["recruiter_response_rate"] = 0.04
    c["redrob_signals"]["interview_completion_rate"] = 0.08
    return c


def honeypot_candidate():
    c = good_candidate()
    c["candidate_id"] = "CAND_9999996"
    c["profile"]["years_of_experience"] = 2.0    # Claims only 2 years
    c["career_history"] = [
        {
            "company": f"Company{i}", "title": "Engineer",
            "start_date": "2015-01-01", "end_date": None,
            "duration_months": 30,                    # 5 jobs × 30 months = 150 months total
            "is_current": i == 4,
            "industry": "Tech", "company_size": "51-200",
            "description": "Built ML systems.",
        }
        for i in range(5)
    ]
    c["skills"] = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 1}
        for i in range(6)
    ]
    return c


# ── Technical Depth Tests ─────────────────────────────────────────────────────

class TestTechnicalDepth:
    def test_good_candidate_scores_high(self):
        score, _ = score_technical_depth(good_candidate(), JD_REQUIREMENTS)
        assert score > 0.6, f"Good AI engineer should score >0.6, got {score}"

    def test_no_ai_skills_scores_low(self):
        c = good_candidate()
        c["skills"] = [{"name": "Microsoft Excel", "proficiency": "expert", "endorsements": 5, "duration_months": 60}]
        c["career_history"][0]["description"] = "Managed Excel spreadsheets and PowerPoint presentations."
        c["profile"]["summary"] = "Finance professional with 7 years of spreadsheet experience."
        score, _ = score_technical_depth(c, JD_REQUIREMENTS)
        assert score < 0.3, f"Non-AI candidate should score <0.3, got {score}"

    def test_inflated_expert_claim_gets_reduced_weight(self):
        c = good_candidate()
        # Expert with 0 endorsements and 1 month = suspicious
        c["skills"] = [
            {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 1},
        ]
        score_inflated, _ = score_technical_depth(c, JD_REQUIREMENTS)
        c2 = good_candidate()
        c2["skills"] = [
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 12, "duration_months": 30},
        ]
        score_real, _ = score_technical_depth(c2, JD_REQUIREMENTS)
        assert score_inflated < score_real, "Validated skills should score higher than inflated claims"

    def test_assessment_score_adds_bonus(self):
        c = good_candidate()
        c["redrob_signals"]["skill_assessment_scores"] = {"Python": 95, "machine learning": 98}
        score_with, _ = score_technical_depth(c, JD_REQUIREMENTS)
        c2 = good_candidate()
        c2["redrob_signals"]["skill_assessment_scores"] = {}
        score_without, _ = score_technical_depth(c2, JD_REQUIREMENTS)
        assert score_with >= score_without


# ── Career Trajectory Tests ────────────────────────────────────────────────────

class TestCareerTrajectory:
    def test_good_candidate_passes(self):
        score, _ = score_career_trajectory(good_candidate())
        assert score >= 0.5

    def test_no_career_returns_low_default(self):
        c = good_candidate()
        c["career_history"] = []
        score, bd = score_career_trajectory(c)
        assert score == 0.2
        assert "no_career_history" in bd.get("reason", "")

    def test_production_signals_boost_score(self):
        c_prod = good_candidate()
        score_prod, _ = score_career_trajectory(c_prod)
        c_no_prod = good_candidate()
        c_no_prod["career_history"][0]["description"] = "Attended meetings. Wrote design documents."
        score_no_prod, _ = score_career_trajectory(c_no_prod)
        assert score_prod > score_no_prod

    def test_management_only_gets_penalized(self):
        c = good_candidate()
        c["profile"]["current_title"] = "VP of Engineering"
        c["career_history"][0]["title"] = "VP of Engineering"
        score, _ = score_career_trajectory(c)
        assert score < 0.7


# ── Disqualifier Tests ─────────────────────────────────────────────────────────

class TestDisqualifiers:
    def test_services_only_triggers_penalty(self):
        penalty, disqs = compute_disqualifier_penalty(services_only_candidate(), JD_REQUIREMENTS)
        assert penalty >= 0.20
        assert len(disqs) >= 1

    def test_good_candidate_no_penalty(self):
        penalty, disqs = compute_disqualifier_penalty(good_candidate(), JD_REQUIREMENTS)
        assert penalty < 0.10
        assert len(disqs) == 0

    def test_penalty_capped_at_030(self):
        c = services_only_candidate()
        # Add more disqualifiers to try to push past the cap
        c["profile"]["summary"] = "Computer vision YOLO object detection specialist. No production code."
        c["profile"]["current_title"] = "Computer Vision Researcher"
        penalty, _ = compute_disqualifier_penalty(c, JD_REQUIREMENTS)
        assert penalty <= 0.30, "Penalty must never exceed 0.30"

    def test_title_hopper_gets_penalized(self):
        c = good_candidate()
        c["career_history"] = [
            {"company": f"Co{i}", "title": "Engineer",
             "start_date": f"202{i}-01-01", "end_date": f"202{i}-10-01",
             "duration_months": 9, "is_current": i == 3,
             "industry": "Tech", "company_size": "51-200",
             "description": "Worked on ML features."}
            for i in range(4)
        ]
        penalty, disqs = compute_disqualifier_penalty(c, JD_REQUIREMENTS)
        assert penalty >= 0.05


# ── Behavioral Multiplier Tests ────────────────────────────────────────────────

class TestBehavioralMultiplier:
    def test_active_candidate_high_multiplier(self):
        mult = compute_behavioral_multiplier(good_candidate()["redrob_signals"])
        assert mult > 0.6

    def test_ghost_candidate_low_multiplier(self):
        mult = compute_behavioral_multiplier(ghost_candidate()["redrob_signals"])
        assert mult < 0.25

    def test_multiplier_always_in_valid_range(self):
        for signals in [
            good_candidate()["redrob_signals"],
            ghost_candidate()["redrob_signals"],
            {},
            {"last_active_date": "2020-01-01", "open_to_work_flag": False,
             "recruiter_response_rate": 0.0, "interview_completion_rate": 0.0}
        ]:
            mult = compute_behavioral_multiplier(signals)
            assert 0.0 <= mult <= 1.0, f"Multiplier {mult} out of range"

    def test_offer_minus1_is_neutral(self):
        """offer_acceptance_rate = -1 (no history) must not penalize."""
        sigs = good_candidate()["redrob_signals"].copy()
        sigs["offer_acceptance_rate"] = -1
        mult_no_hist = compute_behavioral_multiplier(sigs)
        sigs["offer_acceptance_rate"] = 0.8
        mult_high = compute_behavioral_multiplier(sigs)
        assert abs(mult_no_hist - mult_high) < 0.15

    def test_github_minus1_is_neutral(self):
        """github_activity_score = -1 (no GitHub) must be neutral 0.5."""
        feats = compute_signal_features({"github_activity_score": -1, "last_active_date": "2026-06-20"})
        assert feats["github_score"] == 0.5

    def test_future_active_date_clamped(self):
        """last_active_date in the future should be treated as 0 days inactive."""
        sigs = good_candidate()["redrob_signals"].copy()
        sigs["last_active_date"] = "2027-01-01"  # Future date
        mult = compute_behavioral_multiplier(sigs)
        assert 0.0 <= mult <= 1.0  # Must not crash or go out of range


# ── Honeypot Detection Tests ───────────────────────────────────────────────────

class TestHoneypot:
    def test_normal_candidate_no_penalty(self):
        mult = compute_honeypot_multiplier(good_candidate())
        assert mult == 1.0

    def test_timeline_impossibility_penalized(self):
        mult = compute_honeypot_multiplier(honeypot_candidate())
        assert mult < 1.0

    def test_inflated_skills_penalized(self):
        c = good_candidate()
        c["skills"] = [
            {"name": f"SkillX{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 1}
            for i in range(6)
        ]
        mult = compute_honeypot_multiplier(c)
        assert mult < 1.0


# ── Location Tests ─────────────────────────────────────────────────────────────

class TestLocation:
    def test_pune_perfect_score(self):
        c = good_candidate()
        c["profile"]["location"] = "Pune, Maharashtra"
        assert score_location_fit(c, JD_REQUIREMENTS) == 1.0

    def test_noida_perfect_score(self):
        c = good_candidate()
        c["profile"]["location"] = "Noida, UP"
        assert score_location_fit(c, JD_REQUIREMENTS) == 1.0

    def test_india_with_relocate_high_score(self):
        c = good_candidate()
        c["profile"]["location"] = "Jaipur, Rajasthan"
        c["profile"]["country"] = "India"
        c["redrob_signals"]["willing_to_relocate"] = True
        score = score_location_fit(c, JD_REQUIREMENTS)
        assert score >= 0.80

    def test_outside_india_no_relocate_low(self):
        c = good_candidate()
        c["profile"]["location"] = "Toronto, Canada"
        c["profile"]["country"] = "Canada"
        c["redrob_signals"]["willing_to_relocate"] = False
        score = score_location_fit(c, JD_REQUIREMENTS)
        assert score < 0.25


# ── YoE Tests ──────────────────────────────────────────────────────────────────

class TestYoE:
    def test_sweet_spot(self):
        c = good_candidate()
        c["profile"]["years_of_experience"] = 7.0
        assert score_yoe_fit(c, JD_REQUIREMENTS) == 1.0

    def test_exact_boundaries(self):
        c = good_candidate()
        c["profile"]["years_of_experience"] = 5.0
        assert score_yoe_fit(c, JD_REQUIREMENTS) == 1.0
        c["profile"]["years_of_experience"] = 9.0
        assert score_yoe_fit(c, JD_REQUIREMENTS) == 1.0

    def test_junior(self):
        c = good_candidate()
        c["profile"]["years_of_experience"] = 1.5
        assert score_yoe_fit(c, JD_REQUIREMENTS) < 0.5

    def test_very_senior(self):
        c = good_candidate()
        c["profile"]["years_of_experience"] = 20.0
        assert score_yoe_fit(c, JD_REQUIREMENTS) < 0.6


# ── Edge Cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_completely_empty_candidate_doesnt_crash(self):
        """Most critical: never crash on missing data."""
        empty = {"candidate_id": "CAND_0000000"}
        result = compute_structured_score(empty, JD_REQUIREMENTS)
        assert "structured_score" in result
        assert result["structured_score"] > 0

    def test_none_values_dont_crash(self):
        c = good_candidate()
        c["profile"]["summary"] = None
        c["profile"]["current_title"] = None
        c["career_history"][0]["description"] = None
        result = compute_structured_score(c, JD_REQUIREMENTS)
        assert result["structured_score"] > 0

    def test_score_floor_never_negative(self):
        """Structured score must never go below 0.01."""
        worst = services_only_candidate()
        worst["profile"]["summary"] = "Computer vision YOLO robotics slam lidar"
        worst["profile"]["current_title"] = "Vision Researcher"
        result = compute_structured_score(worst, JD_REQUIREMENTS)
        assert result["structured_score"] >= 0.01

    def test_ghost_ranks_below_active_candidate(self):
        """A ghost with great skills must rank below an active mediocre candidate."""
        good_score = compute_structured_score(good_candidate(), JD_REQUIREMENTS)
        ghost_score = compute_structured_score(ghost_candidate(), JD_REQUIREMENTS)
        good_mult  = compute_behavioral_multiplier(good_candidate()["redrob_signals"])
        ghost_mult = compute_behavioral_multiplier(ghost_candidate()["redrob_signals"])
        good_final  = good_score["structured_score"]  * good_mult
        ghost_final = ghost_score["structured_score"] * ghost_mult
        assert good_final > ghost_final, \
            f"Ghost ({ghost_final:.4f}) should rank below active candidate ({good_final:.4f})"

    def test_services_ranks_below_product_company(self):
        """Services-only career must rank below equivalent product company career."""
        good_sc    = compute_structured_score(good_candidate(),    JD_REQUIREMENTS)["structured_score"]
        service_sc = compute_structured_score(services_only_candidate(), JD_REQUIREMENTS)["structured_score"]
        assert good_sc > service_sc

    def test_behavioral_multiplier_with_empty_signals(self):
        mult = compute_behavioral_multiplier({})
        assert 0.0 <= mult <= 1.0

    def test_education_missing_defaults_to_tier4(self):
        c = good_candidate()
        c["education"] = []
        score = score_education(c)
        assert score == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
