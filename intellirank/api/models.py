from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

class RankRequest(BaseModel):
    jd_text: Optional[str] = Field(None)
    top_n: int = Field(100, ge=10, le=100)

class ScoreBreakdown(BaseModel):
    tech_depth: float
    career_trajectory: float
    company_fit: float
    location_fit: float
    yoe_fit: float
    education: float
    disqualifier_penalty: float
    disqualifiers_triggered: list[str]
    behavioral_multiplier: float
    semantic_score: float
    structured_score: float
    final_score: float

class CandidateResult(BaseModel):
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    title: Optional[str] = None
    yoe: Optional[float] = None
    location: Optional[str] = None
    top_skills: list[str] = []
    score_breakdown: Optional[ScoreBreakdown] = None
    signal_features: Optional[dict] = None

class RankResponse(BaseModel):
    status: str
    total_candidates_scored: int
    ranking_time_seconds: float
    results: list[CandidateResult]

class HealthResponse(BaseModel):
    status: str
    artifacts_ready: bool
    indexed_candidates: int
