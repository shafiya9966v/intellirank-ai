"""
api/main.py
───────────
FastAPI backend for the IntelliRank AI web UI.
Powers the sandbox demo and candidate detail pages.

Endpoints:
  GET  /api/health              — status check
  POST /api/rank                — run ranking on pre-built index
  GET  /api/candidates/{id}     — full score breakdown for one candidate
  GET  /api/export              — download submission.csv
"""
from __future__ import annotations
import sys, os, pickle, time, csv, io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
ARTIFACTS_DIR = ROOT / "artifacts"

from api.models import RankRequest, RankResponse, CandidateResult, HealthResponse, ScoreBreakdown
from api.middleware import configure_middleware
from api.security import validate_jd_text
from core.jd_parser import JD_REQUIREMENTS, parse_jd

app = FastAPI(
    title="IntelliRank AI",
    description="Intelligent Candidate Ranking — Redrob India Runs 2026",
    version="1.0.0",
)
configure_middleware(app)

# ── In-memory cache (loaded once on startup) ──────────────────────────────────
_cache: dict = {"metadata": None, "index": None, "results": []}


def _load_metadata():
    """Load metadata.pkl once and cache it."""
    if _cache["metadata"] is None:
        meta_path = ARTIFACTS_DIR / "metadata.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                _cache["metadata"] = {m["candidate_id"]: m for m in pickle.load(f)}
    return _cache["metadata"]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    artifacts_ready = all([
        (ARTIFACTS_DIR / "faiss.index").exists(),
        (ARTIFACTS_DIR / "metadata.pkl").exists(),
        (ARTIFACTS_DIR / "jd_embedding.npy").exists(),
    ])
    meta = _load_metadata()
    return HealthResponse(
        status="ready" if artifacts_ready else "needs_precomputation",
        artifacts_ready=artifacts_ready,
        indexed_candidates=len(meta) if meta else 0,
    )


@app.post("/api/rank", response_model=RankResponse)
async def rank_candidates(request: RankRequest, req: Request):
    """Run the full ranking pipeline on the pre-built FAISS index."""
    t0 = time.time()

    artifacts_ready = all([
        (ARTIFACTS_DIR / "faiss.index").exists(),
        (ARTIFACTS_DIR / "metadata.pkl").exists(),
    ])
    if not artifacts_ready:
        raise HTTPException(
            status_code=503,
            detail="Artifacts not ready. Run embed.py first."
        )

    jd_text = validate_jd_text(request.jd_text) if request.jd_text else None
    jd      = parse_jd(jd_text) if jd_text else JD_REQUIREMENTS

    try:
        import faiss
        import numpy as np

        index = faiss.read_index(str(ARTIFACTS_DIR / "faiss.index"))
        jd_emb = np.load(str(ARTIFACTS_DIR / "jd_embedding.npy")).astype(np.float32)
        with open(ARTIFACTS_DIR / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        k = min(2000, index.ntotal)
        scores, idxs = index.search(jd_emb.reshape(1, -1), k)

        from core.signals import compute_behavioral_multiplier
        from core.reasoning import generate_reasoning

        scored = []
        for faiss_idx, sem_score in zip(idxs[0], scores[0]):
            if faiss_idx < 0 or faiss_idx >= len(metadata):
                continue
            meta = metadata[faiss_idx]
            structured = float(meta.get("structured_score", 0.5))
            behavioral = float(meta.get("behavioral_mult", 0.5))
            hp_mult    = float(meta.get("honeypot_mult", 1.0))
            final = (float(sem_score) * 0.35 + structured * 0.65) * behavioral * hp_mult
            scored.append((max(0.001, min(1.0, final)), float(sem_score), meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_n = min(request.top_n, 100)
        top = scored[:top_n]

        results = []
        for rank, (final_sc, sem_sc, meta) in enumerate(top, 1):
            cid = meta["candidate_id"]
            ps  = meta.get("profile_snapshot", {})

            candidate_lite = {
                "profile":       {"current_title": ps.get("current_title"), "years_of_experience": ps.get("yoe"), "location": ps.get("location")},
                "skills":        [{"name": s, "endorsements": 0, "duration_months": 0} for s in meta.get("top_skills", [])],
                "career_history": [meta.get("recent_job", {})] if meta.get("recent_job") else [],
            }
            score_bd = {
                "structured_score":        meta.get("structured_score", 0.5),
                "tech_depth":              meta.get("tech_depth", 0.5),
                "career_trajectory":       meta.get("career_trajectory", 0.5),
                "company_fit":             meta.get("company_fit", 0.5),
                "disqualifier_penalty":    meta.get("disq_penalty", 0.0),
                "disqualifiers_triggered": meta.get("disqs", []),
                "tech_breakdown":          meta.get("tech_breakdown", {}),
            }
            sig_ft = meta.get("signal_features") or {}
            sig_ft["behavioral_multiplier"] = meta.get("behavioral_mult", 0.5)

            reasoning = generate_reasoning(candidate_lite, score_bd, sig_ft, rank, final_sc)

            results.append(CandidateResult(
                candidate_id=cid,
                rank=rank,
                score=round(final_sc, 4),
                reasoning=reasoning,
                title=ps.get("current_title"),
                yoe=ps.get("yoe"),
                location=ps.get("location"),
                top_skills=meta.get("top_skills", []),
                score_breakdown=ScoreBreakdown(
                    tech_depth=meta.get("tech_depth", 0),
                    career_trajectory=meta.get("career_trajectory", 0),
                    company_fit=meta.get("company_fit", 0),
                    location_fit=meta.get("location_fit", 0),
                    yoe_fit=meta.get("yoe_fit", 0),
                    education=meta.get("education", 0),
                    disqualifier_penalty=meta.get("disq_penalty", 0),
                    disqualifiers_triggered=meta.get("disqs", []),
                    behavioral_multiplier=meta.get("behavioral_mult", 0),
                    semantic_score=round(sem_sc, 4),
                    structured_score=meta.get("structured_score", 0),
                    final_score=round(final_sc, 4),
                ),
                signal_features=sig_ft,
            ))

        _cache["results"] = results
        elapsed = time.time() - t0
        return RankResponse(
            status="success",
            total_candidates_scored=len(scored),
            ranking_time_seconds=round(elapsed, 2),
            results=results,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")


@app.get("/api/candidates/{candidate_id}")
async def get_candidate(candidate_id: str):
    """Return full score breakdown for a specific candidate."""
    # Validate ID format
    import re
    if not re.match(r"^CAND_\d{7}$", candidate_id):
        raise HTTPException(status_code=400, detail="Invalid candidate_id format")

    meta = _load_metadata()
    if not meta or candidate_id not in meta:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found in index")

    return JSONResponse(content=meta[candidate_id])


@app.get("/api/export")
async def export_csv():
    """Download the submission CSV."""
    if not _cache["results"]:
        raise HTTPException(status_code=404, detail="No ranking results yet. Run /api/rank first.")

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in _cache["results"]:
        writer.writerow([r.candidate_id, r.rank, f"{r.score:.4f}", r.reasoning])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=submission.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
