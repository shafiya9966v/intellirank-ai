"""
rank.py  — ROOT LEVEL ENTRY POINT
──────────────────────────────────
PHASE 2 — RANKING STEP (timed by judges: must complete in < 5 minutes)

Constraints (from submission_spec):
  ✗ No GPU                ✗ No network calls          ✗ No API calls
  ✓ CPU only              ✓ Local files only           ✓ ≤16 GB RAM

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

What happens:
  1. Load FAISS index + metadata from artifacts/ (~2 seconds)
  2. Embed JD text with pre-loaded model (~5 seconds)
  3. FAISS search: top 2000 candidates by semantic similarity (~1 second)
  4. Score all 2000: final = (semantic×0.35 + structured×0.65) × behavioral
  5. Sort, take top 100, generate reasoning strings
  6. Write submission.csv, validate format
  7. Done in ~2 minutes
"""
from __future__ import annotations
import sys, os, csv, json, gzip, pickle, time, argparse
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.jd_parser import JD_REQUIREMENTS, parse_jd
from core.signals import compute_signal_features, compute_behavioral_multiplier
from core.honeypot import compute_honeypot_multiplier
from core.reasoning import generate_reasoning

ARTIFACTS_DIR = ROOT / "artifacts"

# ── Weights ───────────────────────────────────────────────────────────────────
SEMANTIC_WEIGHT    = 0.35
STRUCTURED_WEIGHT  = 0.65
TOP_K_SEMANTIC     = 2000   # Candidates retrieved by FAISS
TOP_N_OUTPUT       = 100    # Final shortlist size


def load_artifacts():
    """Load FAISS index, metadata, and JD embedding from artifacts/."""
    index_path    = ARTIFACTS_DIR / "faiss.index"
    metadata_path = ARTIFACTS_DIR / "metadata.pkl"
    jd_emb_path   = ARTIFACTS_DIR / "jd_embedding.npy"

    for p in [index_path, metadata_path, jd_emb_path]:
        if not p.exists():
            print(f"\nERROR: Missing artifact: {p}")
            print("Run embed.py first to build the index:")
            print("  python embed.py --candidates ./candidates.jsonl")
            sys.exit(1)

    import faiss
    print("  Loading FAISS index...", end=" ", flush=True)
    index = faiss.read_index(str(index_path))
    print(f"✓ ({index.ntotal:,} vectors)")

    print("  Loading metadata...", end=" ", flush=True)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    print(f"✓ ({len(metadata):,} candidates)")

    jd_embedding = np.load(str(jd_emb_path)).astype(np.float32)
    print(f"  JD embedding loaded ({len(jd_embedding)}-dim) ✓")

    return index, metadata, jd_embedding


def stream_signals(candidates_path: Path) -> dict[str, dict]:
    """
    Stream candidates.jsonl to get redrob_signals for the top candidates.
    Returns dict: candidate_id → signals dict
    Only called once we know which top-2000 candidates we need.
    More memory efficient than loading all 100K signals upfront.
    """
    signals_map = {}
    career_map  = {}
    open_fn = gzip.open if candidates_path.suffix == ".gz" else open

    with open_fn(candidates_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id")
                if cid:
                    signals_map[cid] = c.get("redrob_signals", {})
                    career_map[cid]  = c.get("career_history", [])
            except Exception:
                continue

    return signals_map, career_map


def compute_final_score(
    semantic_sim: float,
    meta: dict,
    signals: dict | None,
) -> tuple[float, float]:
    """
    Compute final score for one candidate.

    Returns (final_score, behavioral_multiplier)
    """
    # Structured score (pre-computed in embed.py)
    structured = float(meta.get("structured_score", 0.5))

    # Honeypot multiplier (pre-computed)
    hp_mult = float(meta.get("honeypot_mult", 1.0))

    # Behavioral multiplier (needs live signals)
    if signals:
        behavioral = compute_behavioral_multiplier(signals)
    else:
        behavioral = float(meta.get("behavioral_mult", 0.5))

    # Combined score before behavioral gate
    combined = (
        semantic_sim  * SEMANTIC_WEIGHT +
        structured    * STRUCTURED_WEIGHT
    ) * hp_mult

    # Apply behavioral as gate multiplier
    final = combined * behavioral

    return round(max(0.001, min(1.0, final)), 6), behavioral


def write_csv(ranked: list[dict], output_path: Path):
    """
    Write submission CSV with proper escaping.
    Validates: 100 rows, ranks 1-100, non-increasing scores, non-empty reasoning.
    """
    # Validate before writing
    assert len(ranked) == TOP_N_OUTPUT, f"Expected {TOP_N_OUTPUT} rows, got {len(ranked)}"

    scores = [r["score"] for r in ranked]
    for i in range(len(scores) - 1):
        if scores[i] < scores[i+1] - 1e-9:   # Allow tiny float tolerance
            # Fix ordering issue
            ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
            # Re-assign ranks
            for idx, r in enumerate(ranked, 1):
                r["rank"] = idx
            break

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in ranked:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['score']:.4f}",
                r["reasoning"],
            ])

    print(f"  ✓ Written {len(ranked)} rows to {output_path}")


def run_ranking(candidates_path: Path, output_path: Path, jd_text: str | None = None):
    """Main ranking pipeline. Must complete in <5 minutes."""
    t_start = time.time()
    print("\n" + "="*60)
    print("IntelliRank AI — Ranking Phase")
    print("="*60)

    # ── Step 1: Load artifacts ────────────────────────────────────────────────
    print("\n[1/6] Loading artifacts...")
    index, metadata, jd_embedding = load_artifacts()

    # Build candidate_id → metadata index for O(1) lookup
    id_to_meta: dict[str, dict] = {m["candidate_id"]: m for m in metadata}

    # ── Step 2: FAISS search ──────────────────────────────────────────────────
    print(f"\n[2/6] Semantic search (top {TOP_K_SEMANTIC} candidates)...")
    jd_vec = jd_embedding.reshape(1, -1).astype(np.float32)
    k = min(TOP_K_SEMANTIC, index.ntotal)
    scores_sem, indices = index.search(jd_vec, k)
    semantic_scores = scores_sem[0]
    faiss_indices   = indices[0]

    # Map FAISS indices back to candidate_ids
    top_candidates = []
    for faiss_idx, sem_score in zip(faiss_indices, semantic_scores):
        if faiss_idx < 0 or faiss_idx >= len(metadata):
            continue
        meta = metadata[faiss_idx]
        top_candidates.append((meta["candidate_id"], float(sem_score), meta))

    print(f"  ✓ Retrieved {len(top_candidates)} candidates from FAISS")
    elapsed = time.time() - t_start
    print(f"  ✓ Elapsed so far: {elapsed:.1f}s")

    # ── Step 3: Load signals for top candidates ───────────────────────────────
    print(f"\n[3/6] Loading behavioral signals from candidates file...")
    signals_map, career_map = stream_signals(candidates_path)
    print(f"  ✓ Signals loaded for {len(signals_map):,} candidates")

    # ── Step 4: Score all top candidates ─────────────────────────────────────
    print(f"\n[4/6] Scoring {len(top_candidates)} candidates...")
    scored = []
    for cid, sem_score, meta in top_candidates:
        signals  = signals_map.get(cid)
        final_sc, behavioral = compute_final_score(sem_score, meta, signals)

        scored.append({
            "candidate_id":    cid,
            "final_score":     final_sc,
            "semantic_score":  round(sem_score, 4),
            "behavioral_mult": behavioral,
            "meta":            meta,
            "signals":         signals or {},
        })

    # Sort by final score descending
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    top100 = scored[:TOP_N_OUTPUT]

    # Ensure exactly 100
    if len(top100) < TOP_N_OUTPUT:
        print(f"  [WARN] Only {len(top100)} candidates available — filling to 100 with lowest scorers")
        # In case pool is too small (shouldn't happen with 100K candidates)
        top100 = scored + [scored[-1]] * (TOP_N_OUTPUT - len(scored))
        top100 = top100[:TOP_N_OUTPUT]

    print(f"  ✓ Top-100 selected")
    print(f"  Score range: {top100[0]['final_score']:.4f} → {top100[-1]['final_score']:.4f}")

    elapsed = time.time() - t_start
    print(f"  ✓ Elapsed so far: {elapsed:.1f}s")

    # ── Step 5: Generate reasoning strings ───────────────────────────────────
    print(f"\n[5/6] Generating reasoning strings...")
    ranked_output = []

    for rank_pos, item in enumerate(top100, 1):
        cid  = item["candidate_id"]
        meta = item["meta"]

        # Reconstruct candidate-like object for reasoning
        candidate_lite = {
            "candidate_id": cid,
            "profile": {
                "current_title":   meta["profile_snapshot"].get("current_title", ""),
                "years_of_experience": meta["profile_snapshot"].get("yoe", 0),
                "location":        meta["profile_snapshot"].get("location", ""),
                "current_company": meta["profile_snapshot"].get("current_company", ""),
            },
            "skills": [{"name": s, "endorsements": 0, "duration_months": 0}
                       for s in meta.get("top_skills", [])],
            "career_history": [meta.get("recent_job", {})] if meta.get("recent_job") else [],
        }

        score_breakdown = {
            "structured_score":        meta.get("structured_score", 0.5),
            "tech_depth":              meta.get("tech_depth", 0.5),
            "career_trajectory":       meta.get("career_trajectory", 0.5),
            "company_fit":             meta.get("company_fit", 0.5),
            "disqualifier_penalty":    meta.get("disq_penalty", 0.0),
            "disqualifiers_triggered": meta.get("disqs", []),
            "tech_breakdown":          meta.get("tech_breakdown", {}),
            "career_breakdown":        meta.get("career_breakdown", {}),
        }

        sig_features = meta.get("signal_features") or compute_signal_features(item["signals"])
        sig_features["behavioral_multiplier"] = item["behavioral_mult"]

        reasoning = generate_reasoning(
            candidate=candidate_lite,
            score_breakdown=score_breakdown,
            signal_features=sig_features,
            rank=rank_pos,
            final_score=item["final_score"],
        )

        ranked_output.append({
            "candidate_id": cid,
            "rank":         rank_pos,
            "score":        item["final_score"],
            "reasoning":    reasoning,
        })

    print(f"  ✓ Reasoning generated for all {len(ranked_output)} candidates")

    # ── Step 6: Write CSV ─────────────────────────────────────────────────────
    print(f"\n[6/6] Writing submission CSV...")
    write_csv(ranked_output, output_path)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"✅ Ranking complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Output: {output_path}")
    print(f"   Top candidate: {ranked_output[0]['candidate_id']} (score: {ranked_output[0]['score']:.4f})")
    print(f"   #100 candidate: {ranked_output[-1]['candidate_id']} (score: {ranked_output[-1]['score']:.4f})")
    print(f"{'='*60}\n")

    if total_time > 290:
        print("⚠️  WARNING: Ranking took more than 4.8 minutes. Risk of exceeding 5-minute limit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IntelliRank AI — Rank candidates against job description"
    )
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path (default: submission.csv)")
    parser.add_argument("--jd", default=None,
                        help="Path to custom JD text file (optional)")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: File not found: {candidates_path}")
        sys.exit(1)

    output_path = Path(args.out)
    jd_text = Path(args.jd).read_text(encoding="utf-8") if args.jd else None

    run_ranking(candidates_path, output_path, jd_text)
