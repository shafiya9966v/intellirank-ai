"""
core/embed.py  (also callable as: python embed.py)
───────────────
PHASE 1 — PRE-COMPUTATION (runs once offline, not timed by judges)

What this does:
  1. Streams candidates.jsonl line-by-line (never loads full 465MB into RAM)
  2. Builds a text corpus per candidate (title + summary + career descriptions)
  3. Embeds each candidate using BGE-small-en-v1.5 (33MB, CPU-efficient)
  4. Builds a FAISS IndexFlatIP (exact cosine search)
  5. Pre-computes all structured scores (tech_depth, company_fit, etc.)
  6. Saves faiss.index + metadata.pkl to artifacts/

Runtime: ~20-40 minutes on CPU for 100K candidates (allowed — this is pre-computation)
Memory:  ~4GB peak (batched processing, 512 candidates per batch)
"""
from __future__ import annotations
import sys, os, json, gzip, pickle, time, argparse
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.jd_parser import JD_REQUIREMENTS, parse_jd
from core.scorer import compute_structured_score
from core.signals import compute_signal_features, compute_behavioral_multiplier
from core.honeypot import compute_honeypot_multiplier

ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 512
MAX_TEXT_CHARS = 2000   # Truncate very long summaries for memory


# ── Candidate text builder ────────────────────────────────────────────────────

def build_corpus_text(candidate: dict) -> str:
    """
    Build a rich text representation of the candidate for embedding.
    Combines: title, headline, summary, and all career descriptions.
    Truncated to MAX_TEXT_CHARS to keep memory manageable.
    """
    p = candidate.get("profile", {})
    parts = [
        p.get("current_title") or "",
        p.get("headline") or "",
        (p.get("summary") or "")[:800],   # Summary gets the most space
    ]

    # Add career descriptions (most recent 5 jobs)
    career = sorted(
        candidate.get("career_history", []),
        key=lambda j: j.get("start_date") or "",
        reverse=True
    )[:5]
    for job in career:
        title = job.get("title") or ""
        desc  = (job.get("description") or "")[:200]
        parts.append(f"{title}: {desc}")

    # Add top skill names
    skills = [s.get("name", "") for s in candidate.get("skills", [])[:15]]
    parts.append(" ".join(skills))

    full = " ".join(p for p in parts if p).strip()
    return full[:MAX_TEXT_CHARS]


# ── JSONL streaming ───────────────────────────────────────────────────────────

def stream_candidates(filepath: Path) -> Iterator[dict]:
    """
    Stream candidates.jsonl line by line. Handles both plain and gzipped files.
    Skips and logs malformed lines — never crashes.
    """
    open_fn = gzip.open if filepath.suffix == ".gz" else open
    errors = 0
    with open_fn(filepath, "rt", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                if errors <= 5:
                    print(f"  [WARN] Malformed JSON at line {line_num} — skipping")
    if errors > 0:
        print(f"  [WARN] Total malformed lines skipped: {errors}")


# ── Hard filter (pre-embedding, very fast) ────────────────────────────────────

def passes_hard_filter(candidate: dict) -> bool:
    """
    Quick filter to remove obvious non-fits before expensive embedding.
    Returns True if candidate should proceed to embedding + scoring.
    """
    p      = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    yoe     = float(p.get("years_of_experience") or 0)
    title   = (p.get("current_title") or "").lower()
    country = (p.get("country") or "").lower()
    relocate = bool(signals.get("willing_to_relocate", False))

    # ── Filter 1: YoE extremes ─────────────────────────────────────────────
    if yoe < 1.5 or yoe > 28:
        return False

    # ── Filter 2: Clearly non-technical current title ──────────────────────
    # Only filter if title is STRONGLY non-technical
    HARD_REJECT_TITLES = [
        "accountant", "auditor", "chef", "nurse", "doctor", "teacher",
        "driver", "electrician", "plumber", "security guard", "receptionist",
        "civil engineer",   # Not software
    ]
    if any(t in title for t in HARD_REJECT_TITLES):
        # But give benefit of doubt if they have AI skills
        skill_names = " ".join((s.get("name") or "").lower() for s in skills)
        if not any(kw in skill_names for kw in ["python", "machine learning", "data", "ml", "ai"]):
            return False

    # ── Filter 3: Zero technical skills AND no technical career ───────────
    all_text = " ".join([
        (s.get("name") or "").lower() for s in skills
    ] + [
        (j.get("description") or "").lower() for j in career
    ])
    TECH_SIGNALS = ["python", "java", "sql", "data", "machine learning", "software",
                    "engineering", "model", "algorithm", "api", "code", "developer"]
    if not any(sig in all_text for sig in TECH_SIGNALS):
        return False

    # ── Filter 4: Outside India AND unwilling to relocate ─────────────────
    if "india" not in country and not relocate:
        return False

    return True


# ── Main embedding loop ───────────────────────────────────────────────────────

def build_index(candidates_path: Path, jd_text: str | None = None):
    """
    Main pre-computation function.
    Builds FAISS index + metadata.pkl from candidates.jsonl.
    """
    print("\n" + "="*60)
    print("IntelliRank AI — Pre-computation Phase")
    print("="*60)

    # Load model (local, no network needed after first download)
    print("\n[1/5] Loading embedding model (BGE-small-en-v1.5)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print(f"  ✓ Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        sys.exit(1)

    # Parse JD
    jd = parse_jd(jd_text) if jd_text else JD_REQUIREMENTS
    print(f"\n[2/5] JD parsed. YoE range: {jd.yoe_min}–{jd.yoe_max}")

    # Embed JD text
    jd_embedding = model.encode(jd.raw_text, normalize_embeddings=True)
    np.save(ARTIFACTS_DIR / "jd_embedding.npy", jd_embedding)
    print(f"  ✓ JD embedding saved ({len(jd_embedding)}-dim)")

    # Stream and process candidates
    print(f"\n[3/5] Processing candidates from: {candidates_path}")
    embedding_dim = model.get_sentence_embedding_dimension()

    all_embeddings  = []
    all_metadata    = []  # Parallel list to all_embeddings
    batch_texts     = []
    batch_meta_temp = []

    total_seen = 0
    total_kept = 0
    t0 = time.time()

    for candidate in tqdm(stream_candidates(candidates_path), desc="Candidates", unit="cand"):
        total_seen += 1

        # Hard filter
        if not passes_hard_filter(candidate):
            continue

        total_kept += 1
        corpus_text = build_corpus_text(candidate)
        batch_texts.append(corpus_text)

        # Pre-compute structured scores (stored in metadata to avoid recompute at rank time)
        score_bd = compute_structured_score(candidate, jd)
        sig_ft   = compute_signal_features(candidate.get("redrob_signals", {}))
        hp_mult  = compute_honeypot_multiplier(candidate)

        batch_meta_temp.append({
            "candidate_id":      candidate["candidate_id"],
            "structured_score":  score_bd["structured_score"],
            "tech_depth":        score_bd["tech_depth"],
            "career_trajectory": score_bd["career_trajectory"],
            "company_fit":       score_bd["company_fit"],
            "location_fit":      score_bd["location_fit"],
            "yoe_fit":           score_bd["yoe_fit"],
            "education":         score_bd["education"],
            "disq_penalty":      score_bd["disqualifier_penalty"],
            "disqs":             score_bd["disqualifiers_triggered"],
            "tech_breakdown":    score_bd["tech_breakdown"],
            "career_breakdown":  score_bd["career_breakdown"],
            "company_breakdown": score_bd["company_breakdown"],
            "behavioral_mult":   sig_ft.get("behavioral_multiplier", 0.5),
            "signal_features":   sig_ft,
            "honeypot_mult":     hp_mult,
            # Store minimal profile info for reasoning generation
            "profile_snapshot":  {
                "current_title":   candidate.get("profile", {}).get("current_title"),
                "yoe":             candidate.get("profile", {}).get("years_of_experience"),
                "location":        candidate.get("profile", {}).get("location"),
                "current_company": candidate.get("profile", {}).get("current_company"),
            },
            "top_skills": [
                s["name"] for s in sorted(
                    candidate.get("skills", []),
                    key=lambda s: (s.get("endorsements", 0), s.get("duration_months", 0)),
                    reverse=True
                )[:5]
            ],
            "recent_job": {
                "title":   candidate.get("career_history", [{}])[0].get("title"),
                "company": candidate.get("career_history", [{}])[0].get("company"),
            } if candidate.get("career_history") else {},
        })

        # Process in batches for memory efficiency
        if len(batch_texts) >= BATCH_SIZE:
            embeddings = model.encode(
                batch_texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(embeddings)
            all_metadata.extend(batch_meta_temp)
            batch_texts.clear()
            batch_meta_temp.clear()

    # Process remaining batch
    if batch_texts:
        embeddings = model.encode(
            batch_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(embeddings)
        all_metadata.extend(batch_meta_temp)

    elapsed = time.time() - t0
    print(f"\n  ✓ Processed {total_seen:,} candidates in {elapsed:.1f}s")
    print(f"  ✓ Passed hard filter: {total_kept:,} candidates")

    # Build FAISS index
    print(f"\n[4/5] Building FAISS index...")
    try:
        import faiss
        all_vecs = np.vstack(all_embeddings).astype(np.float32)
        print(f"  Matrix shape: {all_vecs.shape}")

        index = faiss.IndexFlatIP(embedding_dim)
        index.add(all_vecs)
        print(f"  ✓ FAISS index built with {index.ntotal:,} vectors")

        faiss.write_index(index, str(ARTIFACTS_DIR / "faiss.index"))
        print(f"  ✓ Saved faiss.index")
    except Exception as e:
        print(f"  ✗ FAISS error: {e}")
        sys.exit(1)

    # Save metadata
    print(f"\n[5/5] Saving metadata...")
    with open(ARTIFACTS_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(all_metadata, f, protocol=4)
    print(f"  ✓ Saved metadata.pkl ({len(all_metadata):,} entries)")

    print(f"\n{'='*60}")
    print("Pre-computation complete!")
    print(f"  faiss.index   → {ARTIFACTS_DIR / 'faiss.index'}")
    print(f"  metadata.pkl  → {ARTIFACTS_DIR / 'metadata.pkl'}")
    print(f"  jd_embedding  → {ARTIFACTS_DIR / 'jd_embedding.npy'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntelliRank pre-computation")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--jd", default=None,
                        help="Path to custom job_description.txt (optional)")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: candidates file not found: {candidates_path}")
        sys.exit(1)

    jd_text = None
    if args.jd and Path(args.jd).exists():
        jd_text = Path(args.jd).read_text(encoding="utf-8")

    build_index(candidates_path, jd_text)
