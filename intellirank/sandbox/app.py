"""
sandbox/app.py — Gradio demo for HuggingFace Spaces.
Runs on 50-candidate sample. Full pipeline, live demo.
"""
import sys, json, csv, io, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "candidates_sample.jsonl"

def load_sample():
    candidates = []
    if SAMPLE_PATH.exists():
        with open(SAMPLE_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: candidates.append(json.loads(line))
                    except: pass
    return candidates

def run_ranking_demo(jd_override=""):
    from core.jd_parser import JD_REQUIREMENTS, parse_jd
    from core.scorer import compute_structured_score
    from core.signals import compute_signal_features
    from core.honeypot import compute_honeypot_multiplier
    from core.reasoning import generate_reasoning

    t0 = time.time()
    candidates = load_sample()
    if not candidates:
        return "<p>❌ Sample data not found.</p>", "Error: no data"

    jd = parse_jd(jd_override) if jd_override.strip() else JD_REQUIREMENTS
    scored = []

    for c in candidates:
        score_bd  = compute_structured_score(c, jd)
        sig_ft    = compute_signal_features(c.get("redrob_signals", {}))
        hp_mult   = compute_honeypot_multiplier(c)
        behavioral = sig_ft["behavioral_multiplier"]
        sem_sim   = score_bd["structured_score"] * 0.9 + 0.05
        final     = (sem_sim * 0.35 + score_bd["structured_score"] * 0.65) * hp_mult * behavioral
        top_skills = [s["name"] for s in sorted(c.get("skills",[]), key=lambda s:(s.get("endorsements",0),s.get("duration_months",0)), reverse=True)[:3]]
        scored.append({
            "candidate_id": c["candidate_id"],
            "title":  c["profile"].get("current_title","Unknown"),
            "yoe":    c["profile"].get("years_of_experience",0),
            "location": c["profile"].get("location",""),
            "final":  max(0.001, min(1.0, final)),
            "tech":   score_bd["tech_depth"],
            "avail":  behavioral,
            "disqs":  score_bd["disqualifiers_triggered"],
            "skills": top_skills,
            "score_bd": score_bd,
            "sig_ft": sig_ft,
            "cand":   c,
        })

    scored.sort(key=lambda x: -x["final"])
    elapsed = time.time() - t0
    rows_html = ""
    for rank, item in enumerate(scored[:20], 1):
        dq = f'<br><span style="color:#dc2626;font-size:11px">⚠ {item["disqs"][0][:55]}</span>' if item["disqs"] else ""
        badges = " ".join(f'<span style="background:#EFF6FF;color:#1E40AF;border-radius:4px;padding:1px 6px;font-size:11px">{s}</span>' for s in item["skills"])
        bar = int(item["final"] * 180)
        rows_html += f"""<tr style="border-bottom:1px solid #F3F4F6">
          <td style="padding:10px;font-weight:600;color:#1E40AF">#{rank}</td>
          <td style="padding:10px;font-size:11px;color:#9CA3AF">{item['candidate_id']}</td>
          <td style="padding:10px"><div style="font-weight:500">{item['title'][:38]}</div>
              <div style="font-size:12px;color:#6B7280">{item['location']} · {item['yoe']:.0f}yr</div></td>
          <td style="padding:10px">{badges}</td>
          <td style="padding:10px">
            <div style="display:flex;align-items:center;gap:6px">
              <div style="width:70px;height:7px;background:#F3F4F6;border-radius:4px">
                <div style="width:{bar}px;height:100%;background:#1E40AF;border-radius:4px"></div></div>
              <b>{item['final']:.3f}</b></div>
            <div style="font-size:11px;color:#6B7280">Tech:{item['tech']:.2f} · Avail:{item['avail']:.2f}</div>
            {dq}</td></tr>"""

    table = f"""<div style="font-family:system-ui,sans-serif">
      <div style="background:#EFF6FF;padding:10px 14px;border-radius:8px;margin-bottom:14px;font-size:13px;color:#1E40AF">
        ✅ Ranked {len(scored)} candidates in {elapsed:.2f}s · 5-layer pipeline · Showing top 20
      </div>
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="background:#F9FAFB;font-size:12px;color:#6B7280">
          <th style="padding:10px;text-align:left">Rank</th><th style="padding:10px;text-align:left">ID</th>
          <th style="padding:10px;text-align:left">Candidate</th><th style="padding:10px;text-align:left">Key Skills</th>
          <th style="padding:10px;text-align:left">Score</th></tr></thead>
        <tbody>{rows_html}</tbody></table></div>"""

    status = f"✅ Ranked {len(scored)} candidates in {elapsed:.2f}s | Top: {scored[0]['title']} ({scored[0]['final']:.4f})"
    return table, status

def export_csv_demo():
    from core.jd_parser import JD_REQUIREMENTS
    from core.scorer import compute_structured_score
    from core.signals import compute_signal_features
    from core.honeypot import compute_honeypot_multiplier
    from core.reasoning import generate_reasoning

    candidates = load_sample()
    scored = []
    for c in candidates:
        score_bd  = compute_structured_score(c, JD_REQUIREMENTS)
        sig_ft    = compute_signal_features(c.get("redrob_signals", {}))
        hp_mult   = compute_honeypot_multiplier(c)
        behavioral = sig_ft["behavioral_multiplier"]
        sem       = score_bd["structured_score"] * 0.9 + 0.05
        final     = (sem * 0.35 + score_bd["structured_score"] * 0.65) * hp_mult * behavioral
        sig_ft["behavioral_multiplier"] = behavioral
        scored.append((max(0.001, min(1.0, final)), score_bd, sig_ft, c))
    scored.sort(key=lambda x: -x[0])

    out = io.StringIO()
    w = csv.writer(out, quoting=csv.QUOTE_ALL)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, (final_sc, score_bd, sig_ft, c) in enumerate(scored, 1):
        reasoning = generate_reasoning(c, score_bd, sig_ft, rank, final_sc)
        w.writerow([c["candidate_id"], rank, f"{final_sc:.4f}", reasoning])
    return out.getvalue()

JD_DEFAULT = """Senior AI Engineer – Founding Team | Redrob AI | Pune/Noida | 5-9 Years | 35-38 LPA

Hard Requirements:
- Production embedding-based retrieval (sentence-transformers, BGE, FAISS, vector DBs)
- Strong Python, eval frameworks (NDCG, MRR, A/B testing)
- 5-9 years at product companies

NOT a Fit:
- Entire career at TCS/Infosys/Wipro/Accenture/Cognizant
- AI experience only LangChain/OpenAI wrappers
- CV/Speech/Robotics primary domain"""

with gr.Blocks(title="IntelliRank AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 IntelliRank AI\n**Intelligent Candidate Ranking — Redrob India Runs 2026** | Team: Force Push")
    with gr.Row():
        with gr.Column(scale=1):
            jd_input = gr.Textbox(label="Job Description", value=JD_DEFAULT, lines=14)
            with gr.Row():
                rank_btn   = gr.Button("🚀 Rank Candidates", variant="primary")
                export_btn = gr.Button("📥 Export CSV")
        with gr.Column(scale=2):
            status_out  = gr.Textbox(label="Status", interactive=False)
            results_out = gr.HTML()
    csv_out = gr.Textbox(label="CSV Output", lines=8, visible=False)
    rank_btn.click(run_ranking_demo, inputs=[jd_input], outputs=[results_out, status_out])
    export_btn.click(export_csv_demo, inputs=[], outputs=[csv_out]).then(
        lambda: gr.update(visible=True), outputs=[csv_out])
    gr.Markdown("---\n**Pipeline:** BGE embeddings → FAISS → 7-dim scoring → honeypot detection → behavioral gating → reasoning")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
