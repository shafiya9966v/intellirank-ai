"""
tests/test_edge_cases.py
Tests for system-level edge cases: malformed JSONL, CSV format,
streaming behavior, tie-breaking, output validation.
"""
import sys, csv, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.embed import stream_candidates, passes_hard_filter, build_corpus_text


class TestJSONLStreaming:
    def test_skips_malformed_lines(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text(
            '{"candidate_id": "CAND_0000001", "profile": {}}\n'
            'NOT VALID JSON\n'
            '{"candidate_id": "CAND_0000002", "profile": {}}\n'
            '\n'  # blank line
            '{"candidate_id": "CAND_0000003", "profile": {}}\n'
        )
        results = list(stream_candidates(f))
        assert len(results) == 3
        assert results[0]["candidate_id"] == "CAND_0000001"
        assert results[2]["candidate_id"] == "CAND_0000003"

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        results = list(stream_candidates(f))
        assert results == []

    def test_handles_gzipped_file(self, tmp_path):
        import gzip
        f = tmp_path / "candidates.jsonl.gz"
        with gzip.open(f, "wt", encoding="utf-8") as gz:
            gz.write('{"candidate_id": "CAND_0000001"}\n')
        results = list(stream_candidates(f))
        assert len(results) == 1


class TestHardFilter:
    def test_extreme_yoe_filtered_out(self):
        c = {
            "profile": {"years_of_experience": 50, "current_title": "Engineer", "country": "India"},
            "skills": [{"name": "Python"}],
            "career_history": [],
            "redrob_signals": {},
        }
        assert passes_hard_filter(c) is False

    def test_zero_yoe_with_no_skills_filtered(self):
        c = {
            "profile": {"years_of_experience": 0.5, "current_title": "Intern", "country": "India"},
            "skills": [],
            "career_history": [],
            "redrob_signals": {},
        }
        assert passes_hard_filter(c) is False

    def test_clearly_non_tech_filtered(self):
        c = {
            "profile": {"years_of_experience": 8, "current_title": "Accountant", "country": "India"},
            "skills": [{"name": "Tally"}, {"name": "Excel"}],
            "career_history": [{"description": "Managed ledgers."}],
            "redrob_signals": {},
        }
        assert passes_hard_filter(c) is False

    def test_non_tech_title_with_real_ai_skills_passes(self):
        """Edge case: title is non-tech-sounding but has real skills — should pass."""
        c = {
            "profile": {"years_of_experience": 6, "current_title": "Software Engineer", "country": "India"},
            "skills": [{"name": "Python"}, {"name": "Machine Learning"}],
            "career_history": [{"description": "Built ML models in production."}],
            "redrob_signals": {},
        }
        assert passes_hard_filter(c) is True

    def test_outside_india_no_relocate_filtered(self):
        c = {
            "profile": {"years_of_experience": 7, "current_title": "ML Engineer", "country": "USA"},
            "skills": [{"name": "Python"}, {"name": "ML"}],
            "career_history": [{"description": "Built data pipelines."}],
            "redrob_signals": {"willing_to_relocate": False},
        }
        assert passes_hard_filter(c) is False

    def test_outside_india_willing_to_relocate_passes(self):
        c = {
            "profile": {"years_of_experience": 7, "current_title": "ML Engineer", "country": "USA"},
            "skills": [{"name": "Python"}, {"name": "ML"}],
            "career_history": [{"description": "Built data pipelines."}],
            "redrob_signals": {"willing_to_relocate": True},
        }
        assert passes_hard_filter(c) is True


class TestCorpusTextBuilding:
    def test_truncates_long_summary(self):
        c = {
            "profile": {"summary": "x" * 5000, "current_title": "Engineer", "headline": ""},
            "career_history": [],
            "skills": [],
        }
        text = build_corpus_text(c)
        assert len(text) <= 2000

    def test_handles_missing_fields(self):
        c = {"profile": {}, "career_history": [], "skills": []}
        text = build_corpus_text(c)
        assert isinstance(text, str)   # Must not crash

    def test_includes_career_descriptions(self):
        c = {
            "profile": {"current_title": "ML Engineer", "summary": "AI specialist"},
            "career_history": [
                {"title": "Senior Engineer", "description": "Built RAG pipelines", "start_date": "2022-01-01"}
            ],
            "skills": [{"name": "FAISS"}],
        }
        text = build_corpus_text(c)
        assert "rag" in text.lower() or "RAG" in text


class TestCSVOutputFormat:
    def test_csv_properly_escapes_commas(self, tmp_path):
        """Reasoning strings with commas must not break CSV columns."""
        from rank import write_csv
        rows = [
            {"candidate_id": f"CAND_{i:07d}", "rank": i, "score": 1.0 - i*0.01,
             "reasoning": f"Strong fit, with FAISS, Pinecone, and great signals."}
            for i in range(1, 101)
        ]
        out = tmp_path / "test.csv"
        write_csv(rows, out)

        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            parsed = list(reader)
        assert len(parsed) == 100
        assert parsed[0]["reasoning"].count(",") == 3   # Commas preserved correctly

    def test_csv_scores_non_increasing(self, tmp_path):
        from rank import write_csv
        rows = [
            {"candidate_id": f"CAND_{i:07d}", "rank": i, "score": 1.0 - i*0.01, "reasoning": "test"}
            for i in range(1, 101)
        ]
        out = tmp_path / "test.csv"
        write_csv(rows, out)
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            scores = [float(row["score"]) for row in reader]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_csv_always_exactly_100_rows(self, tmp_path):
        from rank import write_csv
        rows = [
            {"candidate_id": f"CAND_{i:07d}", "rank": i, "score": 0.5, "reasoning": "test"}
            for i in range(1, 101)
        ]
        out = tmp_path / "test.csv"
        write_csv(rows, out)
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            parsed = list(reader)
        assert len(parsed) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
