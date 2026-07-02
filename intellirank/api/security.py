from __future__ import annotations
import re
from pathlib import Path

ALLOWED_EXTENSIONS = {".jsonl", ".json", ".gz"}
MAX_FILE_SIZE_MB   = 600
MAX_JD_LENGTH      = 10_000

def validate_jd_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {3,}", "  ", text)
    return text[:MAX_JD_LENGTH].strip()

def validate_upload_file(filename: str, file_size: int) -> tuple[bool, str]:
    path = Path(filename)
    suffix = path.suffix.lower()
    if filename.endswith(".jsonl.gz"):
        suffix = ".gz"
    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
    size_mb = file_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB"
    return True, ""
