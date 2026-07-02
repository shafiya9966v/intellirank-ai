# IntelliRank AI — Reproducible Ranking Container
# Judges run this to verify the submission at Stage 3
# Constraints: CPU only, no network during ranking step, 16GB RAM, 5 min

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Make artifacts directory
RUN mkdir -p artifacts

# Default command: run the ranking step
# Pre-computation (embed.py) must be run separately before this
CMD ["python", "rank.py", "--candidates", "./candidates.jsonl", "--out", "./submission.csv"]
