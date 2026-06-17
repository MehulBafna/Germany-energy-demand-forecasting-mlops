# ─────────────────────────────────────────────────────────────
# EnergyPulse Dockerfile
# Builds a container for the FastAPI prediction service
# ─────────────────────────────────────────────────────────────

# Step 1 — Base image
# We start from an official Python image (not bare Ubuntu)
# "slim" = smaller image, no unnecessary tools
FROM python:3.11-slim

# Step 2 — Set working directory inside container
# All commands from here run inside /app
WORKDIR /app

# Step 3 — Install system dependencies
# These are OS-level packages needed by some Python libraries
# --no-install-recommends = don't install optional packages (keeps image small)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 4 — Copy requirements first (before code)
# Why? Docker caches each step. If requirements haven't changed,
# Docker skips reinstalling them even if your code changed.
# This makes rebuilds much faster.
COPY requirements.txt .

# Step 5 — Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6 — Copy project code
# We copy code AFTER installing requirements (for caching benefit)
COPY . .

# Step 7 — Create necessary directories
RUN mkdir -p data/raw data/processed models monitoring/reports monitoring/plots

# Step 8 — Set environment variables inside container
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1    
# PYTHONUNBUFFERED=1 means logs print immediately (not buffered)
# Important for seeing logs in real time in production

# Step 9 — Expose the port FastAPI runs on
# This documents which port the container uses
# Still needs -p 8000:8000 in docker run to actually map it
EXPOSE 8000

# Step 10 — Health check
# Docker periodically runs this to check if container is healthy
# If it fails 3 times → container marked as "unhealthy"
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Step 11 — Start command
# This runs when container starts
# Using uvicorn directly (not reload mode — that's for development only)
CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
