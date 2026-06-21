# ── Stage: base image ──────────────────────────────────────────────────────────
# Use slim Python image for CPU-only (Mac/Linux without GPU).
# Switch to pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime for NVIDIA GPU support.
FROM python:3.12-slim

# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    # Cache HuggingFace models inside the container (can be volume-mounted)
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Install Python dependencies ────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ────────────────────────────────────────────────────
COPY app/ ./app/
COPY .env.example .

# ── Expose Gradio port ─────────────────────────────────────────────────────────
EXPOSE 7860

# ── Healthcheck ────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Run the application ────────────────────────────────────────────────────────
CMD ["python", "-m", "app.main"]
