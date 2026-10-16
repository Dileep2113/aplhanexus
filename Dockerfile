# Multi-stage production container for AlphaNexus Institutional Suite
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies for curl_cffi and compiling C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final lean runtime image
FROM python:3.11-slim

WORKDIR /app

# Install runtime libcurl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed wheels from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PORT=8050
ENV ENABLE_LIVE_BROKER_EXECUTION=false

# Copy application source code
COPY . .

# Expose web service port
EXPOSE 8050

# Run web app dashboard and background scheduler daemon
CMD ["python", "run_web.py"]
