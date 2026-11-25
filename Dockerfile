# Lightweight Python runtime container
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY report/ ./report/

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to ensure container availability
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the interactive inference dashboard
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
