FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 FLIGHTLAKE_ROOT=/data/lake AEROSIG_ARTIFACTS=/artifacts
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
EXPOSE 8001
CMD ["uvicorn", "aerosig.service:app", "--host", "0.0.0.0", "--port", "8001"]
