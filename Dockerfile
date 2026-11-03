FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ARG IMAGE_TAG=dev
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    IMAGE_TAG=${IMAGE_TAG} \
    MODELS_PATH=/models/latex-ocr

# Model weights live in a volume, not in the image (IP-018).
VOLUME ["/models"]

CMD ["celery", "--workdir", "src", "-A", "worker", "worker", "--loglevel=info", "-E", "--concurrency=1", "--queues=evilflowers_equation"]
