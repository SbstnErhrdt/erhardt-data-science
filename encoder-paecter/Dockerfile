FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY encoder-paecter/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY encoder-paecter/ /app

CMD ["python", "cron.py"]
