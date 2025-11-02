FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

ARG TORCH_VERSION=2.2.2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && ln -s /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

COPY encoder-paecter/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY encoder-paecter/ /app

CMD ["python", "cron.py"]
