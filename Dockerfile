# Dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# 1) System deps
RUN apt-get update && \
    apt-get install -y python3 python3-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# 2) In‐container venv (optional but keeps system clean)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3) Copy in your code & requirements
WORKDIR /workspace
COPY requirements.txt /workspace/
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
COPY . /workspace

# 4) Default entrypoint
ENTRYPOINT ["python", "main.py"]
