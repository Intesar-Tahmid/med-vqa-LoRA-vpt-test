FROM ghcr.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Install git and build tools which are required for pip install from git repositories
RUN apt-get update && apt-get install -y git build-essential gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]