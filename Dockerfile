# RTX-STone Docker Image
# PyTorch 2.10 with native SM 12.0 support for RTX 50-series GPUs
#
# Build:
#   docker build -t rtx-stone:latest .
#
# Run:
#   docker run --gpus all -it rtx-stone:latest
#
# Run with Jupyter:
#   docker run --gpus all -p 8888:8888 -it rtx-stone:latest jupyter notebook --ip=0.0.0.0 --allow-root

FROM nvidia/cuda:12.0-devel-ubuntu22.04

LABEL maintainer="RTX-STone Contributors"
LABEL description="PyTorch with native SM 12.0 support for RTX 50-series GPUs"
LABEL version="2.10.0a0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    nvidia-utils-570 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Create working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt /workspace/
COPY pyproject.toml /workspace/
COPY setup.py /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy RTX-STone package
COPY rtx_stone/ /workspace/rtx_stone/
COPY *.py /workspace/
COPY examples/ /workspace/examples/
COPY notebooks/ /workspace/notebooks/
COPY integrations/ /workspace/integrations/
COPY README.md LICENSE CHANGELOG.md /workspace/

# Install RTX-STone in development mode
RUN pip install -e .

# Install optional dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    matplotlib \
    vllm \
    langchain \
    langchain-community \
    sentence-transformers \
    faiss-gpu

# Set up Jupyter
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose Jupyter port
EXPOSE 8888

# Expose vLLM API port
EXPOSE 8000

# Default command
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
