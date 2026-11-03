FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN nvcc --version
ENV CUDA_HOME=/usr/local/cuda
CMD ["python", "train.py"]
