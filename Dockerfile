FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3 ffmpeg libsm6 libxext6 git \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    libegl1-mesa libegl1 libgles2 mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]