# modified from: https://github.com/huggingface/diffusers/blob/main/docker/diffusers-pytorch-cuda/Dockerfile
# added download diffuser to image at build 


FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"


# CMD ["/bin/bash", "nvidia-smi"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   libsndfile1-dev \
                   python3.8 \
                   python3-pip \
                   python3.8-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
         accelerate \
         diffusers[torch]>=0.10 \
         ftfy \
         torch \ 
         torchvision \
         transformers \
         triton \
         safetensors \
         xformers==0.0.16rc399 \
         huggingface_hub

RUN python -c 'from huggingface_hub.hf_api import HfFolder; HfFolder.save_token("xxx")'


WORKDIR /app
COPY ecs_run.py /app/
COPY download_diffusers.py /app/

RUN mkdir /app/model_data

# model_id = "runwayml/stable-diffusion-v1-5"
# cache_path = "/app/model_data

# download model

# ENV HUGGINGFACE_TOKEN=xxxxxx

# read env variable HUGGINGFACE_TOKEN and add it to ENV variable
# ENV HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

RUN HUGGINGFACE_TOKEN=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 python3 download_diffusers.py --model_id runwayml/stable-diffusion-v1-5 --cache_path /app/model_data


ENTRYPOINT ["/bin/bash"]

# ENTRYPOINT ["python3", "ecs_run.py", "--model_id", "runwayml/stable-diffusion-v1-5", "--cache_path", "./model_data"]


## build docker image with env variable HUGGINGFACE_TOKEN=your_token
# docker build -t <image_name> --build-arg HUGGINGFACE_TOKEN=your_token .


# run docker image with env variable HUGGINGFACE_TOKEN=your_token
# docker run -it --rm --gpus all -e HUGGINGFACE_TOKEN=your_token <image_id>

# python3 ecs_run.py --model_id runwayml/stable-diffusion-v1-5 --cache_path ./model_data