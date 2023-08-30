FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
       build-essential \
       libpq-dev

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1


COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace
