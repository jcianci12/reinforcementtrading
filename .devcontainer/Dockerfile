FROM mcr.microsoft.com/devcontainers/python:3.11


ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN pip install gym==0.25.2

RUN pip install tensorflow
RUN pip install keras-rl2
RUN pip install stable_baselines3
# RUN pip install gym-anytrading

RUN pip install python-binance
RUN pip install ccxt
RUN pip install finta
RUN pip install scikit-learn
RUN pip install seaborn



WORKDIR /workspace
