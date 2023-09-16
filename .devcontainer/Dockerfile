FROM mcr.microsoft.com/devcontainers/python


ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1


RUN pip install tensorflow
RUN pip install stable_baselines3
RUN pip install gym
RUN pip install gym-anytrading

RUN pip install python-binance
RUN pip install ccxt
RUN pip install finta
RUN pip install scikit-learn



WORKDIR /workspace
