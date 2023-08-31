FROM mcr.microsoft.com/devcontainers/python:3.7


ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# COPY requirements.txt .
# RUN conda install python=3.7
RUN pip install tensorflow
RUN pip install stable_baselines3
RUN pip install gym-anytrading
RUN pip install python-binance
RUN pip install ccxt
RUN pip install gym-anytrading
RUN pip install shimmy>=0.2.1
# RUN pip install tensorflow-gpu
# RUN pip install protobuf
# RUN pip uninstall -y protobuf
# RUN pip install protobuf==3.19.4
# RUN apt-get update && apt-get install -y python3-opencv

# RUN pip install opencv-python
# RUN pip uninstall -y gym
# RUN pip install gym==0.19.0


WORKDIR /workspace
# # Set up Jupyter notebook
# EXPOSE 8888
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]