# Gym stuff
import datetime
import gymnasium as gym
import gym_anytrading

# Stable baselines - rl stuff
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import A2C

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from api import fetch_ohlcv_range


# init
# we need some data to create the environment
# 
def fetch_range():
    end = datetime.datetime.now()
    start = end - datetime.timedelta(0,60*9000)
    return start,end
def get_training_data():
    s, e = fetch_range()
    trainingdata = fetch_ohlcv_range(
        True, s, e, "BTCUSDT", "1m", "spot")
    return trainingdata

def get_env(data):
    env = gym.make('stocks-v0', df=data, frame_bound=(5,len(data)), window_size=5)
    return env
    

# initialise the model
# if the model doesnt have any training data, we need to train it on some back data and ensure its win rate is ok

# if its ok we can sub to the candles

# sub to the candles
def main():
    trainingdata = get_training_data()
    env = get_env(trainingdata)
    print(env.action_space)
    state = env.reset()
    while True: 
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated        
        if done: 
            print("info", info)
            break
            
    plt.cla()
    env.unwrapped.render_all()
    plt.savefig("test.png")
    plt.close()

main()
# on candle

