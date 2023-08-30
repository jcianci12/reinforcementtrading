# Gym stuff
import datetime
import gymnasium as gym
import gym_anytrading

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
# from stable_baselines3_contrib.ppo_lstm import PPO

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from api import fetch_ohlcv_range



#get the range
def fetch_range():
    end = datetime.datetime.now()
    start = end - datetime.timedelta(0,60*1000)
    return start,end
#get the data
def get_training_data():
    s, e = fetch_range()
    trainingdata = fetch_ohlcv_range(
        True, s, e, "BTCUSDT", "1m", "spot")
    return trainingdata

#get the environment
def get_env(data):
    env = gym.make('stocks-v0', df=data, frame_bound=(5,len(data)), window_size=5)
    return env

#train the model
def gettrainedmodel(data):
    env_maker = lambda: get_env(data)
    env = DummyVecEnv([env_maker])

    model = A2C('MlpPolicy', env, verbose=1) 
    model.learn(total_timesteps=1000000)
    return model

#evaluate the model
def evaluate(data,model):
    env = gym.make('stocks-v0', df=data, frame_bound=(5,len(data)), window_size=5)
    state = env.reset()
    while True: 
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated        
        if done: 
            print("info", info)
            break

    plt.figure(figsize=(15,6))
    plt.cla()
    env.unwrapped.render_all()
    plt.savefig("evaluate.png")
    plt.close()

def main():
    #get the data
    trainingdata = get_training_data()
    #get the env
    env = get_env(trainingdata)
    
    # print(env.action_space)
    #train the model
    model = gettrainedmodel(trainingdata)        

main()
# on candle

