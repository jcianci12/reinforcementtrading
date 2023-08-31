# Gym stuff
import datetime
import gym as gym
import gym_anytrading

# Stable baselines - RL stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from api import fetch_ohlcv_range
from load_model_or_create_if_not_exist import load_model_or_create_if_not_exist
from save_model import save_model



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
    

    state = env.reset()
    while True: 
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        if done: 
            print("info", info)
            break
            
    plt.figure(figsize=(15,6))
    plt.cla()
    plt.savefig("evaluate.png")
    plt.close()


    return env



#evaluate the model
def evaluate(data,model):
    env = gym.make('stocks-v0', df=data, frame_bound=(5,100), window_size=5)
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

    #train the model
def gettrainedmodel(data,model):
    env_maker = lambda: get_env(data)
    env = DummyVecEnv([env_maker])
    

    model.learn(total_timesteps=1000000)
    return model

def main():
    #get the data
    trainingdata = get_training_data()
    #get the env
    env = get_env(trainingdata)
    # model = load_model_or_create_if_not_exist("Model",env) 
    env_maker = lambda: gym.make('stocks-v0', df=trainingdata, frame_bound=(5,100), window_size=5)
    env = DummyVecEnv([env_maker])
    model = A2C('MlpPolicy', env, verbose=1) 
    model.learn(total_timesteps=1000000)


    env = gym.make('stocks-v0', df=trainingdata, frame_bound=(90,110), window_size=5)
    obs = env.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("info", info)
            break

    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.savefig("evaluate.png")
    plt.close()
    # plt.show()
    # print(env.action_space)
    #train the model
    # model = gettrainedmodel(trainingdata,model)
    # save_model(model,"Model")

main()
# on candle

