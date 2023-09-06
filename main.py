# Gym stuff
import datetime
import os
import shutil
# import gym
import gym_anytrading
import gymnasium as gym

# Stable baselines - RL stuff
from stable_baselines3 import A2C
# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from api import fetch_ohlcv_range,get_dates_and_data_from_latest_file
from load_model_or_create_if_not_exist import load_model_or_create_if_not_exist
from save_model import save_model



#get the range
def fetch_range():
    end = datetime.datetime.now()    
    start = end - datetime.timedelta(0,60*1000)
    return start,end
#get the data
def get_training_data():
    s,e,data = get_dates_and_data_from_latest_file()
    if(data.empty):
        s, e = fetch_range()
        df_new = fetch_ohlcv_range(
        s, e, "BTCUSDT", "1m", "spot")
    else:
        df_new = fetch_ohlcv_range(
        s, e, "BTCUSDT", "1m", "spot")
        a = data['Date'].iloc[-1]
        b = df_new['Date'].iloc[1]
        # Calculate the time delta in seconds
        delta_seconds = (a - b) / 1000

        # Convert the time delta to a timedelta object
        print(f"concatentating last date from cached csv{a} with df_new {b} time delta {delta_seconds}")

        df_new = pd.concat([data,df_new])

        # Create the file name using the start and end dates in Unix format

    file_name = f"{df_new['Date'].iloc[0]}_{df_new['Date'].iloc[-1]}.csv"
    if(os.path.exists("data")):
        shutil.rmtree("data")
    # Save the DataFrame to a CSV file with the specified file name
    os.mkdir("data")
    df_new.to_csv(f"data/{file_name}",index=False)

    return df_new


#get the environment
def get_env(data,s,e):
    env = gym.make('stocks-v0', df=data, frame_bound=(s,e), window_size=5)
    

    state = env.reset()
    while True: 
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        if done: 
            print("info", info)
            break
            
    plt.figure(figsize=(15,6))
    plt.cla()
    plt.savefig("env.png")
    plt.close()
    return env

def explore(env):
    # Explore the environment
    env.action_space

    state = env.reset()
    while True:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        if done:
            print("info", info)
            break

    plt.figure(figsize=(15, 6))
    plt.cla()
    env.render_all()
    plt.savefig("explore")



    #train the model
def gettrainedmodel(model):
    # Creating our dummy vectorizing environment

    model.learn(total_timesteps=100000)
    return model
#evaluate the model
def evaluate(data,model,env):

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
    plt.savefig("evaluate")
    plt.close()

def main():
    #get the data
    trainingdata = get_training_data()

    #get the env
    env = get_env(trainingdata,5,200)
    print(trainingdata)
    print(env)
    explore(env)
    model = load_model_or_create_if_not_exist("model",env)
    model = gettrainedmodel(model)
    save_model(model,"Model")   

    evaluate(trainingdata,model,get_env(trainingdata,190,210))

main()
# on candle

