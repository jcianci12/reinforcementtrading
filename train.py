# Gym stuff
import datetime
import os
import shutil
import subprocess
import webbrowser

# Stable baselines - RL stuff
from stable_baselines3 import A2C, PPO
# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Trading_Env import TradingEnv


from api import fetch_ohlcv_range,get_dates_and_data_from_latest_file
from evaluate import evaluate
from load_model_or_create_if_not_exist import load_model_or_create_if_not_exist
from prep_data import prep_data
from save_model import save_model
import signal
from stable_baselines3.common.env_checker import check_env


#get the range
def fetch_range():
    end = datetime.datetime.now()    
    start = end - datetime.timedelta(0,300*1000)
    return start,end
#get the data
def get_training_data():
    s,e,data = get_dates_and_data_from_latest_file()
    if(data.empty):
        interval = '5m'
        s, e = fetch_range()
        df_new = fetch_ohlcv_range(
        s, e, "BTCUSDT", "5m", "spot")
    else:
        
        df_new = fetch_ohlcv_range(
        s, e, "BTCUSDT", "5m", "spot")
        a = data['date'].iloc[-1]
        b = df_new['date'].iloc[0]
        # Calculate the time delta in seconds
        delta_seconds = (a - b) / 1000

        # Convert the time delta to a timedelta object
        print(f"concatentating last date from cached csv{a} with df_new {b} time delta {delta_seconds}")

        df_new = pd.concat([data,df_new])


        # Create the file name using the start and end dates in Unix format

    file_name = f"{df_new['date'].iloc[0]}_{df_new['date'].iloc[-1]}.csv"
    if(os.path.exists("data")):
        shutil.rmtree("data")
    # Save the DataFrame to a CSV file with the specified file name
    os.mkdir("data")
    df_new.to_csv(f"data/{file_name}",index=False)
    df_new.columns = df_new.columns.str.lower()


    return df_new



#get the environment
def get_env(X_train,y_train):
    env = TradingEnv(X_train,y_train)
    
    # 
    # Assuming `CustomEnv` is your custom environment class
    check_env(env)
    return env

    #train the model
def trainmodel(model: PPO) -> PPO:
    # Launch TensorBoard
    # Open TensorBoard in a web browser
    # webbrowser.open("http://localhost:6006")
    print("training")
    model.learn(total_timesteps=500000,tb_log_name="PPO")
    save_model(model,"Model")   

# Close the TensorBoard server
    return model

def main():
    #get the data
    trainingdata = get_training_data()
    trainingdata = prep_data(trainingdata)
    # ret = np.log(trainingdata/trainingdata.shift(1)).iloc[1:].close
    ret = trainingdata.shift(1).close
    n = 100
    X_train = trainingdata.iloc[:n].values
    X_test = trainingdata.iloc[-n:].values
    y_train = ret.iloc[:n].values
    y_test = ret.iloc[-n:].values


    env = get_env(X_train,y_train)
    
        # sample action:
    print("sample action:", env.action_space.sample())

    # observation space shape:
    print("observation space shape:", env.observation_space.shape)

    # sample observation:
    print("sample observation:", env.observation_space.sample())

    # explore(env)
    model = load_model_or_create_if_not_exist("model",env)
    model = trainmodel(model)

    # evaluate(X_test,y_test,model,env)
    main()

main()
# on candle
