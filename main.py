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
    
    state = env.reset()
    while True: 
        action = env.action_space.sample()
        #how can I use this info returned from the step function?
        #self.ohlcv[self.current_step-5:self.current_step].astype(np.float32), self.reward, done, {}
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated
        if done:
            print("Info",info) 
            break           
    plt.figure(figsize=(15,6))
    plt.cla()
    plt.savefig("env.png")
    plt.close()
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

def evaluate(X_test,y_test,model):
    equity = [1]
    last_action = 0
    trading_cost = 0.01
    current_amount = 1000  # Initialize starting amount here
    current_asset_amount = 0  # Initialize starting amount here

    for i in range(5, X_test.shape[0]):
        observation = np.append(X_test[i-5:i], [current_amount, current_asset_amount]).astype(np.float32)
        action, _ = model.predict(observation, deterministic=True)
        print(f"choosing:{action}")
        currenty_test = y_test[i]

        amount = action[0]  # Extract the amount from the action

        if amount > 0:  # Buying
            if amount * (1 + trading_cost) > current_amount:
                reward = -1  # Punishment
            else:
                change = (1 + currenty_test - trading_cost)
                current_amount -= amount * change
                current_asset_amount += amount
                reward = np.log1p(change)
        elif amount < 0:  # Selling
            if abs(amount) > current_asset_amount:
                reward = -1  # Punishment
            else:
                change = (1 + -1 * currenty_test - trading_cost)
                current_amount -= amount * change  # Subtract because amount is negative
                current_asset_amount += amount  # Add because amount is negative
                reward = np.log1p(change)
        elif amount == 0:  # Holding
            change = 1
            reward = np.log1p(change)
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        new = equity[-1] * (1 + reward)  # Calculate new equity value based on the reward
        equity.append(new)
        last_action = action


    plt.figure(figsize=(15, 10))
    plt.title("Equity Curve")
    plt.xlabel("timestep")
    plt.ylabel("equity")
    plt.plot(equity)
    plt.savefig("equity")
    plt.close()



def main():
    #get the data
    trainingdata = get_training_data()
    trainingdata = prep_data(trainingdata)
    ret = np.log(trainingdata/trainingdata.shift(1)).iloc[1:].close


    X_train = trainingdata.iloc[:-50].values
    X_test = trainingdata.iloc[-50:].values
    y_train = ret.iloc[:-50].values
    y_test = ret.iloc[-50:].values


    env = get_env(X_train,y_train)
    
        # sample action:
    print("sample action:", env.action_space.sample())

    # observation space shape:
    print("observation space shape:", env.observation_space.shape)

    # sample observation:
    print("sample observation:", env.observation_space.sample())

    # explore(env)
    model = load_model_or_create_if_not_exist("model",env)
    # model = trainmodel(model)

    evaluate(X_test,y_test,model)

main()
# on candle
