import csv
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    
    FLAT = 0
    LONG = 1
    SHORT = 2

    def __init__(self, ohlcv, ret):
        super(TradingEnv, self).__init__()
        
        self.ohlcv = ohlcv
        self.ret = ret
        self.trading_cost = 0.01
        self.reward = 1
        
        # The number of step the training has taken, starts at 5 since we're using the previous 5 data for observation.
        self.current_step = 5
        # The last action
        self.last_action = 0  # Now a scalar value representing the last amount bought/sold

        # Define action space as Box with shape (1,), representing the amount to buy (positive) or sell (negative)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        dfwidth = ohlcv.shape[1]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5 * ohlcv.shape[1] + 2,), dtype=np.float64)
        
        self.initialCash = 1000  # Initialize starting amount here
        self.AssetAmount = 0  # Initialize starting amount here
        self.previous_price = 0

        self.cash = self.initialCash  # Initialize current amount here
        self.current_asset_amount = self.AssetAmount  # Initialize current amount here
        # Reset the history at the start of each episode
        self.price_action_history = []
        headers = ['Cash', 'Current Asset Amount', 'Ret', 'Action', 'Prev','Current','Rew','change']
        self.price_action_history.append(headers)   

        self.price_action_history.append((self.initialCash,self.AssetAmount,self.ret[self.current_step], 0,self.reward,self.cash,self.current_asset_amount,None))
 

    def step(self, action):
        current_price = self.ret[self.current_step]
        prev_portfolio_value = self.cash+(self.current_asset_amount*self.previous_price)

        amount = action[0]*1000#multiply out from the normalised action space to get to our max spend amount.
        


        if amount > 0:  # Buying
            if amount * (1 + self.trading_cost) > self.cash:
                self.reward -=0.1 # Punishment
                self.reward
            else:
                self.cash -= amount
                self.current_asset_amount += amount/current_price                  
        elif amount < 0:  # Selling
            if abs(amount/current_price) > self.current_asset_amount:
                self.reward -=0.1  # Punishment
                self.reward
            else:
                #becauase we are dealing with negatives. We need to reverse
                self.cash -= amount   
                self.current_asset_amount += amount /current_price 
        # elif amount ==0:  # Holding
        #     # self.reward =+ np.log1p(change)
        #     self.reward += change
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        current_portfolio_value = self.cash+(self.current_asset_amount*current_price)

        change =  current_portfolio_value- prev_portfolio_value       
        self.reward += change*10

        self.previous_price = current_price


        # self.reward = self.reward*10
        self.last_action = action
        self.current_step += 1

        # Have we iterate all data points?
        done = (self.current_step == self.ret.shape[0]-1)
        # print(f"Step: {self.current_step}, Action: {amount}, Reward: {self.reward}, Cash: {self.cash}, Asset: {self.current_asset_amount}, Total: {current_portfolio_value} ")

        # Return observation, reward, terminated, truncated, and info dictionary
        observation = np.append(self.ohlcv[self.current_step-5:self.current_step], [self.cash, self.current_asset_amount]).astype(np.float32)        
        truncated = False  # You'll need to define this based on your environment's logic
        info = {}  # Empty info dictionary
        # Assuming price_action_history is a list of tuples
        self.price_action_history.append((self.cash, 
                                              self.current_asset_amount, 
                                              self.ret[self.current_step], 
                                              action[0], 
                                              prev_portfolio_value,
                                              current_portfolio_value,
                                              self.reward,
                                              change))
        # if(done):
            # print(f"Step: {self.current_step}, Action: {action}, Reward: {self.reward}, Cash: {self.cash}, Asset: {self.current_asset_amount*current_price}, Total: {current_portfolio_value} ")
                        # Plot the price and action over time at the end of each episode

            # pd.DataFrame(self.price_action_history).to_csv('train.csv',mode='w' )




            
        return observation, self.reward, done, truncated, info

    def plotGraph(self):
        n=50
        cash,asset_amount,prices, actions,reward,prev,current,change = zip(*self.price_action_history[-n:])
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('price', color=color)
        ax1.plot(prices, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('action', color=color)  
        ax2.plot(actions, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        fig.savefig('plot.png')
        plt.close(fig)


    def reset(self ,seed=None):

        self.cash = self.initialCash  # Reset current amount at the start of each episode
        self.current_asset_amount = self.AssetAmount  # Initialize starting amount here
        # Reset the number of step the training has taken
        self.current_step = 5
        self.reward = 1
        # Reset the last action
        self.last_action = 0  # Now a scalar value representing the last amount bought/sold
        # must return np.array type
        obs = np.append(self.ohlcv[self.current_step-5:self.current_step], [self.cash, self.current_asset_amount]).astype(np.float32)        
        info = {}  # Empty info dictionary
        self.price_action_history = []

        return obs, info
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f'Equity Value: {self.reward}')