import gymnasium as gym
import numpy as np


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
        self.action_space = gym.spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32)

        dfwidth = ohlcv.shape[1]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5 * ohlcv.shape[1] + 2,), dtype=np.float64)
        
        self.basePairAmount = 1000  # Initialize starting amount here
        self.AssetAmount = 0  # Initialize starting amount here

        self.current_amount = self.basePairAmount  # Initialize current amount here
        self.current_asset_amount = self.AssetAmount  # Initialize current amount here


    def reset(self ,seed=None):
        self.current_amount = self.basePairAmount  # Reset current amount at the start of each episode
        self.current_asset_amount = self.AssetAmount  # Initialize starting amount here
        # Reset the number of step the training has taken
        self.current_step = 5
        self.reward = 1
        # Reset the last action
        self.last_action = 0  # Now a scalar value representing the last amount bought/sold
        # must return np.array type
        obs = np.append(self.ohlcv[self.current_step-5:self.current_step], [self.current_amount, self.current_asset_amount]).astype(np.float32)        
        info = {}  # Empty info dictionary
        return obs, info
    

    def step(self, action):
        amount = action[0]

        if amount > 0:  # Buying
            if amount * (1 + self.trading_cost) > self.current_amount:
                self.reward = -1  # Punishment
            else:
                change = (1 + self.ret[self.current_step] - self.trading_cost)
                self.current_amount -= amount * change
                self.current_asset_amount += amount
                self.reward = np.log1p(change)
        elif amount < 0:  # Selling
            if abs(amount) > self.current_asset_amount:
                self.reward = -1  # Punishment
            else:
                change = (1 + -1 * self.ret[self.current_step] - self.trading_cost)
                self.current_amount -= amount * change  # Subtract because amount is negative
                self.current_asset_amount += amount  # Add because amount is negative
                self.reward = np.log1p(change)
        elif amount ==0:  # Holding
            change = 1
            self.reward = np.log1p(change)
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        self.last_action = action
        self.current_step += 1

        # Have we iterate all data points?
        done = (self.current_step == self.ret.shape[0]-1)
        
        # Return observation, reward, terminated, truncated, and info dictionary
        observation = np.append(self.ohlcv[self.current_step-5:self.current_step], [self.current_amount, self.current_asset_amount]).astype(np.float32)        
        truncated = False  # You'll need to define this based on your environment's logic
        info = {}  # Empty info dictionary
        if(done):
            print(f"Step: {self.current_step}, Action: {action}, Reward: {self.reward}, Cash: {self.current_amount}, Asset: {self.current_asset_amount} ")

        return observation, self.reward, done, truncated, info



    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f'Equity Value: {self.reward}')