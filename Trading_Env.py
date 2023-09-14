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
        self.last_action = 0

        # Define action and observation space
        # Example when using discrete actions, we have 3: LONG, SHORT and FLAT.
        n_actions = 3
        self.action_space = gym.spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent, shape for (5 previous data poionts, OHLCV)
        dfwidth = ohlcv.shape[1]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, ohlcv.shape[1]), dtype=np.float64)

        startingamountusdt = 1000

    def reset(self ,seed=None):
        # Reset the number of step the training has taken
        self.current_step = 5
        self.reward = 1
        # Reset the last action
        self.last_action = 0
        # must return np.array type
        obs = self.ohlcv[self.current_step-5:self.current_step].astype(np.float32)
        info = {}  # Empty info dictionary
        return obs, info
    

    def step(self, action):
        if action == self.LONG:
            self.reward *= np.log1p(1 + self.ret[self.current_step] - (self.trading_cost if self.last_action != action else 0))
            # print (f"action was {action} and the reward is now:{self.reward} " )
        elif action == self.SHORT:
            self.reward *= np.log1p(1 + -1 * self.ret[self.current_step] - (self.trading_cost if self.last_action != action else 0))
            # print (f"action was {action} and the reward is now:{self.reward} ")
        elif action == self.FLAT:
            self.reward *= np.log1p(1 - (self.trading_cost if self.last_action != action else 0))
            # print (f"action was {action} and the reward is now:{self.reward} ")
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        # print(f"Step: {self.current_step}, Action: {action}, Reward: {self.reward}")
        self.last_action = action
        self.current_step += 1

        # Have we iterate all data points?
        done = (self.current_step == self.ret.shape[0]-1)
        
        # Return observation, reward, terminated, truncated, and info dictionary
        observation = self.ohlcv[self.current_step-5:self.current_step].astype(np.float32)
        truncated = False  # You'll need to define this based on your environment's logic
        info = {}  # Empty info dictionary
        return observation, self.reward, done, truncated, info



    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f'Equity Value: {self.reward}')