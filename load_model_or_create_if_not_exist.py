import gymnasium
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Union
import os
from keras.models import Sequential
from keras.layers import Dense,Flatten
import  numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def load_model_or_create_if_not_exist(filename: str, env: Union[gymnasium.Env, DummyVecEnv]) -> PPO:
    # if False:

    if os.path.exists(filename):
        return PPO.load(filename)
    else:
        env_maker = lambda: env
        dummyenv = DummyVecEnv([env_maker])
        model = PPO('MlpPolicy', env, verbose=1,tensorboard_log="logdir")
    return model

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states.shape))  # Use the shape of the Gym space
    model.add(Dense(24, activation="relu"))
    model.add(Dense(np.prod(actions.shape), activation='linear'))  # Use the size of the Gym space    return model
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn
