import gymnasium
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Union
import os


def load_model_or_create_if_not_exist(filename: str, env: Union[gymnasium.Env, DummyVecEnv]) -> PPO:
    if False:

    # if os.path.exists(filename):
        return PPO.load(filename)
    else:
        env_maker = lambda: env
        dummyenv = DummyVecEnv([env_maker])
        model = PPO('MlpPolicy', env, verbose=1,tensorboard_log="logdir")
    return model

