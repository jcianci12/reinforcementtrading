from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv



import os


def load_model_or_create_if_not_exist(filename, env):


    if os.path.exists(filename):
        return A2C.load(filename)
    else:
        env_maker = lambda: env
        dummyenv = DummyVecEnv([env_maker])

        # Initializing and training the A2C model
        model = A2C('MlpPolicy', dummyenv, verbose=1)
        return model

