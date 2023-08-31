from stable_baselines3 import A2C


import os


def load_model_or_create_if_not_exist(filename, env):
    if os.path.exists(filename):
        return A2C.load(filename)
    else:
        model = A2C('MlpPolicy', env, verbose=1)