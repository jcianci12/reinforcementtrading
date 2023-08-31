from stable_baselines import A2C


import os


def load_model_or_create_if_not_exist(filename, env):
    if os.path.exists(filename):
        return A2C.load(filename)
    else:
        return A2C('MlpLstmPolicy', env, verbose=1)