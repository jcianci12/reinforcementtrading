import numpy as np
from matplotlib import pyplot as plt

from load_model_or_create_if_not_exist import load_model_or_create_if_not_exist


def evaluate(X_test,y_test,model,env):
    equity = [1]
    last_action = 0
    trading_cost = 0.01
    current_amount = 1000  # Initialize starting amount here
    current_asset_amount = 0  # Initialize starting amount here
    reward = 0
        # Create lists to store prices and actions for plotting
    prices = []
    actions = []
    for i in range(5, X_test.shape[0]):
        observation = np.append(X_test[i-5:i], [current_amount, current_asset_amount]).astype(np.float32)

        model = load_model_or_create_if_not_exist("model",env)
        action, _ = model.predict(observation, deterministic=True)
        print(f"choosing:{action} with data{observation}")
        currenty_test = y_test[i]

        amount = action[0]*1000  # Extract the amount from the action

        if amount > 0:  # Buying
            if amount * (1 + trading_cost) > current_amount:
                reward = -1  # Punishment
            else:
                change = (1 + currenty_test - trading_cost)
                current_amount -= amount * change
                current_asset_amount += amount
                reward =+ change
        elif amount < 0:  # Selling
            if abs(amount) > current_asset_amount:
                reward = -1  # Punishment
            else:
                change = (1 + -1 * currenty_test - trading_cost)
                current_amount -= amount * change  # Subtract because amount is negative
                current_asset_amount += amount  # Add because amount is negative
                reward =+ change
        elif amount == 0:  # Holding
            change = 1
            reward =+ change
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        new = equity[-1] * (1 + reward)  # Calculate new equity value based on the reward
        equity.append(new)
        last_action = action
        print(f"Step: {i}, Action: {amount}, Reward: {reward}, Cash: {current_amount}, Asset: {current_asset_amount}, Total: {current_amount+current_asset_amount} ")
                # Append price and action to the lists
        prices.append(currenty_test)
        actions.append(action[0])
                # After the loop, plot the price and action over time
    # After the loop, plot the price and action over time for the last 50 time steps
    plt.figure(figsize=(15, 10))
    plt.title("Price and Action Over Time (Last 50 Time Steps)")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.plot(prices[-50:], label='Price')
    plt.plot(actions[-50:], label='Action')
    plt.legend()
    plt.savefig("price_and_action_last_50")
    plt.close()
