import numpy as np
from matplotlib import pyplot as plt


def plot_chart(real_values,predicted_values,predicted_times,time):
    # Plot real and predicted values
    plt.plot(time, np.array( real_values)[:,0], color='blue', label='Real Values')
    plt.plot(predicted_times, predicted_values, color='green', label='Predicted Values')
    plt.legend()
    plt.savefig("plot.png")
    plt.close()