from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras

# Your data
data = np.array([2, 4,8,16,32,64])

# Prepare data
n_past = 1
trainX, trainY = [], []
for i in range(n_past, len(data)):
    trainX.append(data[i - n_past:i])
    trainY.append(data[i])
trainX, trainY = np.array(trainX), np.array(trainY)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# Define model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(50, activation='relu', input_shape=(n_past, 1)))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(trainX, trainY, epochs=200, verbose=0)

# Demonstrate prediction
x_input = np.array([])  # last value from your data
x_input = x_input.reshape((1, n_past, 1))
yhat = model.predict(x_input)
print(f'Predicted value: {yhat[0][0]}')

# Plot the values
plt.plot(data, 'o-', label='Actual')
plt.plot(np.append(data[:-1], yhat[0]), 'x-', label='Predicted')
plt.legend()
plt.title('Actual vs Predicted')
plt.xlabel('Day')
plt.ylabel('Value')

# Save the plot
plt.savefig('lstm.png')
