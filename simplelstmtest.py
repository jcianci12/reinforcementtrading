import os
import shutil
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras


# Define the number of rows
num_rows = 10

# Generate random OHLCV data
data = np.random.rand(num_rows, 5)

# Scale the price data (columns 0-3) to the range 25000-30000
data[:, :4] = data[:, :4] * (30000 - 25000) + 25000

# Create a date range
dates = pd.date_range(end='today', periods=num_rows)

# Create a DataFrame with the OHLCV data and dates
df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
df['Date'] = dates
# Convert the 'Date' column to timestamps (number of seconds since 1970-01-01)
df['Date'] = (df['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
data = df.values

# Print the DataFrame
print(df)

# Prepare data
n_past = 1
trainX, trainY = [], []
for i in range(n_past, len(data)):
    trainX.append(data[i - n_past:i])
    trainY.append(data[i][5])
trainX, trainY = np.array(trainX), np.array(trainY)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],  trainX.shape[2]))

# Define model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(50, activation='relu', input_shape=(n_past, trainX.shape[2])))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(trainX, trainY, epochs=200, verbose=0)

# Demonstrate prediction
# x_input = np.array([64])  # last value from your data
x_input = trainX.reshape((1, n_past,  trainX.shape[2]))
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


