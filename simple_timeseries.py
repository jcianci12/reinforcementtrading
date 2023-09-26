import numpy as np
import tensorflow as tf
from tensorflow import keras

# Your time series data
time = np.array(range(1, 21))
values = np.array([i*2 for i in range(1, 21)])

# Preprocess the data
x_train = time[:-1]
y_train = values[1:]

# Normalize the data
x_train_norm = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_norm = (y_train - np.mean(y_train)) / np.std(y_train)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(x_train_norm, y_train_norm, epochs=500)

# Predict the next value
x_predict = np.array([20])
x_predict_norm = (x_predict - np.mean(x_train)) / np.std(x_train)
y_predict_norm = model.predict(x_predict_norm)

# De-normalize the predicted value
y_predict = y_predict_norm * np.std(y_train) + np.mean(y_train)

print(f"The predicted value for day 20 is: {y_predict} | Actual:{values[:1]}")
