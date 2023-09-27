from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

s,e = 0,41
# Your time series data
time = np.array(range(s, e))
values = np.array([i*2 for i in range(s, e)])

# Preprocess the data
x_train = time
y_train = values

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



def predict_value(i):
    # Predict the next value
    x_predict = np.array([i])
    x_predict_norm = (x_predict - np.mean(x_train)) / np.std(x_train)
    y_predict_norm = model.predict(x_predict_norm)

    # De-normalize the predicted value
    y_predict = y_predict_norm * np.std(y_train) + np.mean(y_train)

    return y_predict

# Call the function for 10 moments beyond the training data and plot the results
real_values = values.tolist()
predicted_values = []
predicted_times = list(range(e, e+10))
for i in predicted_times:
    predicted_value = predict_value(i)
    predicted_values.append(predicted_value[0][0])  # Flatten the predicted values
    print(f"The predicted value for day {i} is: {predicted_value}")

# Plot real and predicted values
plt.plot(time, real_values, color='blue', label='Real Values')
plt.plot(predicted_times, predicted_values, color='green', label='Predicted Values')
plt.legend()
plt.savefig("plot.png")
plt.close()
