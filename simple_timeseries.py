import datetime
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from api import fetch_ohlcv_range, get_dates_and_data_from_latest_file
from save_model import save_model
# s,e,data = get_dates_and_data_from_latest_file()

#get the range
def fetch_range():
    end = datetime.datetime.now()    
    start = end - datetime.timedelta(0,300*100)
    return start,end
s,e =fetch_range()
df_new = fetch_ohlcv_range(
        s, e, "BTCUSDT", "5m", "spot")

# Exclude 'time' column
features = [col for col in df_new.columns if col != 'date']
values = df_new[features].values

# Your time series data
time = np.array(range(len(values)))

# Preprocess the data
x_train = time
y_train = values

# Normalize the data
x_train_norm = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_norm = (y_train - np.mean(y_train)) / np.std(y_train)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(units=len(features), input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')


model = keras.models.load_model('Model.h5')
# Train the model
# m = model.fit(x_train_norm, y_train_norm, epochs=500)
save_model(model,"Model") 

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
predicted_times = list(range(0, len(values)+10))
for i in predicted_times:
    predicted_value = predict_value(i)
    predicted_values.append(predicted_value[0][0])  # Flatten the predicted values
    print(f"The predicted value for day {i} is: {predicted_value}")

# Plot real and predicted values
plt.plot(time, np.array( real_values)[:,0], color='blue', label='Real Values')
plt.plot(predicted_times, predicted_values, color='green', label='Predicted Values')
plt.legend()
plt.savefig("plot.png")
plt.close()
