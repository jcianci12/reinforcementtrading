import datetime
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from api import fetch_ohlcv_range, get_dates_and_data_from_latest_file
from plot_chart import plot_chart
from prep_data import prep_data
from save_model import save_model
# s,e,data = get_dates_and_data_from_latest_file(
from  config import *
from show_heatmap import show_heatmap
import seaborn as sns

#get the range
def fetch_range():
    end = datetime.datetime.now()    
    start = end - datetime.timedelta(0,300*100)
    return start,end
def get_data():
    s,e =fetch_range()
    df_new = fetch_ohlcv_range(
            s, e, "BTCUSDT", "5m", "spot")
    df_new = prep_data(df_new)
    return df_new

def prep_training_data(data):
        
    # Exclude 'time' column
    features = [col for col in data.columns if col != 'date']
    values = data[features].values

    # Your time series data
    time = np.array(range(len(values)))

    # Preprocess the data
    x_train = time
    y_train = values

    # Normalize the data
    x_train_norm = (x_train - np.mean(x_train)) / np.std(x_train)
    y_train_norm = (y_train - np.mean(y_train)) / np.std(y_train)
    # Normalize the data using z-score normalization
    # x_train_norm = (x_train - np.mean(x_train)) / np.std(x_train)
    # y_train_norm = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)

    return features,values,time,x_train,y_train,x_train_norm,y_train_norm





def predict_value(i,model,x_train,y_train):
    # Predict the next value
    x_predict = np.array([i])
    x_predict_norm = (x_predict - np.mean(x_train)) / np.std(x_train)
    y_predict_norm = model.predict(x_predict_norm)

    # De-normalize the predicted value
    y_predict = y_predict_norm * np.std(y_train) + np.mean(y_train)

    return y_predict


def predict(values,model,x_train,y_train):
    # Call the function for 10 moments beyond the training data and plot the results
    real_values = values.tolist()
    predicted_values = []
    predicted_times = list(range(0, len(values)+10))
    for i in predicted_times:
        predicted_value = predict_value(i,model,x_train,y_train)
        predicted_values.append(predicted_value[0][3])  # Flatten the predicted values
        print(f"The predicted value for day {i} is: {predicted_value}")
    return real_values,predicted_values,predicted_times

def get_existing_model():
    model = keras.models.load_model('Model.h5')
    return model
def init_model(features):
    # model = keras.Sequential([
    #         keras.layers.Dense(64, input_shape=[1]),
    #         keras.layers.LeakyReLU(),
    #         keras.layers.Dense(64),
    #         keras.layers.LeakyReLU(),
    #         keras.layers.Dense(units=len(features))
    #     ])
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=[1]),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(64),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(units=len(features))
    ])

        # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model
def train_model(model,x_train_norm,y_train_norm):
    m = model.fit(x_train_norm, y_train_norm, epochs=2000)
    save_model(model,"Model") 
    return model


def main():
    data = get_data()
    data = prep_data(data)
    show_heatmap(data)

    features,values,time,x_train,y_train,x_train_norm,y_train_norm = prep_training_data(data)

    try:
        # Try to load the existing model
        model = get_existing_model()
    except:
        # If the model does not exist, initialize a new one
        model = init_model(features)
        
    model = train_model(model,x_train_norm,y_train_norm)
    save_model(model,"model")
    real_values,predicted_values,predicted_times=predict(values,model,x_train,y_train)
    plot_chart(real_values,predicted_values,predicted_times,time)
    main()
        






main()