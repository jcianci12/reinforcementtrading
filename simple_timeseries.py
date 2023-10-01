import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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


scaler = StandardScaler()

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

def prep_training_data(df):
       
    #Separate dates for future plotting
    train_dates = pd.to_datetime(df['date'])
    print(train_dates.tail(15)) #Check last few dates. 

    #Variables for training
    cols = list(df)[1:6]
    #Date and volume columns are not used in training. 
    print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Close']

    #New dataframe with only training data - 5 columns
    df_for_training = df[cols].astype(float)

    # df_for_plot=df_for_training.tail(5000)
    # df_for_plot.plot.line()

    #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)


    #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
    #In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 

    #Empty lists to be populated using formatted training data
    trainX = []
    trainY = []

    n_future = 1   # Number of days we want to look into the future based on the past days.
    n_past = 14  # Number of past days we want to use to predict the future.

    #Reformat input data into a shape: (n_samples x timesteps x n_features)
    #In my example, my df_for_training_scaled has a shape (12823, 5)
    #12823 refers to the number of data points and 5 refers to the columns (multi-variables).
    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))


    return trainX,trainY





def predict_value(i,model,x_train,y_train):
    # Predict the next value
    x_predict = np.array([i])
    x_predict_norm = (x_predict - np.mean(x_train)) / np.std(x_train)
    y_predict_norm = model.predict(x_predict_norm)

    # De-normalize the predicted value
    y_predict = y_predict_norm * np.std(y_train) + np.mean(y_train)

    return y_predict


def predict(model,trainX,trainY):
    # Call the function for 10 moments beyond the training data and plot the results
    history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.savefig('train.png')
    plt.close()

    #Predicting...
    #Libraries that will help us extract only business days in the US.
    #Otherwise our dates would be wrong when we look back (or forward).  
    # from pandas.tseries.holiday import USFederalHolidayCalendar
    # from pandas.tseries.offsets import CustomBusinessDay
    # us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    #Remember that we can only predict one day in future as our model needs 5 variables
    #as inputs for prediction. We only have all 5 variables until the last day in our dataset.
    n_past = 16
    n_days_for_prediction=15  #let us predict past 15 days

    # predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()
    # print(predict_period_dates)

    #Make prediction
    prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction

    #Perform inverse transformation to rescale back to original range
    #Since we used 5 variables for transform, the inverse expects same dimensions
    #Therefore, let us copy our values 5 times and discard them after inverse transform
    prediction_copies = np.repeat(prediction, trainY.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
        
    df_forecast = pd.DataFrame({'Date':np.array(trainX), 'Open':y_pred_future})
    df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


    original = df[['Date', 'Open']]
    original['Date']=pd.to_datetime(original['Date'])
    original = original.loc[original['Date'] >= '2020-5-1']

    sns.lineplot(original['Date'], original['Open'])
    sns.lineplot(df_forecast['Date'], df_forecast['Open'])

    return model

def get_existing_model():
    model = keras.models.load_model('Model.h5')
    return model
def init_model(trainX,trainY):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(keras.layers.LSTM(32, activation='relu', return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

        # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model
def train_model(model,trainX,trainY):
    # m = model.fit(x_train_norm, y_train_norm, epochs=2000)
    history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

    save_model(model,"Model") 
    return model


def main():
    data = get_data()
    data = prep_data(data)
    show_heatmap(data)

    trainX,trainY = prep_training_data(data)

    try:
        # Try to load the existing model
        model = get_existing_model()
    except:
        # If the model does not exist, initialize a new one
        model = init_model(trainX,trainY)
        
    model = train_model(model,trainX,trainY)
    save_model(model,"model")

    model=predict(model,trainX,trainY)

    # plot_chart(real_values,predicted_values,predicted_times,time)
    # main()
        






main()