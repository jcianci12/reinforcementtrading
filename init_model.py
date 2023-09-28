from tensorflow import keras


def init_model(x_train,y_train):
    # model = keras.Sequential([
    #         keras.layers.Dense(64, input_shape=[1]),
    #         keras.layers.LeakyReLU(),
    #         keras.layers.Dense(64),
    #         keras.layers.LeakyReLU(),
    #         keras.layers.Dense(units=len(features))
    #     ])
    # model = keras.Sequential([
    #     keras.layers.Dense(64, input_shape=[1]),
    #     keras.layers.LeakyReLU(),
    #     keras.layers.Dense(64),
    #     keras.layers.LeakyReLU(),
    #     keras.layers.Dense(units=len(y_train.columns))
    # ])
    # Define the LSTM model
    inputs = keras.layers.Input(shape=(x_train.shape[0], y_train.shape[1]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
    model.summary()

        # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model