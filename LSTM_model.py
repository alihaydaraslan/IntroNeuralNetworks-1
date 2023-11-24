import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def predict_future(start, end, ticker, future_time_steps, time_steps):

    data = yf.download(ticker, start, end)

    stock_prices = data['Adj Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_prices_normalized = scaler.fit_transform(stock_prices)

    X, Y = create_dataset(stock_prices_normalized, time_steps)

    split_ratio = 0.8
    split_index = int(split_ratio * len(X))

    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(time_steps, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=50, batch_size=32)

    loss = model.evaluate(X_test, Y_test)
    print(f'Mean Squared Error on Test Data: {loss}')

    future_inputs = []
    last_data_points = stock_prices_normalized[-time_steps:]

    for _ in range(future_time_steps):
        new_input = np.array(last_data_points[-time_steps:]).reshape(1, time_steps, 1)
        future_inputs.append(new_input)

        next_prediction = model.predict(new_input)

        last_data_points = np.concatenate([last_data_points, next_prediction])

    future_inputs = np.array(future_inputs).reshape(-1, time_steps, 1)

    future_predictions = model.predict(future_inputs)

    future_predictions_original_scale = scaler.inverse_transform(future_predictions)

    # plt.figure(figsize=(10, 6))
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(stock_prices, label='Historical Prices')
    # plt.title('Historical Prices')
    # plt.legend()
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(range(len(stock_prices), len(stock_prices) + future_time_steps), future_predictions_original_scale, label='Future Predictions')
    # plt.title('Future Predictions')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    return {
        'historical_prices': stock_prices.tolist(),
        'predicted_prices': future_predictions_original_scale.tolist()
    }


def create_dataset(dataset, time_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_steps):
        a = dataset[i:(i+time_steps), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_steps, 0])
    return np.array(dataX), np.array(dataY)

