import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout,  LSTM


def normalize_candles(candles, scaler):
    candles[['open', 'high', 'low', 'close', 'avg_price', 'ohlc_price', 'oc_diff']] = scaler.fit_transform(
        candles[['open', 'high', 'low', 'close', 'avg_price', 'ohlc_price', 'oc_diff']])
    return candles


def redOrGreen(open, close):
    if (open < close):
        return 1  # GREEN
    if (open >= close):
        return 0  # RED


def redOrGreenFromNumber(number):
    if (number > 0.8):
        return "GREEN"
    if (number <= 0.3):
        return "RED"
    return "NEUTRAL"


def trainModel(trainCandles, prediction_minutes=60, model_name='lstm_1m_10_model'):
    tf.keras.backend.clear_session()
    # Prepare Data

    x_train = []
    y_train = []
    normalizedCandles = trainCandles[[
        'open', 'high', 'low', 'close', 'avg_price', 'ohlc_price', 'oc_diff']].to_numpy(copy=True)
    for x in range(prediction_minutes, len(normalizedCandles)):
        xdata = normalizedCandles[x-prediction_minutes:x]
        predictionData = []
        for candleX in xdata:
            predictionData.append(
                [candleX[0], candleX[1], candleX[2], candleX[3], candleX[4], candleX[5], candleX[6]])
        candleY = normalizedCandles[x]
        x_train.append(predictionData)
        y_train.append(redOrGreen(candleY[0], candleY[3]))

    # split train and test
    x_toSplit, y_toSplit = x_train, y_train
    sizeOf70percentage = int(len(x_toSplit)/.95)
    x_test = np.array(x_toSplit[sizeOf70percentage:len(x_toSplit)])
    y_test = np.array(y_toSplit[sizeOf70percentage:len(x_toSplit)])
    x_train = np.array(x_toSplit[0: sizeOf70percentage])
    y_train = np.array(y_toSplit[0: sizeOf70percentage])

    model = None
    model = Sequential()
    model.add(LSTM(units=10, return_sequences=True,
              input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='Adam', loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=16)

    return model


def predictFromModel(candles, model, scaler):
    predicted_candle = model.predict(np.array([candles]))
    return predicted_candle[0]


def addCountFeature(df):
    # Add additional features
    df['avg_price'] = (df['low'] + df['high']) / 2
    # df['range'] = df['high'] - df['low']
    df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close']) / 4
    df['oc_diff'] = df['open'] - df['close']
    return df


def main(candlesArray):
    candles = pd.DataFrame(candlesArray, columns=[
        'open', 'high', 'low', 'close'])
    candles = candles.dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    candles = addCountFeature(candles)
    normalized_all_candles = normalize_candles(candles, scaler)
    timestamp = datetime.now()
    MODEL_NAME = 'model-' + str(timestamp)
    prediction_unit = 40
    model = trainModel(normalized_all_candles, prediction_unit, MODEL_NAME)
    predictDirection = predictFromModel(
        candles.tail(prediction_unit), model, scaler)

    color = redOrGreenFromNumber(predictDirection)
    return [color, str(predictDirection[0])]
