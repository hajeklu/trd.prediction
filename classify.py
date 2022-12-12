import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


def redOrGreen(open, close):
    if (open < close):
        return 1  # GREEN
    if (open >= close):
        return 0  # RED


def normalize_candles(candles):
    print("Scaling..")
    scaler = MinMaxScaler(feature_range=(0, 1))
    candles[['open', 'high', 'low', 'close']] = scaler.fit_transform(
        candles[['open', 'high', 'low', 'close']])
    print('Scaler min: ' + str(scaler.data_min_))
    print('Scaler max: ' + str(scaler.data_max_))
    return candles


def train(data, prediction_minutes):
    x_train = []
    y_train = []
    normalizedCandles = data[['open', 'high',
                              'low', 'close']].to_numpy(copy=True)
    for x in range(prediction_minutes, len(normalizedCandles)):
        xdata = normalizedCandles[x - prediction_minutes:x]
        candleY = [redOrGreen(normalizedCandles[x][0],
                              normalizedCandles[x][3])]
        x_train.append(xdata)
        y_train.append(candleY)

    print(x_train[0])
    print(y_train[0])
    print("Spliting..")
    # split train and test
    x_toSplit, y_toSplit = x_train, y_train
    sizeOf70percentage = int(len(x_toSplit)/.90)
    x_test = np.array(x_toSplit[sizeOf70percentage:len(x_toSplit)])
    y_test = np.array(y_toSplit[sizeOf70percentage:len(x_toSplit)])
    x_train = np.array(x_toSplit[0: sizeOf70percentage])
    y_train = np.array(y_toSplit[0: sizeOf70percentage])

    print("Total size of samples: " + str(len(x_train)))
    print("Inout shape: " + str(x_train.shape))

    print("Creatng model..")
    model = Sequential()
    model.add(Dense(units=128, activation='relu',
              input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=16)
    return model


def predictFromModel(candles, model):
    predicted_candle = model.predict(np.array([candles]))
    return predicted_candle


def call(input_json):
    df = pd.DataFrame(input_json)
    df['difference'] = df['open'] - df['close']
    df['difference_abs'] = np.abs(df['difference'])
    mean = df['difference_abs'].mean()
    df['direction'] = np.where(df['difference'] < 0, 0, 1)
    df['size'] = np.where(df['difference_abs'] < mean, 0, 1)
    df['timestamp'] = df['timestamp'].apply(pd.to_datetime)
    df['time_hour'] = df['timestamp'].dt.hour
    data = df[['open', 'high', 'low', 'close']]
    prediction_minutes = 5
    normalized_candles = normalize_candles(data)
    model = train(normalized_candles, prediction_minutes)
    data_to_predict = normalized_candles.tail(prediction_minutes)[['open', 'high',
                                                                   'low', 'close']]
    predictedDirection = predictFromModel(data_to_predict, model)
    return predictedDirection
