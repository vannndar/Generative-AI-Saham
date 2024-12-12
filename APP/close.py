import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model

def predict_stock (url):
    # data = pd.read_csv(r'E:\Cool Yeah\Semester 5\RSBP\Generative-AI-Saham\APP\data\bbni_jk.csv')
    data = pd.read_csv(url)
    data['date_str'] = pd.to_datetime(data['date_str'])

    data = data.rename(columns={'date_str': 'date'})

    data_used = data.filter(['Close'])

    dataset = data_used.values

    training_data_len = int(np.ceil( len(dataset) * .80 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=5, epochs=30)

    test_data = scaled_data[training_data_len - 60: , :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    y_test = dataset[training_data_len:, :]
    test = predictions[:-1, 0]

    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    mae = mean_absolute_error(predictions, y_test)

    print("RMSE: ", rmse)
    print("MAE: ", mae)

    train = data[:training_data_len]
    # ambil data dari training_data_len sampai akhir -1
    valid = data[training_data_len-1:-1] 
    valid['Predictions'] = predictions[:, 0] 

    return train, valid, rmse, mae 


def predict_stock_only(url, model_path):
    data = pd.read_csv(url)
    data['date_str'] = pd.to_datetime(data['date_str'])

    data = data.rename(columns={'date_str': 'date'})

    data_used = data.filter(['Close'])

    dataset = data_used.values

    training_data_len = int(np.ceil( len(dataset) * .80 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)


    model = load_model(model_path)

    test_data = scaled_data[training_data_len - 60: , :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    y_test = dataset[training_data_len:, :]
    test = predictions[:-1, 0]

    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    mae = mean_absolute_error(predictions, y_test)

    print("RMSE: ", rmse)
    print("MAE: ", mae)

    train = data[:training_data_len]
    # ambil data dari training_data_len sampai akhir -1
    valid = data[training_data_len-1:-1] 
    valid['Predictions'] = predictions[:, 0] 

    return train, valid, rmse, mae

def create_plotly_figure(train, valid, rmse, mae):
    from plotly import graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['date'], y=train['Close'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=valid['date'], y=valid['Close'], mode='lines', name='Val'))
    fig.add_trace(go.Scatter(x=valid['date'], y=valid['Predictions'], mode='lines', name='Predictions'))
    fig.update_layout(title='Model',
                      xaxis_title='Date',
                      yaxis_title='Close Price Rupiah (Rp)')
    return fig
