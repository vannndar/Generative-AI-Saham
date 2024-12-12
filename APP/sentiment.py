import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model

def predict_sentiment(fileUrl, sentimentUrl):
    data = pd.read_csv(fileUrl)
    data['date_str'] = pd.to_datetime(data['date_str'])

    data = data.rename(columns={'date_str': 'date'})
    dataUsed = data[['Close', 'date']]
    dataUsed['date'] = pd.to_datetime(dataUsed['date'])
    dataUsed = dataUsed.set_index('date')

    dataset = dataUsed.values

    training_data_len = int(np.ceil( len(dataset) * .80 ))

    indonesian_weekdays = {
    "Senin": "Monday", "Selasa": "Tuesday", "Rabu": "Wednesday",
    "Kamis": "Thursday", "Jumat": "Friday", "Sabtu": "Saturday", "Minggu": "Sunday"
    }
    indonesian_months = {
        "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr", "Mei": "May",
        "Jun": "Jun", "Jul": "Jul", "Agu": "Aug", "Sep": "Sep", "Okt": "Oct",
        "Nov": "Nov", "Des": "Dec"
    }



    news = pd.read_csv(sentimentUrl)
    # delete columns except date and sentiment and title
    news = news[['publish_date', 'title', 'sentiment']]

    # Replace Indonesian weekdays and months with English equivalents
    for indo_day, eng_day in indonesian_weekdays.items():
        news['publish_date'] = news['publish_date'].str.replace(indo_day, eng_day, regex=False)
    for indo_month, eng_month in indonesian_months.items():
        news['publish_date'] = news['publish_date'].str.replace(indo_month, eng_month, regex=False)

    news = news.rename(columns={'publish_date': 'date'})
    news['date'] = pd.to_datetime(news['date'])
    news = news.set_index('date')

    news = news[news.index >= '2019-12-17']

    news.index = news.index.normalize()
    news = news.groupby(news.index).first()

    # Pastikan data terurut berdasarkan index
    news.index = news.index.normalize()

    # Merge DataFrame berdasarkan index
    dfMerge = dataUsed.merge(news, how='left', left_index=True, right_index=True)
    dataUsed = dataUsed.sort_index()
    news = news.sort_index()

    # Buat kolom tambahan untuk mencatat tanggal pertama di dataUsed setelah berita
    news['nearest_date'] = dataUsed.index.searchsorted(news.index, side='right')
    news['nearest_date'] = news['nearest_date'].map(lambda x: dataUsed.index[x] if x < len(dataUsed.index) else None)

    # Gabungkan dataUsed dengan berita berdasarkan nearest_date
    news_mapped = news.dropna(subset=['nearest_date']).set_index('nearest_date')
    dfMerge = dataUsed.merge(news_mapped, how='left', left_index=True, right_index=True)
    dfMerge_titles = dfMerge['title'].dropna()

    # Filter berita di news yang tidak ada di dfMerge berdasarkan 'title'
    missing_news = news[~news['title'].isin(dfMerge_titles)]

    dfMerge['sentiment'] = dfMerge['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1}).fillna(0)

    # Contoh data sentimen
    sentiment_data = dfMerge.filter(['sentiment']).values  # Pastikan ada kolom 'Sentiment'

    # Normalisasi data sentimen
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_sentiment = scaler.fit_transform(sentiment_data)

    scaled_data = dfMerge.filter(['Close']).values

    # Normalisasi data harga
    scaled_data = scaler.fit_transform(scaled_data)

    # Gabungkan harga (Close) dan sentimen
    scaled_combined_data = np.hstack((scaled_data, scaled_sentiment))

    train_data = scaled_combined_data[0:int(training_data_len), :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, :])  # Tambahkan semua fitur
        y_train.append(train_data[i, 0])  # Target tetap 'Close'

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))  # Sesuaikan fitur
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=5, epochs=30)

    test_data = scaled_combined_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, 0]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, :])  # Tambahkan semua fitur

    x_test = np.array(x_test)

    # Reshape data uji
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    start = len(predictions) - len(y_test)

    predictions = predictions[start:]

    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    mae = mean_absolute_error(predictions, y_test)

    print("RMSE: ", rmse)
    print("MAE: ", mae)

    train = data[:training_data_len]
    valid = data[training_data_len-1:-1]  
    valid['Predictions'] = predictions[:, 0] 

    return train, valid, rmse, mae


def predict_sentiment_model(url, sentimentUrl, model_path):
    # Load stock price data
    data = pd.read_csv(url)
    data['date_str'] = pd.to_datetime(data['date_str'])
    data = data.rename(columns={'date_str': 'date'})
    dataUsed = data[['Close', 'date']]
    dataUsed['date'] = pd.to_datetime(dataUsed['date'])
    dataUsed = dataUsed.set_index('date')

    # Load sentiment data
    news = pd.read_csv(sentimentUrl)
    news = news[['publish_date', 'title', 'sentiment']]

    # Replace Indonesian weekdays and months with English equivalents
    indonesian_weekdays = {
        "Senin": "Monday", "Selasa": "Tuesday", "Rabu": "Wednesday",
        "Kamis": "Thursday", "Jumat": "Friday", "Sabtu": "Saturday", "Minggu": "Sunday"
    }
    indonesian_months = {
        "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr", "Mei": "May",
        "Jun": "Jun", "Jul": "Jul", "Agu": "Aug", "Sep": "Sep", "Okt": "Oct",
        "Nov": "Nov", "Des": "Dec"
    }

    for indo_day, eng_day in indonesian_weekdays.items():
        news['publish_date'] = news['publish_date'].str.replace(indo_day, eng_day, regex=False)
    for indo_month, eng_month in indonesian_months.items():
        news['publish_date'] = news['publish_date'].str.replace(indo_month, eng_month, regex=False)

    news = news.rename(columns={'publish_date': 'date'})
    news['date'] = pd.to_datetime(news['date'])
    news = news.set_index('date')
    news = news[news.index >= '2019-12-17']
    news.index = news.index.normalize()
    news = news.groupby(news.index).first()

    # Merge sentiment data with stock price data
    dfMerge = dataUsed.merge(news, how='left', left_index=True, right_index=True)
    dfMerge['sentiment'] = dfMerge['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1}).fillna(0)

    # Extract and scale sentiment and price data
    sentiment_data = dfMerge[['sentiment']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_sentiment = scaler.fit_transform(sentiment_data)

    scaled_data = dfMerge[['Close']].values
    scaled_data = scaler.fit_transform(scaled_data)

    # Combine price and sentiment data
    scaled_combined_data = np.hstack((scaled_data, scaled_sentiment))

    # Load model
    model = load_model(model_path)

    # Prepare test data
    training_data_len = int(np.ceil(len(scaled_combined_data) * 0.80))
    test_data = scaled_combined_data[training_data_len - 60:, :]
    x_test = []

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, :])  # include both sentiment and price data

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))  # 2 features (price, sentiment)

    # Prepare true values for y_test
    y_test = dataUsed['Close'].values[training_data_len:]

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 1)))))[:, 0]  # inverse scaling only for price

    # Compute RMSE and MAE
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    mae = mean_absolute_error(predictions, y_test)

    print("RMSE: ", rmse)
    print("MAE: ", mae)

    # Split the data for visualization
    train = data[:training_data_len]
    valid = data[training_data_len-1:-1]
    valid['Predictions'] = predictions

    return train, valid, rmse, mae

