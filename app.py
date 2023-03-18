import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

def get_exchange_data():
    data = yf.download("EURCAD=X", start="2000-01-01", end="2023-03-18")
    return data

def train_arima_model(data):
    arima_model = auto_arima(data, seasonal=True, m=12, trace=True, suppress_warnings=True)
    predictions = arima_model.predict(n_periods=5)
    return np.array(predictions)  # Ajoutez cette conversion


def train_lstm_model(data):
    dataset = data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 7  # Augmenter la taille de la fenêtre pour améliorer les prédictions
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))  # Augmenter le nombre d'unités LSTM
    model.add(Dropout(0.2))  # Ajouter un dropout pour éviter le surapprentissage
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Ajouter une fonction d'arrêt anticipé pour éviter le surapprentissage

    model.fit(trainX, trainY, epochs=200, batch_size=32, verbose=2, validation_split=0.2, callbacks=[early_stopping])

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    future_steps = 5
    last_known = dataset[-look_back:]
    predictions = []

    for i in range(future_steps):
        prediction = model.predict(last_known.reshape(1, 1, look_back))
        predictions.append(prediction[0][0])
        last_known = np.append(last_known[1:], prediction)
        last_known = last_known.reshape(-1, 1)

    return predictions

st.title("Prédictions des taux de change EUR/CAD")
exchange_data = get_exchange_data()
arima_predictions = train_arima_model(exchange_data['Close'])
lstm_predictions = train_lstm_model(exchange_data['Close'])
lstm_predictions = np.array(lstm_predictions).reshape(-1)
st.write("Prédictions ARIMA:")
st.write(arima_predictions)

st.write("Prédictions LSTM:")
st.write(lstm_predictions)

# Trouver le meilleur jour pour investir selon les prédictions
best_day_arima = np.argmin(arima_predictions)
best_day_lstm = np.argmin(lstm_predictions)

# Afficher les graphiques
import plotly.graph_objs as go

arima_chart = go.Scatter(x=pd.date_range(start='2023-03-19', periods=5, inclusive='left'),
                         y=arima_predictions,
                         mode='lines+markers',
                         name='Prédictions ARIMA')

lstm_chart = go.Scatter(x=pd.date_range(start='2023-03-19', periods=5, inclusive='left'),
                        y=lstm_predictions.reshape(-1),
                        mode='lines+markers',
                        name='Prédictions LSTM')

best_day_arima_chart = go.Scatter(x=[pd.date_range(start='2023-03-19', periods=5, inclusive='left')[best_day_arima]],
                                  y=[arima_predictions[best_day_arima]],
                                  mode='markers',
                                  marker=dict(size=10, color='red'),
                                  name="Meilleur jour ARIMA")

best_day_lstm_chart = go.Scatter(x=[pd.date_range(start='2023-03-19', periods=5, inclusive='left')[best_day_lstm]],
                                 y=[lstm_predictions[best_day_lstm]],
                                 mode='markers',
                                 marker=dict(size=10, color='green'),
                                 name="Meilleur jour LSTM")

st.plotly_chart(go.Figure([arima_chart, lstm_chart, best_day_arima_chart, best_day_lstm_chart]))

st.write(f"Meilleur jour pour investir selon ARIMA: {pd.date_range(start='2023-03-19', periods=5, inclusive='left')[best_day_arima]}")
st.write(f"Meilleur jour pour investir selon LSTM: {pd.date_range(start='2023-03-19', periods=5, inclusive='left')[best_day_lstm]}")

st.line_chart(exchange_data['Close'])


   
