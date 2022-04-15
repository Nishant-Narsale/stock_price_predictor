import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import datetime
import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler

company = "AAPL" #ticker symbol for a company (e.g. FB for Facebook)
start = datetime.datetime(2012, 1, 1) #starting date for dataset
end = datetime.datetime(2022, 1, 1) #ending date for dataset

df = web.DataReader(company, 'yahoo', start, end) #fetching data from yahoo finance api

#Scaling all data in 0 to 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1,1)) #reshape is necessary because fit_transform only accept 2d array

df["Close"].values.reshape(-1,1)

pred_days = 60 #number of days you want to predict, to measure accuracy of model

X_train, y_train = [], []

#As we are gonna use LSTM and sequential modelling for prediction,
#we will provide first 60 closing price in first input of X_train
#and this first input(i.e, first array of 60 days closing price) will have 61th day's closing price as it's corresponding output 
#then next input will be 2nd day closing price to 61th day closing price and it's corresponding output will be 62th day's closing price

for x in range(pred_days, len(scaled_data)):
    X_train.append(scaled_data[x-pred_days:x,0])
    y_train.append(scaled_data[x,0])
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Building neural network model

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) #prediction of next day's closing price

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=25, batch_size=32)

model.save(f'{company}_stock_price_predictor.h5', save_format="h5")
