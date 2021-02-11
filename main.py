import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Asset_1 = pd.read_csv(r"EURUSD2019_2020.csv",header=0,index_col=0,parse_dates=True,na_values=99.99)
# Asset_2 = pd.read_csv(r"EURUSD2020_2021.csv")

Asset_1 = pd.read_csv(r"BTC_USD20192020.csv",header=0,index_col=0,parse_dates=True,na_values=99.99)
Asset_2 = pd.read_csv(r"BTC_USD20202021.csv",na_values=99.99)

prices_Asset = Asset_1[["Close"]]
prices_Asset = prices_Asset.fillna(method='ffill')

prices_Asset_2 = Asset_2[["Close"]]
prices_Asset_2 = prices_Asset_2.fillna(method='ffill')


# return_Asset = prices_Asset.iloc[1:]/prices_Asset.iloc[:-1].values-1
# # or
# # return_AAPL = prices_AAPL/prices_AAPL.shift(1)-1
# # or
# # prices_AAPL.pct_change()
#
# # print(return_AAPL.head(10))
#
#
# # return_AAPL.plot.bar()
# # plt.show()
#
# AnnualRetAAPL = (return_AAPL+1).prod()-1
#
# print("AAPL Annual Reuturn: " + str(AnnualRetAAPL))
#
# print("Monthly Volatility :" + str(return_AAPL.std()))
# print("Annual Volatility :" + str(return_AAPL.std()*np.sqrt(252)))
#
#
# wealth_index = 1000*(1+return_AAPL).cumprod()
#
# print(wealth_index.head())
#
# wealth_index.plot.line()
# plt.show()
#
# previous_peak = wealth_index.cummax()
# drawdown = (wealth_index - previous_peak)/previous_peak
#
# print(drawdown.idxmin())
#
# def drawdown(returns_series: pd.Series):
#     wealth_index = 1000*(1+returns_series).cumprod()
#     previous_peak = wealth_index.cummax()
#     drawdowns = (wealth_index-previous_peak)/previous_peak
#     return pd.DataFrame({
#         "Wealth": wealth_index,
#         "Peaks": previous_peak,
#         "Drawdown": drawdowns
#     })
#
# drawdown(return_AAPL["Close"])[["Wealth","Peaks"]].plot()
# plt.show()
#

def CloseSellPosition(ShortPrice, ClosePrice):
    Return = ShortPrice - ClosePrice
    return Return

def CloseBuyPosition(LongPrice, ClosePrice):
    Return = LongPrice - ClosePrice
    return Return

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

Asset_training_scaled = scaler.fit_transform(prices_Asset)

features_set = []
labels = []
for i in range(60, Asset_training_scaled.shape[0]):
    features_set.append(Asset_training_scaled[i-60:i, 0])
    labels.append(Asset_training_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

#######################################################################################################################

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])

history = model.fit(features_set, labels, epochs = 100, batch_size = 256,validation_split=0.2)

model.save("AAPL_E100_B256.h5")
########################################################################################################################

total_asset = pd.concat((prices_Asset, prices_Asset_2), axis=0)
test_inputs = total_asset[len(total_asset) - len(prices_Asset_2) - 60:].values

# test_inputs = prices_Asset_2[len(Asset_2) - 30:].values

test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)

test_features = []
for i in range(60, test_inputs.shape[0]):
    test_features.append(test_inputs[i-60:i, 0])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

predictions = model.predict(test_features)

# # print(predictions)
#
predictions = scaler.inverse_transform(predictions)
#
# print(tf.keras.losses.MSE(prices_Asset_2, predictions))
# mae = tf.keras.losses.MeanAbsoluteError()
# mae(prices_Asset_2, predictions).numpy()

# print(mae)
#
results = model.evaluate(test_features,prices_Asset_2.values)
#
print("test loss, test acc:", results)
print(history.history['accuracy'])


from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.metrics import explained_variance_score


MSE = mean_squared_error(y_true = prices_Asset_2.values, y_pred = predictions)

print("Mean Squared Error: " + str(MSE))
print("Root Mean Squared Error: "+ str(MSE**0.5))


plt.figure(figsize=(10,6))
plt.plot(prices_Asset_2, color='blue', label='Actual EUR/USD Price')
plt.plot(predictions , color='red', label='Predicted EUR/USD Price')
plt.title('EUR/USD Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()