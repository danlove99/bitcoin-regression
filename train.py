import numpy as np 
import pandas as pd 

df = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
df = df.fillna(1)
y = df['Weighted_Price'].values
df = df.drop('Weighted_Price', axis=1)
print(y.shape)
X = df.values
X = (X - X.min())/(X.max()-X.min()) * 20
#X = np.expand_dims(X, axis=2)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
X = X[:2099759:]
y = y[:2099759:]

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(7,1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
# fit model
print("started training")
model.fit(X, y, epochs=1, batch_size=1000)
#model.save('bitcoin.h5')
x_input = np.array([[2.00000000e+01],
                    [5.17876167e-05],
                    [5.17940812e-05],
                    [5.17876167e-05],
                    [5.17938226e-05],
                    [2.26618344e-08],
                    [9.07775444e-05]])

x_input = np.expand_dims(x_input, axis=0)
#should be around 4005
yhat = model.predict(x_input, verbose=0)
print(str(yhat))