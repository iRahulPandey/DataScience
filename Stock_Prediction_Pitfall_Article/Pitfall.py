# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import Adam

# importing data into pandas dataframe
df = pd.read_csv('TSLA.csv', index_col=0, parse_dates=True, squeeze=True)
df.head(n=10)

# calculate returns by first shifting the data
df['PrevClose'] = df['Close'].shift(1)
df['Return'] = (df['Close'] - df['PrevClose']) / df['PrevClose']
df.head()

series = df['Close'].values.reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
series = scaler.transform(series).flatten()

T = 10 # number of days
D = 1  # default size
X = []
Y = []
for t in range(len(series) - T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1, T, 1) # Now the data should be N x T x D
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)

# implementing keras functional approach to build model
# input shape of (10,1) 
i = Input(shape=(T, 1))
x = LSTM(5)(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer='Adam',
)

# train the RNN
lstm_model = model.fit(
  X[:-N//2], Y[:-N//2],
  epochs=100,
  validation_data=(X[-N//2:], Y[-N//2:]),
)