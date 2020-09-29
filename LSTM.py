import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
numpy.random.seed(7)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X=pandas.read_csv('../data/train.csv',usecols=[0,1,2,3,4])
Y=pandas.read_csv('../data/train.csv',usecols=[5])
X = scaler.fit_transform(X)
Y = scaler_y.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test= numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
modelLSTM = Sequential()
nooffeatures = 5
modelLSTM.add(LSTM(50, input_shape=(1,nooffeatures), return_sequences=True))
modelLSTM.add(LSTM(units=50, return_sequences=True))
modelLSTM.add(LSTM(units=50))
modelLSTM.add(Dense(1))
modelLSTM.compile(loss='mean_squared_error', optimizer='adam')
modelLSTM.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
train_Predict = modelLSTM.predict(X_train)
test_Predict = modelLSTM.predict(X_test)
# invert predictions
trainScore = math.sqrt(mean_squared_error(y_train, train_Predict))
print('Train Score: %.2f RMSE' % (trainScore))
train_Predict = scaler_y.inverse_transform(train_Predict)
print('mean of train=',numpy.mean(y_train))
testScore = math.sqrt(mean_squared_error(y_test, test_Predict))
print('Test Score: %.2f RMSE' % (testScore))
print('mean of test=',numpy.mean(y_test)) 
train_Y = scaler_y.inverse_transform(y_train)
test_Predict = list(scaler_y.inverse_transform(test_Predict))
test_Y = list(scaler_y.inverse_transform(y_test))
train_Predict = list(scaler_y.inverse_transform(train_Predict))
test_Predict = [i[0] for i in test_Predict]
test_Y = [i[0] for i in test_Y]
N=len(test_Y)
plt.bar(numpy.arange(N), test_Predict - min(test_Predict), 0.35, label = 'Predicted Values')
plt.bar(numpy.arange(N)+0.35, test_Y - min(test_Y), 0.35, label = 'Actual Values')
plt.legend()
plt.show()
plt.plot(numpy.arange(N), test_Predict - min(test_Predict), 0.35, label = 'Predicted Values', color = 'red')
plt.plot(numpy.arange(N)+0.35, test_Y - min(test_Y), 0.35, label = 'Actual Values', color = 'green')
plt.ylabel(" epoch time ")
plt.xlabel(" test users ")
plt.legend()
plt.show() 
