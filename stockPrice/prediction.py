import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stockID = '2330'
traindata = 'data/'+ stockID + '_2015_2019_ochlv.csv'
testdata = 'data/'+ stockID +'_202001_03_ochlv.csv'
print('traindata = ' + traindata)
print('testdata = ' + testdata)

dataNum = 5
timesteps = 20
epochNum = 200


dataset_train = pd.read_csv(traindata)
training_set = dataset_train.iloc[:,1:dataNum+1].values


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_train = []
Y_train = []
for i in range(timesteps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timesteps:i, 0:dataNum])    
    Y_train.append(training_set_scaled[i, 0])                        
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], dataNum))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()
regressor.add(LSTM(units = 32, input_shape = (X_train.shape[1], dataNum), return_sequences = True))
regressor.add(LSTM(units = 16))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.summary()


regressor.fit(X_train, Y_train, batch_size = 32, epochs = epochNum)


test_set = pd.read_csv(testdata)
real_stock_price = test_set.iloc[:,1:dataNum+1].values
lenOfReal = len(real_stock_price)
inputs = real_stock_price
inputs = sc.transform(inputs)

inputs_test = []
for i in range(timesteps, len(inputs)):
  inputs_test.append(inputs[i-timesteps:i, 0:dataNum])
inputs_test = np.array(inputs_test)
inputs_test = np.reshape(inputs_test, (inputs_test.shape[0], inputs_test.shape[1], dataNum))
predicted_stock_price = regressor.predict(inputs_test)
predicted_stock_price = np.pad(predicted_stock_price,((0,0),(0,dataNum-1)),'constant') 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price = np.delete(predicted_stock_price, [1, 2, 3, 4], axis=1)


real_stock_price = test_set.iloc[timesteps:lenOfReal+1,1:2].values
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
plt.savefig('pic1.png')


real_stock_price_train = pd.read_csv(traindata)
real_stock_price_train = real_stock_price_train.iloc[timesteps:len(real_stock_price_train)+1,1:2].values


predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = np.pad(predicted_stock_price_train,((0,0),(0,dataNum-1)),'constant')
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)
predicted_stock_price_train = np.delete(predicted_stock_price_train, [1, 2, 3, 4], axis=1)

np.savetxt(stockID + '.csv', predicted_stock_price, fmt="%.3f", delimiter=",")

plt.plot(real_stock_price_train, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price_train, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
plt.savefig('pic2.png')


import math
from sklearn.metrics import mean_squared_error

rmseTest = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print("RMSE_test = " + str(rmseTest))
rmseTrain = math.sqrt(mean_squared_error(real_stock_price_train, predicted_stock_price_train))
print("RMSE_train = " + str(rmseTrain))



