# Recurrent Neural Network

# Part 1 -Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#Getting the inputs and the outputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

#Reshaping
X_train = np.reshape(X_train, (1257, 1, 1) )

# Part 2 - Building the RNN
# Importing the Keras library and packages
import tensorflow as tf 

#Initialising the RNN
regressor = tf.keras.models.Sequential()

#Adding the input layer and the LSTM layer
regressor.add(tf.keras.layers.LSTM(units = 4, activation = 'sigmoid', input_shape=(None, 1)))

#Adding the output layer
regressor.add(tf.keras.layers.Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)


# Part 3 - Making the predictions and visualising the results

#Getting the real stock price of 2017
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:, 1:2].values

#Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))

predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title("Google stock price prediction")
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# Part 4 - Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))













