# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
#Only numpy arrays can be used as input in keras
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#.values makes converts it from dataframe to numpy array with one column
#[:,1:2] - All rows of index 1 column
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #Use default argument
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
#First dimension:batch_size
#second_dimension:time_steps
#third_dimension:Indicators in our case is only the Opening price
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
#unit:NUmber of neurons in each layer
#return_sequences:True if there is an LSTM layer following this
#input_shape:same as X_train
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
#Last layer will be fully connected layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
#We get the dataset from Jan 2012 to Jan 2017 by concatenating through stack
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

#We need to start the prediction from First financial day of Jan 2017
#(So we need opening prices from First financial day-60) to 2ndlst price
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

#To prevent warning
inputs = inputs.reshape(-1,1)

#We do not apply apply fit_transform method as it has already been fitted 
#and we need to apply the same scaling so we apply just transform method
inputs = sc.transform(inputs)

#The neural network requires the 3-D structure
#For the test_set we give the network list of 20 vectors each containing
#60 stock prices to predict the stock price of the next day
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])#
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)

#To inverse the scale the predictions as it will predict between 0-1
#but we need original predictions
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
