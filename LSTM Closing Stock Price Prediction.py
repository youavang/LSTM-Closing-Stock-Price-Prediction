#!/usr/bin/env python
# coding: utf-8

# This progream uses an artificial recurrent network called Long Short Term Mememory (LSTM) to predict the closing stock price of a corporation (Apple Inc.) using the past 60 day stock price. We will be analyzing the stock trend from 2012 to 2020-03-26 and train our model. The objective is gauge the accuracy of the LSTM model in predicting the closing stock price that will help us decide whether or not to invest in the stock market now.

# In[64]:


# Import libraries
import math
import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[65]:


# get the stock quote
df = web.DataReader('AAPL', data_source = 'yahoo', start = '2012-01-01', end = '2020-03-26')
df


# In[66]:


# Get the number of rows and column in the data set.
df.shape


# In[67]:


# Visualize the closing price history.

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()


# In[68]:


# Create a new dataframe with only the 'Close' column.
data=df.filter(['Close'])

#Convert the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*0.8)

training_data_len


# In[69]:


# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# In[70]:


# Create the scaled training data set.
train_data = scaled_data[0: training_data_len, :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i <= 60:
        print('x_train =', x_train)
        print('y_train =', y_train)


# In[71]:


# Conver the x_train and y_train to numpy arrays.
x_train, y_train = np.array(x_train), np.array(y_train)


# In[72]:


# Reshape the data to fit LSTM model requirement.
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[73]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[74]:


# Compile the model.
model.compile(optimizer='adam', loss='mean_squared_error')


# In[75]:


# Train the model.
model.fit(x_train, y_train, batch_size = 1, epochs = 1)


# In[76]:


# Create the testing data set
# Create a new array containing scaled values from the index 1543 to 2003.
test_data = scaled_data[training_data_len - 60: , :]

# Create the dat sets x_test and y_test.
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60: i, 0])


# In[77]:


# Convert the data to a numpy array.
x_test = np.array(x_test)


# In[78]:


# Reshape the data.
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape


# In[79]:


# Get the models predicted price values.
predictions = model.predict(x_test)

# Inverse transform the data
predictions = scaler.inverse_transform(predictions)


# In[80]:


# Get the root mean squared error (RMSE). The lower value the RMSE the better the fit.

rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# In[82]:


# Plot the data.
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,18))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[83]:


# Show the valid and predicted prices.
valid


# In[84]:


# Predict the closing price for Apple stock.
# Get the quote
apple_quote = web.DataReader('AAPL', data_source = 'yahoo', start = '2012-01-01', end = '2019-12-31')

# Create a new dataframe.
new_df = apple_quote.filter(['Close'])

# Get the last 60 day closing price values and convert the dataframe to an array.
last_60_days = new_df[-60:].values

# Scale the data to be values between 0 and 1.
last_60_days_scaled = scaler.transform(last_60_days)


# In[85]:


# Create and empty list and append the past 60 days.
X_test = []
X_test.append(last_60_days_scaled)

# Convert the X_test data set to a numpy array
X_test = np.array(X_test)

# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the predicted scaled price.
pred_price = model.predict(X_test)

# Undo the scaling.
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[86]:


# Get the quote
apple_quote2 = web.DataReader('AAPL', data_source = 'yahoo', start = '2020-03-01', end = '2020-03-26')
print(apple_quote2['Close'])


# The predicted closing price from the LSTM model is relatively similar to the actual closing price in recent days. Furthermore, the LSTM model prediction of AAPL closing price was similar in the trend of the actual closing price in the last year. In conclusion, the LSTM model is a good predictor of a stock closing price. The LSTM model should only be used to predict stock closing prices within a few days or a couple weeks.Predicting the closing stock price for next month or three months can be difficult at the moment due to many uncertainties that arrise with the pandemic spread of COVID-19. We can analyse the data and find trends of decreasing closing price and understand the cause of these trends.

# There are four decreasing closing price trend on the plot, in which those years are 2013, 2016, 2018, 2020. In 2013, there was Government shutdown that for 16 days due to Congressional budget dispute over the Patient Protection and Affordable Care Act. In 2016, there was the Presidential election and the Zika virus epidemic. Also, in 2018 there was another Government shutdown for 35 days disputing over funding for a barrier expansion onf the U.S.-Mexico border. The most recent decreasing trend in closing stock price is caused by the pandemic COVID-19. All of these events has a major impact in the US economy.

# By knowing our current events, we can make better predictions of the stock prices and the stock market for the near future. Usually the stock prices and the stock market picks up again once the economy is stable. Decreasing stock prices may not necessarily be bad, for instance, this may be the perfect time to buy more stocks since prices are low and hold on to them until the economy improves. The investment may be a risk, but may be worth it. 

# In[ ]:




