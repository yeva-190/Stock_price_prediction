#!/usr/bin/env python
# coding: utf-8

# In[199]:


import pandas as pd
import numpy as np
import math 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import yfinance as yf 
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


# In[200]:


msft = yf.Ticker("MSFT")
msft_hist = msft.history(period="max")
msft = pd.DataFrame(msft_hist)


# In[201]:


#nio = yf.Ticker('NIO')
#nio_hist = nio.history(period="max")
#nio = pd.DataFrame(nio_hist)


# In[202]:


x = msft.index
y = msft['Close']


# ## EDA

# In[203]:


def df_plot(data, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[204]:


stock_name= "MSFT"
title = (stock_name,"History stock performance")
df_plot(msft , x , y , title=title,xlabel='Date', ylabel='Value',dpi=100)


# In[41]:


print(msft.isnull().sum())


# In[42]:


msft = msft.dropna()


# In[43]:


msft.drop_duplicates(keep=False, inplace=True)


# In[44]:


q1 = msft['Close'].quantile(0.25)
q3 = msft['Close'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
df = msft[msft['Close'] <= upper_bound]


# In[45]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
msft[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(msft[['Open', 'High', 'Low', 'Close', 'Volume']])


# In[99]:


msft = pd.read_csv('msft.csv')


# In[100]:


#print(msft.isnull().sum())


# In[101]:


msft['Date'] = pd.to_datetime(msft['Date'])


# In[49]:


print(msft.duplicated().sum())


# In[50]:


msft.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)


# In[51]:


q1 = msft['Close'].quantile(0.25)
q3 = msft['Close'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
df = msft[msft['Close'] <= upper_bound]


# In[ ]:





# In[52]:


scaler = StandardScaler()
msft[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(msft[['Open', 'High', 'Low', 'Close', 'Volume']])


# In[157]:


plt.figure(figsize=(15,6))
plt.plot(msft['Date'],
         msft['Open'],
         color="blue",
         label="open")
plt.plot(msft['Date'],
         msft['Close'],
         color="red",
         label="close")
plt.title("Microsoft Open-Close Stock")
plt.legend()


# In[158]:


plt.figure(figsize=(15,6))
msft['year'] = msft['Date'].dt.year
sns.boxplot(x='year', y='Close', data=msft)
plt.title('Closing Stock Prices by Year')
plt.xlabel('Year')
plt.ylabel('Closing Stock Price')
plt.xticks(rotation=75)
plt.show()


# In[159]:


plt.figure(figsize=(15, 6))
sns.heatmap(msft.corr(),
            annot=True,
            cbar=False)
plt.show()


# In[160]:


plt.figure(figsize=(15, 6))
sns.distplot(msft['Close'], kde=True)
plt.title('Distribution of Closing Stock Price')
plt.xlabel('Closing Stock Price')
plt.ylabel('Frequency')
plt.show()


# In[25]:


msft = yf.Ticker("MSFT")
msft_hist = msft.history(period="max")
msft = pd.DataFrame(msft_hist)


# In[205]:


msft['Open'].hist(bins=50, density=True, figsize=(15, 6))
plt.ylabel('Number of houses')
plt.xlabel('Sale Price ($)')
plt.title('Histogram of Opening Stock Price')
plt.show()


# In[206]:


plt.figure(figsize=(15, 6))
np.log(msft['Open']).hist(bins=50, density=True)
plt.ylabel('Number of houses')
plt.xlabel('Log of Price')
plt.title('Log of Opening Stock Price')
plt.show()


# In[ ]:





# ## Linear Regression

# In[169]:


x = msft.index
y = msft['Close']


# In[ ]:





# In[170]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[172]:


lr = LinearRegression()
lr.fit(x_train.to_numpy().reshape(-1, 1), y_train)
print("regression coefficient",regression.coef_)
print("regression intercept",regression.intercept_)


# In[175]:


regression_confidence = regression.score(x_test.to_numpy().reshape(-1,1), y_test)
print("linear regression confidence: ", regression_confidence)


# In[177]:


predicted=regression.predict(x_test.to_numpy().reshape(-1,1))


# In[188]:


predicted = predicted.reshape(-1)


# In[193]:


fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(x, y, label='True Values')
ax.plot(x_test, predicted, label='Predictions', color='orange')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Linear Regression: Predictions vs True Values')
ax.legend()

plt.show()


# ## This was just to showcase that Linear Regression is not a good idea for stock prices :D

# In[ ]:





# ## LSTM model

# In[67]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15 , shuffle=False,random_state = 0)


# In[70]:


from sklearn.preprocessing import MinMaxScaler


# In[71]:


close_prices = msft['Close']
values = close_prices.values
training_data_len = math.ceil(len(values)* 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))
train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[72]:


test_data = scaled_data[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[73]:


from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:





# In[74]:


model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()


# In[75]:


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 1, epochs=3)


# In[76]:


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# In[77]:


data = msft.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ## RNN
# 

# In[127]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout


# In[128]:


regressor = Sequential()


# In[129]:


regressor.add(
    SimpleRNN(units = 50, 
              activation = "tanh", 
              return_sequences = True, 
              input_shape = (x_train.shape[1],1))
             )

regressor.add(
    Dropout(0.2)
             )


# In[130]:


regressor.add(
    SimpleRNN(units = 50, 
              activation = "tanh", 
              return_sequences = True)
             )

regressor.add(
    Dropout(0.2)
             )


# In[131]:


regressor.add(
    SimpleRNN(units = 50, 
              activation = "tanh", 
              return_sequences = True)
             )

regressor.add(
    Dropout(0.2)
             )


# In[132]:


regressor.add(
    SimpleRNN(units = 50)
             )

regressor.add(
    Dropout(0.2)
             )


# In[133]:


regressor.add(Dense(units = 1))

regressor.compile(
    optimizer = "adam", 
    loss = "mean_squared_error",
    metrics = ["accuracy"])

history = regressor.fit(x_train, y_train, epochs = 50, batch_size = 32)


# In[134]:


history.history["loss"]


# In[ ]:





# In[135]:


plt.figure(figsize =(10,7))
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Simple RNN model, Loss vs Epoch")
plt.show()


# In[136]:


plt.figure(figsize =(10,5))
plt.plot(history.history["accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracies")
plt.title("Simple RNN model, Accuracy vs Epoch")
plt.show()


# In[137]:


y_pred = regressor.predict(x_train) 
y_pred = scaler.inverse_transform(y_pred)
y_pred.shape


# In[ ]:





# In[ ]:





# In[194]:


length_data = len(msft)    
split_ratio = 0.7          
length_train = round(length_data * split_ratio)  
length_validation = length_data - length_train
print("Data length :", length_data)
print("Train data length :", length_train)
print("Validation data lenth :", length_validation)


# In[ ]:





# In[ ]:





# ## Gradient Boosting Regression

# In[276]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb


# In[277]:


ticker = yf.Ticker("MSFT")
data = ticker.history(period="max")


# In[278]:


data = data.reset_index()
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
data = data.drop(columns=["Dividends", "Stock Splits"])
data = data.dropna()


X = data.drop(columns=["Close"])
y = data["Close"]


# In[ ]:





# In[255]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# In[284]:


gbm = GradientBoostingRegressor(n_estimators=50, learning_rate=0.2, max_depth=3, random_state=42)
gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[270]:


importances = xgb_model.feature_importances_
feature_names = data.columns

indices = importances.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(data.shape[1]), importances[indices])
plt.xticks(range(data.shape[1]), feature_names[indices], rotation=90)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




