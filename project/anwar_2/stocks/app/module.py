# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:02:45 2022

@author: ayoon
"""

import yfinance as yf
import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from matplotlib import pyplot
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier


def Model_Input(start_time_str,end_time_str,stock_str):
    #start_time_str=input("Please enter the start date of the stock: YYYY-MM-DD ")
#end_time_str=input("Please enter the end date of the stock: YYYY-MM-DD ")
#stock_str=input("Please enter the listed stock symbol:")
# start_time_str="2012-01-03"
# end_time_str="2019-08-13"
# STOCK_str="GOOGL"
# print(start_time_str)
# print(end_time_str)
# print(stock_str)
   start_date=datetime.datetime.strptime(start_time_str,'%Y-%m-%d')
   end_date=datetime.datetime.strptime(end_time_str,'%Y-%m-%d')
   data = yf.download(stock_str, start=start_date, end=end_date)
   data["Close"]=pd.to_numeric(data.Close,errors='coerce')
   data = data.dropna()
   trainData = data.iloc[:,4:5].values
   sc = MinMaxScaler(feature_range=(0,1))
   trainData = sc.fit_transform(trainData)
   X_train = []
   Y_train = []
   for i in range (60,1149): 
       X_train.append(trainData[i-60:i,0]) 
       Y_train.append(trainData[i,0])

   X_train,Y_train = np.array(X_train),np.array(Y_train)
   X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) #adding the batch_size axis
   model = Sequential()

   model.add(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1)))
   model.add(Dropout(0.2))

   model.add(LSTM(units=100, return_sequences = True))
   model.add(Dropout(0.2))

   model.add(LSTM(units=100, return_sequences = True))
   model.add(Dropout(0.2))

   model.add(LSTM(units=100, return_sequences = False))
   model.add(Dropout(0.2))

   model.add(Dense(units =1))
   model.compile(optimizer='adam',loss="mean_squared_error")
   hist = model.fit(X_train, Y_train, epochs = 20, batch_size = 32, verbose=2)
   
   plt.plot(hist.history['loss'])
   plt.title('Training model loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train'], loc='upper left')
   plt.show()
   
   testData = data
   testData["Close"]=pd.to_numeric(testData.Close,errors='coerce')
   testData = testData.dropna()
   testData = testData.iloc[:,4:5]
   Y_test = testData.iloc[60:,0:].values 
   #input array for the model
   inputClosing = testData.iloc[:,0:].values 
   inputClosing_scaled = sc.transform(inputClosing)
   inputClosing_scaled.shape
   X_test = []
   length = len(testData)
   timestep = 60
   for i in range(timestep,length):  
       X_test.append(inputClosing_scaled[i-timestep:i,0])
   X_test = np.array(X_test)
   X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
   Y_pred = model.predict(X_test)
   
   predicted_price = sc.inverse_transform(Y_pred)
   plt.plot(Y_test, color = 'red', label = 'Actual Stock Price')
   plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
   plt.title( stock_str+' stock price prediction')
   plt.xlabel('Time')
   plt.ylabel('Stock Price')
   plt.legend()
   fig=plt.gcf()
   plt.show()
   return fig

    