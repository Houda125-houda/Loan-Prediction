# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 20:08:55 2021

@author: LENOVO
"""

# Submarine have to know if the enemy country have planted some mines " explosives that explodes when some object comes in contact with it " in the ocean
# So the submarine need to predict whether it is crossing a mine or a rock 
# so Sonar  sends sound signals and review switchbacks " le fait de revenir " so this signal is then processed to detect whether the object is just a mine or a rock in the ocean 
# the sonar send and receive signals  bounce back " be returned" from metal cylindre and some rocks because mines will be made of metals
# we will use logistic regression model because it works very well for binary classification problem 

import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt #visualisation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score

# there is no header file " names for the columns" so we need to mention that 
sonar_data = pd.read_csv("C:/Users/LENOVO/Documents/GitHub/Rock-vs-Mine-prediction/Copy of sonar data.csv", header = None)
sonar_data.head()
# the last column is categorical value if it's a rock or mine 
sonar_data.shape
# statistical measures of data 
sonar_data.describe()
sonar_data[60].value_counts()
# we will group our data based on mine or rock 
sonar_data.groupby(60).mean()
# separating data and Labels 
X = sonar_data.drop(columns = 60, axis =1) # axis = 1 means we are dropping the column not the row 
y = sonar_data[60]

# stratify means that we need to have almost equal number of rocks in testing data and and equal number of mines in training data  so our data 
# will be splitted based on these two  
X_train, X_test, y_train , y_test = train_test_split(X,y,test_size= 0.1, stratify = y, random_state =1)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()
# training the logistic regression model with training data 
model.fit(X_train, y_train)

# accuracy on training data 
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print('Accuracy on training data : ', training_data_accuracy)

###############################################
#training accuracy is usually the accuracy you get if you apply the model on the training data, while testing accuracy is the accuracy for the testing data.

#It's sometimes useful to compare these to identify overtraining.
###############################################

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy on test data : ', test_data_accuracy)

# making a predictive system 
input_data = ()
# changing the input data to numpy array 
input_data_as_numpy_array = np.asarray(input_data)  # we are converting this into a numpy array because is faster and easy 
# we rechape the np array  as we are predictiong for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(-1,1)  #  reshape(-1,1) represents there is one instance and we are going to predict the label for this one instance 
prediction = model.predict(input_data_reshaped)
print(prediction)

 if(prediction[0] == 'R'):
    print("The object is a Rock!")
 else 
    print(" The object is a mine ")

        