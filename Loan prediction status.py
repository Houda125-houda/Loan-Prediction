# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 21:11:52 2021

@author: LENOVO
"""


#####                                     #####
#####      Project Loan prediction        #####
###############################################

# we are going to predict whether a person is eligible for a loan or not 
# let's start :) 
# we consider one finance company that gives loan for people so before approving the loan  this company analyzes various credential of the person 
# se there are several aspects whether the person is educated, graduated or not , married or single so many parameters so this company wants to automate this loan approval process
# so what happend, the person fill an online application form and based on the information given by the user we need to develop a machine learninh system that can tell the company of this person is elegiblefor this loan or not 


# we are going to use the SVM support vector mOdel which is a supervised learning model ==>  we will give the data which has labels one is : loan approved and the second is rejected


# import the dependencies = functions and librairies 
import pandas as pd
import numpy as np 
import seaborn as sns # plot librairy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score # it's used to evaluate our model it tells how well our model is perfoming on a given dataset 


# loading the data set 
Loan_data = pd.read_csv("C:/Users/LENOVO/Documents/GitHub/Loan-Prediction/dataset_loan.csv")
# copy 
# df = Loan_data.copy()
# Loan_data.info() ===> for types of data have to verify this ^^
type(Loan_data)  # the type of data   == > pandas.core.frame.DataFrame
Loan_data.head()
# the numbre of rows and column 
Loan_data.dtypes
Loan_data.shape
# statistical measures 
Loan_data.describe()
Loan_data["Property_Area"].value_counts()
# The number of missing values 
Loan_data.isna().sum()  # Loan_data.isnull().sum()  ===> in this case we dont have much 

################
#    Test 1    #
################

Loan_data.columns

var_cat = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area','Loan_Status']
var_num  = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term','CoapplicantIncome']
print ("Categorical features are ", var_cat)
print ("Numerical features are ", var_num)
# fillna() method on the dataframe is used for imputing missing values with mean, median, mode (most frequent) or constant value.

# replace the missing values in categorical features using mode "most frequent"
Loan_data['Gender'].fillna(Loan_data['Gender'].mode()[0], inplace = True)
Loan_data['Married'].fillna(Loan_data['Married'].mode()[0], inplace = True)
Loan_data['Dependents'].fillna(Loan_data['Dependents'].mode()[0], inplace = True)
Loan_data['Credit_History'].fillna(Loan_data['Credit_History'].mode()[0], inplace = True)
Loan_data['Self_Employed'].fillna(Loan_data['Self_Employed'].mode()[0], inplace = True)

# replace the missing values in numerical features using the median
Loan_data['LoanAmount'].fillna(Loan_data['LoanAmount'].median(), inplace = True)
Loan_data['Loan_Amount_Term'].fillna(Loan_data['Loan_Amount_Term'].median(), inplace = True)

# check the replaced values 
Loan_data.isna().sum()
# analysis of each variable "analyse univari??" we will start by the target  or dependent variable "Loan_status"
Loan_data['Loan_Status'].value_counts()
# with percentage using normalize so the values will be between 0 and 1 then we muptiply by 100 
Loan_data['Loan_Status'].value_counts(normalize = True)*100
Loan_data['Loan_Status'].value_counts(normalize = True).plot.bar(title ='Approved Loan or not')

# Let's start with categorical features 
# Gender , Married, Dependents, Credit_history, Self_employed
Loan_data['Gender'].value_counts() 
Loan_data['Gender'].value_counts(normalize = True)*100
Loan_data['Gender'].value_counts(normalize = True).plot.bar(title ='Sexe comparison')
# dependents : number of children 
Loan_data['Dependents'].value_counts() 
Loan_data['Dependents'].value_counts(normalize = True)*100

#############################
#    Numerical features     #
############################
Loan_data[var_num]
Loan_data.describe()

####         applicatIncome        ####
plt.figure(1)  # the first figure will be devided in 2 subplot 
plt.subplot(121)  # the first subplot 
sns.distplot(Loan_data['ApplicantIncome'])

plt.subplot(122) # the second subplot 
Loan_data['ApplicantIncome'].plot.box(figsize = (16,5))
plt.suptitle('')
plt.show()

###        CoapplicantIncome'  ####
plt.figure(1)  # the first figure will be devided in 2 subplot 
plt.subplot(121)  # the first subplot 
sns.distplot(Loan_data[ 'CoapplicantIncome'])

plt.subplot(122) # the second subplot 
Loan_data[ 'CoapplicantIncome'].plot.box(figsize = (16,5))
plt.suptitle('')
plt.show()


# bivari?? analysis
# we have 8 categorical variables 
fig,axes = plt.subplots(4,2,figsize = (12,15)) # 4 ==> rows & 2 ===> columns
for idx, cat_col in enumerate (var_cat):
    row, col = idx//2, idx%2
    sns.countplot(x= cat_col, data = Loan_data, hue = 'Loan_Status',ax = axes[row,col])
plt.subplots_adjust(hspace = 1)
    
#for i, j in enumerate(var_cat):   i is for the index of each value, and j is for the content or the values of each row
     #print(i,j)
     
# a visual for numerical variables 
matrix = Loan_data.corr()
f, ax = plt.subplots(figsize = (10,12))
sns.heatmap(matrix, vmax = 8, square = True, cmap = 'BuPu', annot = True) # cmap = colors 

####### ####### ####### ########
###      Model Creation     ####
####### ####### ####### ########
#### cat part ##
df_cat = Loan_data[var_cat]
df_cat= pd.get_dummies(df_cat, drop_first = True)


#### Num part ##
df_num = Loan_data[var_num]
# Concatenation of the two parts cat and num
df_encoded = pd.concat([df_cat, df_num], axis = 1 )

y = df_encoded['Loan_Status_Y']
X = df_encoded.drop('Loan_Status_Y', axis = 1)
   ########################
####       Training      ####
 ##########################
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state= 6)
clf = LogisticRegression()
## train the model nammed clf
clf.fit(X_train,y_train)
## Test the model 
pred = clf.predict(X_test)
# to compare the y_test and pred 
accuracy_score(y_test, pred)  ## 0.83 ==> to ameliorate the score we can test to drop some unnecessary columns or to normalize the whole values [0-1]

X.columns
# now we will use our model and test it 
profil_test = [[1,1,1,0,0,0,1,0,1,0,100,0,400,360]]
clf.predict(profil_test) # ===> array([1], dtype=uint8) it's a yes ^^

## Now we will save our  model using pickle
import pickle 
pickle.dump(clf, open('Loan_prediction.pkl', 'wb'))


################
#    Test 2    #
##  With SVM  ##
################
Classifier = svm.SVC(kernel = 'linear')
# Training the support vector machine 
Classifier.fit(X_train, y_train)

# Accuracy score on training data 
X_train_prediction = Classifier.predict(X_train)
# there is a concept in machine learnng which is overtraining means that the model train more on training data so in both accuracy_sores are almost the same in training and test so there is no overfitting 
training_data_accuracy = accuracy_score(X_train_prediction, y_train) # we compare the values predicted by our model with the original labels of Y_train 
print("the accuracy score on training data  is ", training_data_accuracy )



X_test_prediction = Classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test) 
print("the accuracy score on training data  is ", test_data_accuracy )


