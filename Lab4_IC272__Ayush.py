# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:41:03 2022

@author: Ayush
"""

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

#Question 1
print("Question 1")


data = pd.read_csv('C:\\Users\\Ayush\\OneDrive\\Desktop\\LAB4-2022\\SteelPlateFaults-2class.csv')



Y = data['Class']
X = data.copy()


[X_train, X_test, Y_train, Y_test] = train_test_split(
    X, Y, test_size=0.3, random_state=42, shuffle=True)


X_train.to_csv('SteelPlateFaults-train.csv')
X_test.to_csv('SteelPlateFaults-test.csv')



X_train = X_train.drop('Class', axis=1)
X_test = X_test.drop('Class', axis=1)





highest_accuracy_Q1_K = 1
highest_accuracy_Q1_value = 0


def KNN_Q1(K):
    global highest_accuracy_Q1_K
    global highest_accuracy_Q1_value

    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(X_train, Y_train)
    Y_predict = neigh.predict(X_test)

    print("Confusion Matrix for K =", K)
    cfm = confusion_matrix(Y_test.to_numpy(), Y_predict)
    print(cfm)
    print()

    cay = accuracy_score(Y_test.to_numpy(), Y_predict)
    print("Classification Accuracy for K =", K, " is ", cay)
    print()

    if cay > highest_accuracy_Q1_value:
        highest_accuracy_Q1_K = K
        highest_accuracy_Q1_value = cay


KNN_Q1(1)
KNN_Q1(3)
KNN_Q1(5)
print("Highest Accuracy K is: ", highest_accuracy_Q1_K)
print("Highest Accuracy value is: ", highest_accuracy_Q1_value)
print()

#Question 2

max_dict = X_train.max()
min_dict = X_train.min()


diff = max_dict - min_dict

df2_train = pd.DataFrame()
df2_test = pd.DataFrame()

for col in X_train:
    if diff[col] != 0:
        df2_train[col] = (X_train[col] - min_dict[col]) / diff[col]
    else:
        df2_train[col] = X_train[col]


for col in X_test:
    if diff[col] != 0:
        df2_test[col] = (X_test[col] - min_dict[col]) / diff[col]
    else:
        df2_test[col] = X_test[col]

df2_train.to_csv('SteelPlateFaults-train-Normalised.csv')
df2_test.to_csv('SteelPlateFaults-test-normalised.csv')


df2_test[df2_test.isin([np.nan, np.inf, -np.inf]).any(1)]


highest_accuracy_Q2_K = 1
highest_accuracy_Q2_value = 0


def KNN_Q2(K):
    global highest_accuracy_Q2_K
    global highest_accuracy_Q2_value

    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(df2_train, Y_train)
    Y_predict = neigh.predict(df2_test)

    
    print("Confusion Matrix for K =", K)
    cfm = confusion_matrix(Y_test.to_numpy(), Y_predict)
    print(cfm)
    print()

    cay = accuracy_score(Y_test.to_numpy(), Y_predict)
    print("Classification Accuracy for K =", K, " is ", cay)
    print()

    if cay > highest_accuracy_Q2_value:
        highest_accuracy_Q2_K = K
        highest_accuracy_Q2_value = cay


KNN_Q2(1)
KNN_Q2(3)
KNN_Q2(5)

print("Highest Accuracy K is: ", highest_accuracy_Q2_K)
print("Highest Accuracy value is: ", highest_accuracy_Q2_value)
print()


#Question 3 

print("Question 3")

train = pd.read_csv('SteelPlateFaults-train.csv')
x_test = pd.read_csv('SteelPlateFaults-test.csv')
x_test = x_test.drop('TypeOfSteel_A300',axis = 1)
x_test = x_test.drop('TypeOfSteel_A400',axis = 1)
x_test = x_test.drop('X_Minimum',axis = 1)
x_test = x_test.drop('Y_Minimum',axis = 1)
train = train.drop('TypeOfSteel_A300',axis = 1)
train = train.drop('TypeOfSteel_A400',axis = 1)
train = train.drop('X_Minimum',axis = 1)
train = train.drop('Y_Minimum',axis = 1)
train = train[train.columns[1:]]
x_test = x_test[x_test.columns[1:]]
x_test = x_test[x_test.columns[:-1]]


train0 = train[train["Class"] == 0]
train1 = train[train["Class"] == 1]
xtrain0 = train0[train0.columns[:-1]]
xtrain1 = train1[train1.columns[:-1]]

cov0 = np.cov(xtrain0.T)
cov1 = np.cov(xtrain1.T)

mean0 = np.mean(xtrain0)
mean1 = np.mean(xtrain1)

cov0



def likelihood(xval, mval, covmat):
    '''
    Finding the likelihood using the formula
    '''
    myMat = np.dot((xval-mval).T, np.linalg.inv(covmat))
    inside = -0.5*np.dot(myMat, (xval-mval))
    ex = np.exp(inside)
    du = ((2*np.pi)*5 * (np.linalg.det(covmat))*0.5)
    return(ex/du)


prior0 = len(train0)/len(train)
prior1 = len(train1)/len(train)

predict = []
for i, row in x_test.iterrows():
    p0 = likelihood(row, mean0, cov0) * prior0
    p1 = likelihood(row, mean1, cov1) * prior1
    if p0 > p1:
        predict.append(0)
    else:
        predict.append(1)

bayes_accuracy = accuracy_score(Y_test, predict)
print()
print("These are for Bayes classifier")
print("Confusion Matrix => \n", confusion_matrix(Y_test, predict))
print("Accuracy Score => ", bayes_accuracy)
print()


print("Question 4")
print("KNN")
print(highest_accuracy_Q1_value)
print()
print("KNN Normalised")
print(highest_accuracy_Q2_value)
print()
print("Bayes Classifier")
print(bayes_accuracy)
print()
print('Highest accuracy is achieved in KNN Normalised: ',highest_accuracy_Q2_value)
print()