# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 11:48:07 2014

@author: Olga
"""
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

from time import time
from class_vis import prettyPicture

#Importing data
# 4 columns: Grain Size, Max Annealling T, Max Loading + TARGET
data1 = np.genfromtxt('data-cryst-plane.csv',delimiter=',')


#Removing column
#data1 = np.delete(data1,0,1)

#Another way to remove multiple columns
#in the form np.delete(arr,[1,3,4],1)
#the list [1,3,4] carries the indicies of the columns of array arr that will be delited
data1 = np.delete(data1,[1,2,4],1)

#removing 'NaN'
n=data1[~np.isnan(data1).any(axis=1)]

#Separating data set (FEATURES) from TARGET
data1_target = n[:,-1]
data1_data = np.delete(n,-1,1)

mean = data1_data.mean(axis = 0)
std = data1_data.std(axis = 0)

data1_data = (data1_data - mean)/std

#Splitting the set and rescaling
X_train, X_test, y_train, y_test = train_test_split(data1_data,data1_target,test_size = 0.25, random_state = 33)

######################################
#Test on random array size 100, 3 columns + 1 random 0/1 column of TARGET
'''
test_arr_data = np.random.rand(100,3)
test_arr_target = np.random.randint(2, size=100)
X_train, X_test, y_train, y_test = train_test_split(test_arr_data,test_arr_target,test_size = 0.25, random_state = 33)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''
#######################################



#######################################
'''
SVM algorithm
'''
t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split=7)


clf.fit(X_train, y_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(X_test)
print "prediction time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(pred, y_test)
print "accuracy",accuracy
######################################
# to predict:
# print clf.predict(scaler,transform([20,850]))
######################################

colors = ['yellow','red']

for i in range(len(colors)):
    xs = X_train[:,0][y_train == i]
    ys = X_train[:,1][y_train == i]
    plt.scatter(xs,ys, c = colors[i])

plt.xlabel("Grain size")
plt.ylabel("Annealing time (min)")

    
######################################
    
#    
try:
    plt = plt.figure()
    prettyPicture(clf, X_train, y_train, X_test, y_test)
except NameError:
    pass
