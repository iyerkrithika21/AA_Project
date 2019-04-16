# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:44:15 2019

@author: iyerk
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from local_search_outlier import local_search_outliers
from local_search_outlier import array_set_diff
from local_search_outlier import calc_distances
# import some data to play with
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
n = np.random.randint(1,len(y),size=3)
initial_centers = X[n,:]

X_new = array_set_diff(X,initial_centers)

[centers,outliers] = local_search_outliers(X_new,3,10,initial_centers)

assigned_centers = calc_distances(X,centers,outliers)

print(classification_report(y,assigned_centers))
plt.scatter(X[:,0],X[:,1],c=assigned_centers)

