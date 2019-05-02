# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:44:15 2019

@author: iyerk
"""

def precision_recall(outliers,new_outliers):
    
    nrows, ncols = outliers.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [outliers.dtype]}
    C = np.intersect1d(outliers.view(dtype), new_outliers.view(dtype))
    true_positive = len(C)
    precision = true_positive/len(outliers)
    recall = true_positive/len(new_outliers)
    
    print("Precision is   " + str(precision) + "   Recall is   " +str(recall))
    return precision,recall

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from local_search_outlier_functions import local_search_outliers
from local_search_outlier_functions import calc_distances
from sklearn.decomposition import PCA
from local_search_outlier_functions import array_row_intersection
# import some data to play with
iris = datasets.load_iris()
X = iris.data  
y = iris.target
X_new = X[y!=2]
y_new = y[y!=2]
plt.scatter(X_new[:,0],X_new[:,1],c=y_new)
k=2
z=50
np.random.seed(45)
n = np.random.randint(1,len(y),size=k)
initial_centers = X[n,:]
outlier = X[y==2]
#mean_vector = np.zeros(shape=(1,X.shape[1]))
#outlier = np.random.normal(mean_vector,np.max(X,axis=0)+np.random.randint(low=-5,high=1,size=X.shape[1]),(z,X.shape[1]))
plt.plot(outlier[:,0],outlier[:,1],"1")
plt.legend(["clusters","outliers"])
plt.show()
X_new = np.vstack((X,outlier))

#Local search with outliers
new_centers,new_outliers = local_search_outliers(X,k,z,initial_centers)
precision,recall = precision_recall(outlier,new_outliers)
