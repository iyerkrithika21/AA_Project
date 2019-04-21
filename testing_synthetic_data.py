# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:56:30 2019

@author: iyerk
"""

import matplotlib.pyplot as plt
import numpy as np

from local_search_outlier_functions import local_search_outliers
from local_search_outlier_functions import calc_distances
from synthetic_data import generate_synthetic_data
from local_search_outlier_functions import find_outliers

k = 4
z = 20
dim = 2

centers,dataset,added_outliers = generate_synthetic_data(k,z,dim,200)


clusters = calc_distances(dataset,centers,z)
clusters = clusters.reshape((len(clusters),1))
true_outliers = find_outliers(dataset,centers,clusters,z)
plt.figure(figsize=(10,10))
plt.plot(dataset[:,0],dataset[:,1],'+',markersize=1)
plt.plot(added_outliers[:,0],added_outliers[:,1],'go')
plt.plot(true_outliers[:,0],true_outliers[:,1],'r*',markersize=2)
#plt.plot(added_outliers[:,0],added_outliers[:,1],'go')
plt.show()

dataset = np.vstack((dataset,centers))

n = np.random.randint(1,len(dataset),k)
initial_centers = np.ascontiguousarray(dataset[n,:])

new_centers,new_outliers = local_search_outliers(dataset,k,z,initial_centers)


plt.figure(figsize=(10,10))
plt.plot(dataset[:,0],dataset[:,1],'1')
plt.plot(new_outliers[:,0],new_outliers[:,1],'r+')
plt.plot(true_outliers[:,0],true_outliers[:,1],'b*')
#plt.plot(added_outliers[:,0],added_outliers[:,1],'go')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(initial_centers[:,0],initial_centers[:,1],'r+')
plt.plot(centers[:,0],centers[:,1],'g+')
plt.plot(new_centers[:,0],new_centers[:,1],'b+')
plt.show()

true_positive = len(np.where((true_outliers == new_outliers).all(1))[0])

nrows, ncols = true_outliers.shape
dtype={'names':['f{}'.format(i) for i in range(ncols)],
       'formats':ncols * [true_outliers.dtype]}
C = np.intersect1d(true_outliers.view(dtype), new_outliers.view(dtype))
true_positive = len(C)
precision = true_positive/len(true_outliers)
recall = true_positive/len(new_outliers)

print("Precision is   " + str(precision) + "   Recall is   " +str(recall))