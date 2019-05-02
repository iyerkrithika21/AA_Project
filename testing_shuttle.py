# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:42:06 2019

@author: iyerk
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from local_search_outlier_functions import local_search_outliers
from local_search_outlier_functions import calc_distances
from pca_visualization import PCA_visualization 
from sklearn.preprocessing import StandardScaler

number_of_sample=800
k = 4
path = os.getcwd()
Data_dir = "\\datasets\\shuttle\\"
filename = "shuttle.trn"
file = open(path+Data_dir+"shuttle_training_data.pkl",'rb')
[training_data,training_label] = pickle.load(file)
unique_labels = np.unique(training_label)
labels_hist = []
for i in range(len(unique_labels)):
    labels_hist.append(len(np.where(training_label==unique_labels[i])[0]))
labels_hist = np.array(labels_hist)
np.random.seed(21)
class1 = training_data[np.where(training_label==1)[0]]
class1 = training_data[np.random.randint(low = 0,high = len(class1),size=number_of_sample)]
class4 = training_data[np.where(training_label==4)[0]]
class4 = training_data[np.random.randint(low = 0,high = len(class4),size=number_of_sample)]

class5 = training_data[np.where(training_label==5)[0]]
class5 = training_data[np.random.randint(low = 0,high = len(class5),size=number_of_sample)]
class3 = training_data[np.where(training_label==3)[0]]
class3 = training_data[np.random.randint(low = 0,high = len(class3),size=100)]

outliers = training_data[np.where(training_label>=6)[0]]
final_data = np.vstack((class1,class4))
final_data = np.vstack((final_data,class3))
final_data = np.vstack((final_data,class5))
final_data = np.vstack((final_data,outliers))
#scaler = StandardScaler()
#final_data = scaler.fit_transform(final_data)
z = len(outliers)
initial_centers = final_data[np.random.randint(0,len(final_data),k)]
new_centers,new_outliers = local_search_outliers(final_data,k,z,initial_centers)

name = "shuttle1453_2_"+str(k)+"_"+str(number_of_sample)+"_"+str(z)

#true_positive = len(np.where((outliers == new_outliers))[0])

nrows, ncols = outliers.shape
dtype={'names':['f{}'.format(i) for i in range(ncols)],
       'formats':ncols * [outliers.dtype]}
C = np.intersect1d(outliers.view(dtype), new_outliers.view(dtype))
true_positive = len(C)
precision = true_positive/len(outliers)
recall = true_positive/len(new_outliers)

print("Precision is   " + str(precision) + "   Recall is   " +str(recall))

def array_row_intersection(a,b):
   tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
   index= np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)
   return a[index==False]



new_U_minus_Z = array_row_intersection(final_data,new_outliers)

clusters = calc_distances(new_U_minus_Z,new_centers,z)
clusters = clusters.reshape((len(clusters),1))
PCA_visualization(final_data,new_U_minus_Z,new_outliers,outliers,clusters,name+"_pca1")



'''
markers =["b+","g.","r*"]
plt.figure(0)
plt.scatter(pca_U_minus_Z[:,1],pca_U_minus_Z[:,0],pca_U_minus_Z[:,2],c=clusters[:,0])
plt.show()'''