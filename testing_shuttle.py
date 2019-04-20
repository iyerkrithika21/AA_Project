# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:42:06 2019

@author: iyerk
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle 
import os 

from sklearn import datasets
from sklearn.metrics import classification_report
from local_search_outlier_functions import local_search_outliers
from local_search_outlier_functions import array_set_diff
from local_search_outlier_functions import calc_distances
from local_search_outlier_functions import local_search


path = os.getcwd()
Data_dir = "\\datasets\\shuttle\\"
filename = "shuttle.trn"
file = open(path+Data_dir+"shuttle_training_data.pkl",'rb')
[training_data,training_label] = pickle.load(file)
z = 20
k = 3
n = np.random.randint(1,len(training_data),size=k)
initial_centers = training_data[n,0:1]

X_new = array_set_diff(training_data,initial_centers)
[centers,outliers] = local_search_outliers(X_new[:,0:1],k,z,initial_centers)

