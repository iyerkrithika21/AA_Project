# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:08:37 2019

@author: iyerk
"""

import os 
import numpy as np
import pickle 
path = os.getcwd()


#------------------------------------------------------------------------------
#---------------SHUTTLE DATA---------------------------------------------------
#------------------------------------------------------------------------------
Data_dir = "\\datasets\\shuttle\\"
filename = "shuttle.trn"

with open(path+Data_dir+filename) as f:
    content = f.readlines()

training_data = []
for i in range(len(content)):
    token = content[i]
    token = token.replace('\x00','')
    token = token.strip()
    token=token.split(" ")
    if(len(token)==10):
        token = list(map(float,token))
        training_data.append(token)
    
    
training_data = np.array(training_data)
training_label = training_data[:,-1]
training_data = training_data[:,:9]

with open(path+Data_dir+'shuttle_training_data.pkl', 'wb') as f:
    pickle.dump([training_data,training_label], f)

filename = "shuttle.tst"

with open(path+Data_dir+filename) as f:
    content = f.readlines()

testing_data = []
for i in range(len(content)):
    token = content[i]
    token = token.replace('\x00','')
    token = token.strip()
    token=token.split(" ")
    if(len(token)==10):
        token = list(map(float,token))
        testing_data.append(token)
    
    
testing_data = np.array(testing_data)
testing_label = testing_data[:,-1]
testing_data = testing_data[:,:9]

with open(path+Data_dir+'shuttle_testing_data.pkl', 'wb') as f:
    pickle.dump([testing_data ,testing_label], f)
