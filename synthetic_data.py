# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:45:17 2019

@author: iyerk
"""
import numpy as np
import matplotlib.pyplot as plt




def generate_synthetic_data(k,z,dim,num_samples):
    dataset = np.zeros((1,dim))
    centers = np.random.uniform(0,100,(k,dim))
    for i in range(len(centers)):
        mu = centers[i,:]
        data = np.random.normal(mu,1,(num_samples,dim))
        plt.plot(mu[0],mu[1],'bo')
        plt.plot(data[:,0],data[:,1],'r*')
        dataset = np.vstack([dataset,data])    
    
    outliers = np.random.uniform(0,100,(z,dim))
    dataset = np.vstack([dataset,outliers])
    dataset = dataset[1:,:]
    plt.plot(outliers[:,0],outliers[:,1],'g+')
    plt.show()
    
    return centers,dataset,outliers



#centers,datasets = generate_synthetic_data(4,20,2,100)