# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:47:31 2019

@author: iyerk
"""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

def cost_KM(centers,dataset):
    cost =0
    #Calculate L2 distance of each point in the dataset and all the k centers
    distances = np.sqrt(((dataset - centers[:, np.newaxis])**2).sum(axis=2))
    #Find the minimum distance of each dataset point and k centers and sum the minimum distances
    cost = sum(distances.min(axis=0))
    return cost

def local_search(dataset,centers_1,k):
    alpha = 1000000000000
    e = 0.0001
    centers = deepcopy(centers_1)
    #While the solution improves by performing swaps
    while(alpha*(1-e/k)>cost_KM(centers,dataset)):
        alpha = cost_KM(centers,dataset)
        #temporary improved set of centers
        Cbar=centers
        #for each center try swapping with each point from the dataset
        for i in range(len(centers)):
            
            for j in range(len(dataset)):
                #print("Centers")
                u = dataset[j,:]
                #print(centers)
                new_centers = deepcopy(Cbar)
                new_centers[i,:] = u                
                
                #if this is the most improved center found so far
                if(cost_KM(new_centers,dataset)<cost_KM(Cbar,dataset)):
                    #Update the temp solution
                    Cbar = deepcopy(new_centers)
                    #break
            #break  
        #Update the solution to the best swap found
        centers = deepcopy(Cbar)
    return centers


def calc_distances(dataset, centers,z):
        #Calculate L2 distance of each point in the dataset and all the k centers
        distances = np.sqrt(((dataset - centers[:, np.newaxis])**2).sum(axis=2))
        d = distances.argmin(axis=0)
        return(d)
        
        


def find_outliers(dataset,centers,clusters,z):
    distances = np.zeros(len(dataset))
    for i in range(len(dataset)):
        distances[i] = sum((dataset[i] - centers[clusters[i][0]])**2)
    distances = np.sqrt(distances)
    sorted_distances = np.argsort(distances)
    sorted_distances = sorted_distances[::-1]
    
    return(dataset[sorted_distances[:z]])




def array_set_diff(dataset,outliers):
    a1_rows = dataset.view([('', dataset.dtype)] * dataset.shape[1])
    a2_rows = outliers.view([('', outliers.dtype)] * outliers.shape[1])
    return(np.setdiff1d(a1_rows, a2_rows).view(dataset.dtype).reshape(-1, dataset.shape[1]))


def generate_gaussian_data(k,ns,dim):
    b = np.zeros((k,dim))
    for i in range(k):
        print(k)
        a = np.random.normal(np.random.randint(200),np.random.randint(300),(ns+1,dim))
        b[i,:] = a[ns,:]
        a = a[0:len(a)-1,:]
        if(i==0):
            data = a
        else:
            data = np.vstack((data,a))
    return([data,b])

k=3
z = 10
[a,b] = generate_gaussian_data(k,100,3)

d = deepcopy(b)
c = local_search(a,d,len(d))


clusters = calc_distances(a, c,z)
clusters = clusters.reshape((len(clusters),1))
z = find_outliers(a,c,clusters,z)
U_minus_Z = array_set_diff(a,z)
plt.figure(figsize=(10,10))
plt.plot(a[:,0],a[:,1],'o')
plt.plot(z[:,0],z[:,1],'r*')
plt.plot(c[:,0],c[:,1],'p')
plt.show()