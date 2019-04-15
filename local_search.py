# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:14:08 2019

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

def plot_knn(a,b,c):
    plt.figure(figsize=(10,10))
    plt.plot(a[0:100,0],a[0:100,1],'.') 
    plt.plot(a[100:200,0],a[100:200,1],'g.')
    plt.plot(a[200:300,0],a[200:300,1],'r.') 
    plt.plot(d[:,0],d[:,1],'yd', markersize=8)
    
    plt.plot(c[:,0],c[:,1],'s', markersize=8)
    plt.legend()
    plt.show()
'''
a = np.random.rand(200, 2)
plt.plot(a[:,0],a[:,1],'b+')

b = np.random.rand(5, 2)
d = deepcopy(b)
plt.plot(d[:,0],d[:,1],'r*')
#plt.show()
c = local_search(a,d,len(d))
plt.plot(c[:,0],c[:,1],'go')

plt.show()
'''
k=3
b = np.zeros((k,2))
a1 = np.random.normal(1, 100, (101,2))
b[0,:] = a1[100,:]
a1 = a1[0:len(a1)-1,:]
a2 = np.random.normal(60, 75, (101,2))
b[1,:] = a2[100,:]
a2 = a2[0:len(a2)-1,:]
a3 = np.random.normal(105, 25, (101,2))
b[2,:] = a3[100,:]
a3 = a3[0:len(a3)-1,:]
a = np.vstack((a1,a2))
a = np.vstack((a,a3))

d = deepcopy(b)
c = local_search(a,d,len(d))
plot_knn(a,d,c)