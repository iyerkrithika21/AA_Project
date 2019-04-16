# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 18:05:42 2019

@author: pavit
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

def array_union(A,B):
    C=A
    for item in B:
        if item not in C:
            np.append(C,item)
    return C

def cost(dataset,C,Z):
    U_minus_Z=array_set_diff(dataset,Z)
    return cost_KM(C,U_minus_Z)
    
def local_search_outliers(dataset,k,z,centers_1):
    alpha = 1000000000000
    e = 0.0001
    #finding Z=outliers(C)
    clusters = calc_distances(dataset,centers_1,z)
    clusters = clusters.reshape((len(clusters),1))
    Z = find_outliers(dataset,centers_1,clusters,z)
    C = deepcopy(centers_1)
    #While the solution improves by performing swaps
    while(alpha*(1-e/k)>cost(dataset,C,Z)):
        alpha = cost(dataset,C,Z)
        #local search with no outliers
        U_minus_Z = array_set_diff(dataset,Z)
        C = local_search(U_minus_Z,C,k)
        Cbar = deepcopy(C)
        Zbar = deepcopy(Z)
        #find z additional outliers
        clusters = calc_distances(U_minus_Z,C,z)
        clusters = clusters.reshape((len(clusters),1))
        outlier = find_outliers(U_minus_Z,C,clusters,z)
        Znew = array_union(Z,outlier)  
        if cost(dataset,C,Z)*(1-e/k) > cost(dataset,C,Znew):
            Zbar = deepcopy(Znew)
        #for each center try swapping with each point from the dataset
        for i in range(len(C)):
            for j in range(len(dataset)):
                u = dataset[j,:]
                new_centers = deepcopy(Cbar)
                new_centers[i,:] = u         
                U_minus_Z = array_set_diff(dataset,Z)
                clusters = calc_distances(U_minus_Z,new_centers,z)
                clusters = clusters.reshape((len(clusters),1))
                outliers1 = find_outliers(U_minus_Z,new_centers,clusters,z)
                new_outlier = array_union(Z,outliers1)
                #if this is the most improved center found so far
                if(cost(dataset,new_centers,new_outlier)<cost(dataset,Cbar,Zbar)):
                    #Update the temp solution
                    Cbar = deepcopy(new_centers)
                    Zbar = deepcopy(new_outlier)
        if cost(dataset,C,Z)*(1-e/k)>cost(dataset,Cbar,Zbar):
            C = deepcopy(Cbar)
            Z = deepcopy(Zbar) 
    return [C,Z]
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
Z = find_outliers(a,c,clusters,z)
U_minus_Z = array_set_diff(a,Z)
plt.figure(figsize=(10,10))
plt.plot(a[:,0],a[:,1],'o')
plt.plot(Z[:,0],Z[:,1],'r*')
plt.plot(c[:,0],c[:,1],'p')
plt.show()
A = local_search_outliers(a,k,z,b)
Z=A[1]
c=A[0]
plt.figure(figsize=(10,10))
plt.plot(a[:,0],a[:,1],'o')
plt.plot(Z[:,0],Z[:,1],'r*')
plt.plot(c[:,0],c[:,1],'p')