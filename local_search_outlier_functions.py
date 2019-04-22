# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:06:32 2019

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

def array_union(arr1,arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    u = np.union1d(arr1_view, arr2_view)
    return u.view(arr1.dtype).reshape(-1, arr1.shape[1])
    
def cost(dataset,C,Z):
    U_minus_Z=array_set_diff(dataset,Z)
    return cost_KM(C,U_minus_Z)

def find_kfarthest(dataset,centers,z):
    clusters = calc_distances(dataset,centers,z)
    clusters = clusters.reshape((len(clusters),1))
    distances = np.zeros(len(dataset))
    for i in range(len(dataset)):
        distances[i] = sum((dataset[i] - centers[clusters[i][0]])**2)
    distances = np.sqrt(distances)
    sorted_distances = np.argsort(distances)
    sorted_distances = sorted_distances[::-1]

    return(dataset[sorted_distances[:z]])
    
def local_search_outliers(dataset,k,z,centers_1):
    alpha = 100000000000
    e = 0.001
    print(k)
    #finding Z=outliers(C)
    clusters = calc_distances(dataset,centers_1,z)
    clusters = clusters.reshape((len(clusters),1))
    Z = find_outliers(dataset,centers_1,clusters,z)
    C = deepcopy(centers_1)
    

    
    #While the solution improves by performing swaps
    while(alpha*(1-e/k)>cost(dataset,C,Z)):

    #--------------------------------------------------------------------------
        print("Alpha e by k is " + str(alpha*(1-e/k)))
        print("Cost is " + str(cost(dataset,C,Z)))
    #--------------------------------------------------------------------------
        #print("I am in while loop")
        #calculate prec recall and draw the graph
        alpha = cost(dataset,C,Z)
        
        #local search with no outliers
        U_minus_Z = array_set_diff(dataset,Z)
        print("Calling LS ")
        C = local_search(U_minus_Z,C,k)
        Cbar = deepcopy(C)
        Zbar = deepcopy(Z)
        
        
        #find z additional outliers
        clusters = calc_distances(U_minus_Z,C,z)
        clusters = clusters.reshape((len(clusters),1))
        outlier = find_outliers(U_minus_Z,C,clusters,z)
        Znew_temp = array_union(Z,outlier)
        Znew = find_kfarthest(Znew_temp,C,z)
        
        
        print("2nd")
        print("Cost 2 is" + str(cost(dataset,C,Z)*(1-e/k)))
        print("Cost is " + str(cost(dataset,C,Znew)))
        if cost(dataset,C,Z)*(1-e/k) > cost(dataset,C,Znew):
            Zbar = deepcopy(Znew)
            
        #print("Entering for loop")   
        #for each center try swapping with each point from the dataset
        for i in range(len(C)):
            for j in range(len(dataset)):
                #print("lso for")
                u = dataset[j,:]
                new_centers = deepcopy(Cbar)
                new_centers[i,:] = u         
                U_minus_Z = array_set_diff(dataset,Z)
                clusters = calc_distances(U_minus_Z,new_centers,z)
                clusters = clusters.reshape((len(clusters),1))
                outliers1 = find_outliers(U_minus_Z,new_centers,clusters,z)
                new_outlier_temp = array_union(Z,outliers1)
                new_outlier = find_kfarthest(new_outlier_temp,new_centers,z)
        
                
                #if this is the most improved center found so far
                if(cost(dataset,new_centers,new_outlier)<cost(dataset,Cbar,Zbar)):
                    #Update the temp solution
                    Cbar = deepcopy(new_centers)
                    Zbar = deepcopy(new_outlier)
        print("For loop done")     
        a=cost(dataset,C,Z)*(1-e/k)
        b=cost(dataset,Cbar,Zbar)
        print(a)
        print(b)
        if a>b:
            C = deepcopy(Cbar)
            Z = deepcopy(Zbar) 
            print("If succeded")
        print("Outliers size:"+str(len(Z)))

        
            
            
    return [C,Z]


def local_search(dataset,centers_1,k):
    print("LS executing")
    alpha = 90000000000000000
    e = 0.0001
    centers = deepcopy(centers_1)
    #While the solution improves by performing swaps
    while(alpha*(1-e/k)>cost_KM(centers,dataset)):
        alpha = cost_KM(centers,dataset)
        #temporary improved set of centers
        Cbar=centers
        #print("Inside LS while loop")
        #for each center try swapping with each point from the dataset
        for i in range(len(centers)):

            for j in range(len(dataset)):
                #print("ls for")
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
        #print("ls for done")
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
    if len(outliers) ==0:
        a1_rows = dataset.view([('', dataset.dtype)] * dataset.shape[1])
        return (a1_rows.view(dataset.dtype).reshape(-1, dataset.shape[1]))
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


def plot_knn(a,b,c):
    plt.figure(figsize=(10,10))
    plt.plot(a[0:100,0],a[0:100,1],'.') 
    plt.plot(a[100:200,0],a[100:200,1],'g.')
    plt.plot(a[200:300,0],a[200:300,1],'r.') 
    plt.plot(b[:,0],b[:,1],'yd', markersize=8)
    
    plt.plot(c[:,0],c[:,1],'s', markersize=8)
    plt.legend()
    plt.show()

