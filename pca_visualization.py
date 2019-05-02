# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:42:03 2019

@author: u1135255
"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


def PCA_visualization(final_data,new_U_minus_Z,new_outliers,outliers,clusters,name):
    #scaler = StandardScaler()
    #final_data = scaler.fit_transform(final_data)
    pca = PCA(n_components=3)
    pca.fit(final_data)
    pca_U_minus_Z = pca.transform(new_U_minus_Z)
    pca_outliers = pca.transform(new_outliers)
    true_outliers = pca.transform(outliers)
    
    
    fig = plt.figure(1,figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(pca_U_minus_Z[:,1],pca_U_minus_Z[:,2],pca_U_minus_Z[:,0],s=10)
    
    ax.scatter(pca_outliers[:,1],pca_outliers[:,2],pca_outliers[:,0],c="red",s=80)
    ax.scatter(true_outliers[:,1],true_outliers[:,2],true_outliers[:,0],c="green",s=40)
    plt.legend(["Clusters","Detected Outliers","True Outliers"])
    plt.savefig(name+".jpg")
    plt.show()