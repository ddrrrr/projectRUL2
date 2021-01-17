# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:04:22 2020

@author: www
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from dataset import DataSet
dataset = DataSet.load_dataset("phm_feature_1")
data = dataset.get_value('feature')

one_data = data[0]
pca=PCA(n_components=16)
pca_data=pca.fit_transform(one_data)

tsne=TSNE()
tsne_data = tsne.fit_transform(pca_data)

c = np.arange(tsne_data.shape[0])/tsne_data.shape[0]
plt.scatter(tsne_data[:,0],tsne_data[:,1],c=c)
plt.colorbar()


one_data = data[0]
pca=PCA(n_components=1)
pca_data=pca.fit_transform(one_data)
plt.plot(pca_data)

[1299,688,459,179,400,1200]
[2750,830,870,750,490,1450]

t = np.array([573,33.9,161,146,757,753,139,309,129,58,82])
r = np.array([287.649,13.1051, 70.6877, 145.602, 629.786, 443.072, 346.343, 361.339, 272.634, 203.838, 140.599])
er = (t - r) / t
er *= 100
A = np.zeros(11)
for i in range(11):
    if er[i] <= 0:
        A[i] = np.exp(-np.log(0.5)*(er[i]/5))
    else:
        A[i] = np.exp(np.log(0.5)*(er[i]/20))
        