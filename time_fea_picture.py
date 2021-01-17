# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:31:23 2020

@author: www
"""
import numpy as np
from dataset import DataSet
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
dataset = DataSet.load_dataset("phm_data")
data = dataset.get_value('data')
names = dataset.get_value('bearing_name')

one_data = data[0]
d_mean = np.mean(one_data,axis=1)
d_var = np.std(one_data,axis=1)
d_rms = np.sqrt(np.mean(one_data**2,axis=1))
d_peak = 0.5*(np.max(one_data,axis=1)-np.min(one_data,axis=1))
d_ck = np.mean((one_data - d_mean.reshape([-1,1,2]).repeat(2560,axis=1))**4/(d_var.reshape([-1,1,2]).repeat(2560,axis=1))**2,axis=1)
d_c = d_peak/d_rms
d_s = d_rms / d_mean
d_i = d_peak / d_mean
d_cs = np.mean((one_data - d_mean.reshape([-1,1,2]).repeat(2560,axis=1))**3,axis=1)/d_var**1.5
d_clf = np.abs(np.max(one_data,axis=1))/np.mean(np.sqrt(np.abs(one_data)),axis=1)**2
d_kv = np.mean((one_data - d_mean.reshape([-1,1,2]).repeat(2560,axis=1))**4,axis=1)/(np.mean(one_data**2,axis=1))**2
d_asl = 20*np.log10(np.mean(np.abs(one_data),axis=1))

d_fea = np.concatenate([d_mean,d_var,d_rms,d_peak,d_ck,d_c,d_s,d_i,d_cs,d_clf,d_kv,d_asl],axis=1)

pca=PCA(n_components=8)
pca_data=pca.fit_transform(d_fea)

tsne=TSNE()
tsne_data = tsne.fit_transform(pca_data)

c = np.arange(tsne_data.shape[0])/tsne_data.shape[0]
plt.scatter(tsne_data[:,0],tsne_data[:,1],c=c)
plt.colorbar()





dataset = DataSet.load_dataset("phm_feature_TimeNoNorm")
EN_data = dataset.get_value('feature')
EN_data = EN_data[0]

dataset = DataSet.load_dataset("phm_feature_16bitCOD")
COD_data = dataset.get_value('feature')
COD_data = COD_data[0]
dataset = DataSet.load_dataset("phm_feature_16bitShuff")
Shuff_data = dataset.get_value('feature')
Shuff_data = Shuff_data[0]
combine_data = np.concatenate([COD_data,Shuff_data],axis=1)

c = np.arange(combine_data.shape[0])

pca=PCA(n_components=8)
pca_data=pca.fit_transform(d_fea)

tsne=TSNE()
tsne_data_time_fea = tsne.fit_transform(pca_data)

pca=PCA(n_components=8)
pca_data=pca.fit_transform(EN_data)

tsne=TSNE()
tsne_data_En = tsne.fit_transform(pca_data)

pca=PCA(n_components=8)
pca_data=pca.fit_transform(combine_data)

tsne=TSNE()
tsne_data_combine = tsne.fit_transform(pca_data)

fig, ax = plt.subplots(1,3)
ax = ax.flatten()
ax0=ax[0].scatter(tsne_data_time_fea[1:,0],tsne_data_time_fea[1:,1],c=c)
ax1=ax[1].scatter(tsne_data_En[:,0],tsne_data_En[:,1],c=c)
ax2=ax[2].scatter(tsne_data_combine[:,0],tsne_data_combine[:,1],c=c)
fig.colorbar(ax2,ax=[ax[0],ax[1],ax[2]])