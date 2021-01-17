# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:51:48 2020

@author: www
"""

from dataset import DataSet
dataset = DataSet.load_dataset("phm_data")
data = dataset.get_value('data')
names = dataset.get_value('bearing_name')

import numpy as np
import scipy.signal

one_data = data[0]

stft_data = np.zeros([one_data.shape[0],2,50,50])
for i in range(one_data.shape[0]):
    for j in range(2):
        x,y,z = scipy.signal.stft(one_data[i,:,j],fs=25600,nperseg=256,noverlap=204)
        z = np.abs(z[:50,:50])
        stft_data[i,j,:,:] = z
        
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3)
ax = ax.flatten()

vmax = np.max(stft_data[2800,0,:,:])
#plt.subplot(1,3,1)
ax0=ax[0].pcolormesh(y[0:50],x[0:50],stft_data[200,0,:,:],cmap='jet',vmax=vmax)
#plt.subplot(1,3,2)
ax1=ax[1].pcolormesh(y[0:50],x[0:50],stft_data[2200,0,:,:],cmap='jet',vmax=vmax)
#plt.subplot(1,3,3)
ax2=ax[2].pcolormesh(y[0:50],x[0:50],stft_data[2800,0,:,:],cmap='jet',vmax=vmax)
fig.colorbar(ax2,ax=[ax[0],ax[1],ax[2]])

import pywt
fig, ax = plt.subplots(1,3)
ax = ax.flatten()
t = np.arange(0,0.1,1/25600)


coef1, freqs1 = pywt.cwt(one_data[200,:,0], np.arange(1,40),'shan', sampling_period=1/25600)
coef2, freqs2 = pywt.cwt(one_data[2200,:,0], np.arange(1,40),'shan', sampling_period=1/25600)
coef3, freqs3 = pywt.cwt(one_data[2800,:,0], np.arange(1,40),'shan', sampling_period=1/25600)

vmax = np.max(np.abs(coef3))

ax0=ax[0].pcolormesh(t,freqs1,np.abs(coef1),cmap='jet',vmax=vmax)
ax1=ax[1].pcolormesh(t,freqs2,np.abs(coef2),cmap='jet',vmax=vmax)
ax2=ax[2].pcolormesh(t,freqs3,np.abs(coef3),cmap='jet',vmax=vmax)
fig.colorbar(ax2,ax=[ax[0],ax[1],ax[2]])




from pyhht.emd import EMD
times = []
fre = []
t = np.arange(0,0.1,1/25600)
f = np.arange(1,1281)*10

decomposer = EMD(one_data[200,:,0])               
imfs = decomposer.decompose()
temp_fft_data = np.fft.fft(imfs,axis=1)/imfs.shape[1]
temp_fft_data = temp_fft_data[:,1:1281]
temp_fft_data = np.abs(temp_fft_data)

for i in range(5):
    plt.subplot(5,2,i*2)
    plt.plot(t,imfs[i,:])
    plt.subplot(5,2,i*2+1)
    plt.plot(f,temp_fft_data[i,:])


times = []
fre = []
decomposer = EMD(one_data[2200,:,0])               
imfs = decomposer.decompose()
temp_fft_data = np.fft.fft(imfs,axis=1)/imfs.shape[1]
temp_fft_data = temp_fft_data[:,1:1281]
temp_fft_data = np.abs(temp_fft_data)
for i in range(5):
    times.append(imfs[i,:])
    fre.append(temp_fft_data[i,:])
    
times = []
fre = []
decomposer = EMD(one_data[2800,:,0])               
imfs = decomposer.decompose()
temp_fft_data = np.fft.fft(imfs,axis=1)/imfs.shape[1]
temp_fft_data = temp_fft_data[:,1:1281]
temp_fft_data = np.abs(temp_fft_data)
for i in range(5):
    times.append(imfs[i,:])
    fre.append(temp_fft_data[i,:])

