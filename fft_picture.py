# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:53:04 2020

@author: www
"""
from dataset import DataSet
dataset = DataSet.load_dataset("phm_data")
data = dataset.get_value('data')
names = dataset.get_value('bearing_name')

import numpy as np
fft_data = []
for i in range(len(data)):
    temp_fft_data = np.fft.fft(data[i],axis=1)/data[i].shape[1]
    temp_fft_data = temp_fft_data[:,1:1281,:]
    temp_fft_data = np.abs(temp_fft_data)
    fft_data.append(temp_fft_data)

f = np.arange(1,1281)*10

F_mean = []
for i in range(len(fft_data)):
    temp = np.mean(fft_data[i],axis=1)
    F_mean.append(temp)
    
F_c = []
for i in range(len(fft_data)):
    temp_f = f.reshape([1,-1,1])
    temp_f = np.repeat(temp_f,fft_data[i].shape[0],axis=0)
    temp_f = np.repeat(temp_f,fft_data[i].shape[2],axis=2)
    temp = np.sum(fft_data[i]*temp_f/np.sqrt(fft_data[i]),axis=1)
    F_c.append(temp)
    
F_rms = []
for i in range(len(fft_data)):
    temp = np.sqrt(np.mean(fft_data[i]**2,axis=1))
    F_rms.append(temp)
    
F_std = []
for i in range(len(fft_data)):
    temp_mean = np.repeat(F_mean[i].reshape([-1,1,2]),1280,axis=1)
    temp = np.sqrt(np.mean((fft_data[i]-temp_mean)**2,axis=1))
    F_std.append(temp)
    
F_focus = []
for i in range(len(fft_data)):
    temp_f = f.reshape([1,-1,1])
    temp_f = np.repeat(temp_f,fft_data[i].shape[0],axis=0)
    temp_f = np.repeat(temp_f,fft_data[i].shape[2],axis=2)
    temp_fc = np.repeat(F_c[i].reshape([-1,1,2]),1280,axis=1)
    temp_fenzi = np.sum(np.abs(fft_data[i]*(temp_f-temp_fc)),axis=1)
    temp_fenmu = np.sum(fft_data[i]*temp_f,axis=1)
    temp = 1 - temp_fenzi/temp_fenmu
    F_focus.append(temp)
    
F_fv = []
for i in range(len(fft_data)):
    temp_mean = np.repeat(F_mean[i].reshape([-1,1,2]),1280,axis=1)
    temp = np.sum((fft_data[i]-temp_mean)**4,axis=1) / (1280*F_std[i])
    F_fv.append(temp)
    
import matplotlib.pyplot as plt

for i in range(len(F_mean)):
    plt.plot(F_mean[i][:,0])
    plt.plot(F_mean[i][:,1])
    plt.savefig('./fig/F_mean/'+names[i]+'.png')
    plt.show()
    
for i in range(len(F_c)):
    plt.plot(F_c[i][:,0])
    plt.plot(F_c[i][:,1])
    plt.savefig('./fig/F_c/'+names[i]+'.png')
    plt.show()
    
for i in range(len(F_rms)):
    plt.plot(F_rms[i][:,0])
    plt.plot(F_rms[i][:,1])
    plt.savefig('./fig/F_rms/'+names[i]+'.png')
    plt.show()
    
for i in range(len(F_std)):
    plt.plot(F_std[i][:,0])
    plt.plot(F_std[i][:,1])
    plt.savefig('./fig/F_std/'+names[i]+'.png')
    plt.show()
    
for i in range(len(F_focus)):
    plt.plot(F_focus[i][:,0])
    plt.plot(F_focus[i][:,1])
    plt.savefig('./fig/F_focus/'+names[i]+'.png')
    plt.show()
    
for i in range(len(F_fv)):
    plt.plot(F_fv[i][:,0])
    plt.plot(F_fv[i][:,1])
    plt.savefig('./fig/F_fv/'+names[i]+'.png')
    plt.show()