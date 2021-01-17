# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:28:42 2020

@author: www
"""

import pickle
import numpy as np
with open('rul_result.pkl', 'rb') as f:
    result = pickle.load(f)
with open('rul_target.pkl','rb') as f:
    target = pickle.load(f)
    
import matplotlib.pyplot as plt
plt.plot(result[0][:,-1])
plt.plot(result[0][:,0])

rul = 10**(result[0][:,1]) -1

rul_po = result[0][:,1::2]
rul = result[0][:,::2]
rul = 10**(rul) + temp_m[:rul.shape[0],(7-rul.shape[1]):] -1
rul_cal_sum = np.sum(rul>0,axis=1)
rul_cal_sum[rul_cal_sum==0] = 1
rul = np.sum(rul,axis=1)/rul_cal_sum
plt.plot(rul)
plt.plot(target[1][0])

result = np.load('result.npy')
target = np.load('target.npy')

plt.plot(target[0,0,:])
plt.plot(result[0,0,:])

def encoder(x):
    output = np.zeros(12)
    binx = bin(x)
    for i in range(len(binx)-1):
        output[-i] = int(binx[-i])
    for i in range(1,12):
        output[-i] = int(output[-i-1])^int(output[-i])
    return output
    
def decoder(x):
    output = 0
    for i in range(12):
        if i > 0:
            output += 2**(11-i) * (int(x[i-1])^int(x[i]))
        else:
            output += 2**11 * x[0]  
    return output

last_rul = np.zeros([11,2])
for i in range(11):
    rul = 10**(result[i][:,1]) -1
    last_rul[i,0] = target[3][i][-1]
    last_rul[i,1] = rul[-1]


last_rul = np.zeros([11,2])
temp_m = np.zeros([5000,7])
for i in range(6):
    temp = np.arange(2**(6-i))
    temp = temp.reshape([1,-1])
    temp = temp.repeat(5000//2**(6-i),axis=0).reshape([-1,])
    temp_m[10*2**(6-i):,i] -= temp[:5000-10*2**(6-i)]
for i in range(11):
    rul_po = result[i][:,1::2]
    rul = result[i][:,::2]
    rul = 10**(rul) + temp_m[:rul.shape[0],(7-rul.shape[1]):] -1
    rul_cal_sum = np.sum(rul>0,axis=1)
    rul_cal_sum[rul_cal_sum==0] = 1
    rul = np.sum(rul,axis=1)/rul_cal_sum
    last_rul[i,0] = target[1][i][-1]
    last_rul[i,1] = rul[-1]
    
er = (last_rul[:,0] - last_rul[:,1]) / last_rul[:,0]
er *= 100
A = np.zeros(11)
for i in range(11):
    if er[i] <= 0:
        A[i] = np.exp(-np.log(0.5)*(er[i]/5))
    else:
        A[i] = np.exp(np.log(0.5)*(er[i]/20))
score = np.mean(A)

all_er = np.zeros([12,6])
for j in range(6):
    rate = 0.05*(j+1)
    mean_er = []
    for i in range(11):
        rul = np.exp(result[i]) - 1
        tar = target[1][i]
        slen = np.round(tar.shape[0]*rate).astype(int)
        er = (tar[-slen:] - rul[-slen:].reshape([-1,]))/tar[-slen:]
        mean_er.append(np.mean(np.abs(er)))
        all_er[i,j] = np.mean(np.abs(er))
        all_er[11,j] = np.mean(np.array(mean_er))
        print(np.mean(er))
    np.mean(np.array(mean_er))

mean_abs_er = []
for i in range(11):
    rul = np.exp(result[i]) - 1
    tar = target[1][i]
    slen = np.round(tar.shape[0]*0.25).astype(int)
    er = np.abs((tar[-slen:] - rul[-slen:].reshape([-1,]))/tar[-slen:])
    mean_abs_er.append(np.mean(er))
    print(np.mean(er))
np.mean(np.array(mean_abs_er))


from dataset import DataSet
dataset = DataSet.load_dataset("phm_feature")
data = dataset.get_value('feature')

def _Add_noise(data, SNR=4):
    noise = np.random.randn(data.shape[0],data.shape[1],data.shape[2])
    noise = noise - np.mean(noise, axis=2, keepdims=True).repeat(data.shape[2],axis=2)
    signal_power = np.linalg.norm(data, axis=2, keepdims=True)**2 / data.shape[2]
    noise_variance = signal_power / np.power(10,(SNR/10))
    noise = (np.sqrt(noise_variance) / np.std(noise,axis=2,keepdims=True)).repeat(data.shape[2],axis=2) * noise
    return data + noise

data = np.sin(np.arange(0,120,0.1).reshape([-1,2,120]))
plt.plot(data[0,0,:])
data = _Add_noise(data)
plt.plot(data[0,0,:])



idx = [0]
count = 0
while count < a.shape[0]-1:
    temp_data = a[count+1:min(a.shape[0],count+16),:].copy()
    temp_data -= a[count,:].copy().reshape([1,-1]).repeat(temp_data.shape[0],axis=0)
    temp_data = np.mean(np.power(temp_data,2),axis=1)
    count += temp_data.argmin() +1
    idx.append(count)
test = np.array(idx)
plt.plot(test)


names = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                        'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                        'Bearing3_3']

names = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
for j in range(6):
    rul = np.exp(result[j]) - 1
    tar = target[1][j]
    er = np.abs((tar[1:] - rul.reshape([-1,]))/tar[1:])
    c = []
    for i in range(er.shape[0]):
        if (abs(er[i])>0.50):
            c.append('r')
        elif (abs(er[i])>0.20 and abs(er[i])<=0.50):
            c.append('y')
        elif (abs(er[i])<0.20):
            c.append('g')
    plt.plot(tar)
    plt.scatter(np.arange(rul.shape[0]),rul,c=c,s=3)
    plt.savefig('./fig/result_scatter/'+names[j]+'.png')
    plt.show()
    
    