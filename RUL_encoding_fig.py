# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:40:30 2020

@author: www
"""

import numpy as np
import matplotlib.pyplot as plt
x1 = np.arange(10000)
y1 = x1[::-1]/1000
y1_range = np.zeros([2,10000])
y1_range[0,:] = y1 + 0.4
y1_range[1,:] = y1 - 0.4
plt.plot(x1,y1)
plt.fill_between(x1,y1_range[0,:],y1_range[1,:],color='r',alpha=.2)
y2_encoding = np.log(x1[::-1]+1)
y2_range = np.zeros([2,10000])
y2_range[0,:] = y2_encoding + 0.4
y2_range[1,:] = y2_encoding - 0.4
y2_range = (np.exp(y2_range) -1)/1000

plt.plot(x1,y1,color='k')
plt.fill_between(x1,y1_range[0,:],y1_range[1,:],color='b',alpha=.2)
plt.fill_between(x1,y2_range[0,:],y2_range[1,:],color='r',alpha=.2)
ax = plt.gca()                                            # get current axis 获得坐标轴对象
ax.spines['right'].set_color('none') 
ax.spines['top'].set_color('none')         # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
ax.xaxis.set_ticks_position('bottom')   
ax.yaxis.set_ticks_position('left')          # 指定下边的边作为 x 轴   指定左边的边为 y 轴
ax.spines['bottom'].set_position(('data', 0))   #指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
ax.spines['left'].set_position(('data', 0))
plt.savefig('log_fig_1.png')


plt.plot(x1[9000:],y1[9000:])
plt.fill_between(x1[9000:],y1_range[0,9000:],y1_range[1,9000:],color='b',alpha=.2)
plt.fill_between(x1[9000:],y2_range[0,9000:],y2_range[1,9000:],color='r',alpha=.2)

ax = plt.gca()                                            # get current axis 获得坐标轴对象
ax.spines['right'].set_color('none') 
ax.spines['top'].set_color('none')         # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
ax.xaxis.set_ticks_position('bottom')   
ax.yaxis.set_ticks_position('left')          # 指定下边的边作为 x 轴   指定左边的边为 y 轴
ax.spines['bottom'].set_position(('data', 0))   #指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
ax.spines['left'].set_position(('data', 9000))
plt.savefig('log_fig_2.png')
