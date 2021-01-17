# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 08:39:28 2020

@author: www
"""

import numpy as np
fr = np.array([1800,1650,1500])
fr = fr/60
n=13
d=3.5
D = 25.6

f_bpfo=n*fr/2*(1-d/D)
f_bpfi=n*fr/2*(1+d/D)
f_ftf = fr/2*(1-d/D)
f_bsf = D*fr/2/d*(1-(d/D)**2)