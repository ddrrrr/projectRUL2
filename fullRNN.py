import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet
from DIModel import Encoder, ResBlock, TimeEncoder
import pickle
import pandas as pd
from collections import OrderedDict

class FFTEncoder(nn.Module):
    def __init__(self, fea_size):
        super(FFTEncoder,self).__init__()
        self.fea_size = fea_size
        self.network = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, fea_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

class TimeEncoder(nn.Module):
    def __init__(self, fea_size):
        super(TimeEncoder, self).__init__()
        self.fea_size = fea_size
        self.network = nn.Sequential(
            nn.Conv2d(64*7)
        )