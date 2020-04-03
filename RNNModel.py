import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet
from DIModel import DIModel
import pickle

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.gru = nn.GRU(16*17,16,2,dropout=0.3,bidirectional=True)


class RULProdict():
    def __init__(self):
        pass

    def _preprocess(self):
        pass

    def _GenHealthLabel(self, data):
        rms_data = []
        HealthLabel = []
        posibility = []
        RUL = []
        for _,x in enumerate(data):
            temp_one_data = x.transpose(0,2,1)
            length = temp_one_data.shape[2]

            temp_one_data = np.square(temp_one_data)
            temp_one_data = np.sum(temp_one_data,axis=2) / length
            temp_one_data = np.sqrt(temp_one_data)

            time_len = temp_one_data.shape[0]

            temp_mean = np.mean(temp_one_data[:time_len//2,:],axis=0)
            temp_std = np.std(temp_one_data[:time_len//2,:],axis=0)

            temp_healthlabel = np.zeros(temp_one_data.shape)
            temp_healthlabel[temp_one_data>(temp_mean+3*temp_std)] = 1
            temp_posibility = np.mean(temp_healthlabel, axis=1)
            temp_healthlabel = np.max(temp_healthlabel, axis=1)
            temp_rul = np.arange(temp_one_data.shape[0])[::-1]
            temp_rul[temp_healthlabel==0] = -1

            HealthLabel.append(temp_healthlabel)
            posibility.append(temp_posibility)
            RUL.append(temp_rul)
            rms_data.append(temp_one_data)

        return rms_data ,HealthLabel, posibility, RUL


if __name__ == "__main__":
    p = RULProdict()
    train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
    dataset = DataSet.load_dataset(name = 'phm_data')
    temp_data = dataset.get_value('data',condition={'bearing_name':train_bearings})
    rms_data ,HealthLabel, posibility, RUL = p._GenHealthLabel(temp_data)

    pickle.dump([rms_data,HealthLabel,posibility,RUL], open('test.pkl', 'wb'), True)