import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet
from DIModel import Encoder
import pickle

class RNN_Encoder(nn.Module):
    def __init__(self):
        super(RNN_Encoder,self).__init__()
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

    def _fft(self, data):
        fft_data = np.fft.fft(data,axis=2)/data.shape[2]
        fft_data = (np.abs(fft_data))**2
        fft_data = fft_data[:,:,1:1281]
        return fft_data

    def _GenFeature(self, model_name):
        bearing_dataset = DataSet.load_dataset('phm_data')
        bearing_data = bearing_dataset.get_value('data')
        bearing_name = bearing_dataset.get_value('bearing_name')

        feature_net = torch.load(model_name + '.pkl')
        feature_net.eval()
        feature_dataset = DataSet(name='phm_feature',index=['bearing_name','feature'])

        for i in range(len(bearing_data)):
            temp_one_data = bearing_data[i].transpose(0,2,1)
            temp_one_data = self._fft(temp_one_data)
            length_data = temp_one_data.shape[2]
            temp_one_data = (temp_one_data - np.repeat(np.min(temp_one_data, axis=2, keepdims=True),length_data,axis=2)) \
                / np.repeat((np.max(temp_one_data,axis=2,keepdims=True) - np.min(temp_one_data,axis=2,keepdims=True)),length_data,axis=2)
            temp_one_data = Variable(torch.from_numpy(temp_one_data.copy()).type(torch.FloatTensor)).cuda() 

            temp_one_feature_list = []
            for j in range(temp_one_data.shape[0]//64):
                temp_one_feature_list.append(feature_net(temp_one_data[j:min(j+64,temp_one_data.shape[0]-1)]).data.cpu().numpy())
            
            temp_one_feature = np.vstack(temp_one_feature_list)
            feature_dataset.append([bearing_name[i],temp_one_feature])

        feature_dataset.save()

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = RULProdict()
    # train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
    # dataset = DataSet.load_dataset(name = 'phm_data')
    # temp_data = dataset.get_value('data',condition={'bearing_name':train_bearings})
    # rms_data ,HealthLabel, posibility, RUL = p._GenHealthLabel(temp_data)

    # pickle.dump([rms_data,HealthLabel,posibility,RUL], open('test.pkl', 'wb'), True)

    p._GenFeature('20200411encoder')