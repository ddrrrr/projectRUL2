import datetime
import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
from dataset import DataSet
import scipy.signal

class TimeEncoder(nn.Module):
    def __init__(self,fea_size):
        super(TimeEncoder, self).__init__()
        self.fea_size = fea_size
        self.encoder = nn.Sequential(
            nn.Conv2d(2,32,[1,64],[1,8]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,[1,5],[1,4]),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,[1,6],[1,4]),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,[1,3],[1,2]),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,fea_size,[1,9]),
            nn.BatchNorm2d(fea_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x).view(x.size(0),self.fea_size,-1).transpose(0,2,1)

class FreEncoder(nn.Module):
    def __init__(self,fea_size):
        super(FreEncoder, self).__init__()
        self.fea_size = fea_size
        self.encoder = nn.Sequential(
            nn.Linear(2560,512),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(512,fea_size),
            nn.Tanh(),
        )
    def forward(self,x):
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),-1,2560)
        return self.encoder(x)


class CDC(nn.Module):
    def __init__(self, timestep, batch_size, seq_len, fea_size):
        super(CDC,self).__init__()
        self.batch_size = batch_size
        self.fea_size = fea_size
        self.timestep = timestep
        self.seq_len = seq_len
        # self.encoder = TimeEncoder(fea_size)
        self.encoder = FreEncoder(fea_size)
        self.rnn_hidden_size = 64
        self.gru = nn.GRU(self.fea_size,self.rnn_hidden_size,1,bidirectional=False,batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(self.rnn_hidden_size,self.fea_size) for i in range(self.timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    # def init_hidden(self, batch_size, use_gpu=True):
    #     if use_gpu: return torch.zeros(1, batch_size, 64).cuda()
    #     else: return torch.zeros(1, batch_size, 64)

    def forward(self, x):
        batch = x.size(0)
        t_samples = torch.randint(self.seq_len - self.timestep, size=(1,)).long()
        z = self.encoder(x)
        # z = z.transpose(1,2)    # b*L*128
        nce = 0
        encode_samples = torch.empty((self.timestep, batch, self.fea_size)).float()
        for i in range(1,self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,self.fea_size)
        forward_seq = z[:,:t_samples+1,:]
        output,h = self.gru(forward_seq,torch.zeros(1, self.batch_size, self.rnn_hidden_size).cuda())
        c_t = output[:,t_samples,:].view(batch,self.rnn_hidden_size)
        pred = torch.empty((self.timestep, batch, self.fea_size)).float()
        for i in range(self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in range(self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total),dim=0),torch.arange(0,batch)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce

class Process():
    def __init__(self):
        self.dataset = DataSet.load_dataset(name = 'phm_data')
        self.lr = 1e-3
        self.optimizer = optim.Adam
        self.epochs = 150
        self.batches = 10
        self.batch_size = 8
        self.seq_len = 128
        self.fea_size = 32
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']

        self.network = CDC(32,self.batch_size,self.seq_len,self.fea_size).cuda()

    def Begin(self):
        train_iter = self._preprocess('train')
        test_iter = self._preprocess('test')
        optimizer = self.optimizer(self.network.parameters(),lr=self.lr)

        for e in range(1, self.epochs+1):
            train_loss = 0
            train_acc = 0
            for _ in range(self.batches):
                temp_acc, temp_loss = self._cal_one_batch(train_iter, isTrain=True, opti=optimizer)
                train_acc += temp_acc
                train_loss += temp_loss
            test_acc, test_loss = self._cal_one_batch(test_iter)

            print("[Epoch:%d][train_loss:%.4e][train_acc:%.4f][val_loss:%.4e][val_acc:%.4f]"
                % (e, train_loss/self.batches, train_acc/self.batches, test_loss, test_acc))

            torch.save(self.network.encoder,datetime.datetime.now().strftime("%Y%m%d") + 'encoder.pkl')

    def _preprocess(self, select):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        elif select == 'all':
            temp_data = self.dataset.get_value('data')
        else:
            raise ValueError('error selection for features!')
            
        for i,x in enumerate(temp_data):
            temp_one_data = x.transpose(0,2,1)
            temp_one_data = self._fft(temp_one_data)
            temp_data[i] = temp_one_data
        
        return temp_data

    def _fft(self, data):
        fft_data = np.fft.fft(data,axis=2)/data.shape[2]
        fft_data = (np.abs(fft_data))**2
        fft_data = fft_data[:,:,1:1281]
        data_len = fft_data.shape[2]
        
        fft_data = (fft_data - np.repeat(np.min(fft_data, axis=2, keepdims=True),data_len,axis=2)) \
                / np.repeat((np.max(fft_data,axis=2,keepdims=True) - np.min(fft_data,axis=2,keepdims=True)),data_len,axis=2)
        return fft_data

    def _cal_one_batch(self, data_iter, isTrain=False, opti=None):
        train_idx = [1299,688,459,179,300,1200]
        if isTrain:
            self.network.train()
        else:
            self.network.eval()

        each_data_num = [self.batch_size//len(data_iter)]*len(data_iter)
        for x in random.sample(range(len(data_iter)), self.batch_size % len(data_iter)):
            each_data_num[x] += 1
        batch_data = []
        data_len = data_iter[0].shape[-1]
        for i,x in enumerate(each_data_num):
            if isTrain:
                temp_idx = np.random.randint(train_idx[i],data_iter[i].shape[0]-self.seq_len,size=x)
            else:
                temp_idx = np.random.randint(0,data_iter[i].shape[0]-self.seq_len,size=x)
            for one_idx in temp_idx:
                batch_data.append(data_iter[i][one_idx:one_idx+self.seq_len,:,:].reshape([1,2,self.seq_len,data_len]))
        batch_data = np.concatenate(batch_data, axis=0)
        batch_data = Variable(torch.from_numpy(batch_data.copy()).type(torch.FloatTensor)).cuda()
        acc,loss = self.network(batch_data)

        if isTrain:
            loss.backward()
            opti.step()
            torch.cuda.empty_cache()        #empty useless variable

        return acc, loss.data.cpu().numpy()

    def _gen_feature(self, model_name):
        bearing_data = self._preprocess('all')
        bearing_name = self.dataset.get_value('bearing_name')
        bearing_rul = self.dataset.get_value('RUL')

        feature_net = torch.load(model_name + '.pkl')
        feature_net.eval()
        feature_dataset = DataSet(name='phm_feature',index=['bearing_name','feature','rul','rul_encoding'])

        for i in range(len(bearing_data)):
            temp_one_data = bearing_data[i]
            temp_one_rul = np.arange(temp_one_data.shape[0])[::-1] + bearing_rul[i]
            pe = (np.exp(temp_one_rul+1)).reshape([-1,1])
            temp_one_data = Variable(torch.from_numpy(temp_one_data.copy()).type(torch.FloatTensor)).cuda() 
            temp_one_feature_list = []
            for j in range(temp_one_data.shape[0]//64+1):
                temp_one_fea = feature_net(temp_one_data[j*64:min(j*64+64,temp_one_data.shape[0]-1)])
                temp_one_feature_list.append(temp_one_fea.data.cpu().numpy().reshape([temp_one_fea.shape[0],-1]))
            temp_one_fea = np.concatenate(temp_one_feature_list,axis=0)

            feature_dataset.append([bearing_name[i],temp_one_fea,temp_one_rul,pe])
        
        feature_dataset.save()
            

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = Process()
    p.Begin()
    p._gen_feature('20200530encoder')