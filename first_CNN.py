import datetime
import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
from dataset import DataSet

class First_CNN(nn.Module):
    def __init__(self, fea_size):
        super(First_CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2,32,11,1,5),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32,64,11,1,5),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64,128,11,1,5),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.FC1 = nn.Sequential(
            nn.Linear(128*20,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,fea_size),
            nn.Sigmoid()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(fea_size,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0),-1)
        fea = self.FC1(x)
        out = self.FC2(fea)
        return out, fea

class Process():
    def __init__(self):
        self.dataset = DataSet.load_dataset(name = 'phm_data')
        self.lr = 1e-3
        self.epochs = 200
        self.batches = 50
        self.batch_size = 256
        self.fea_size = 64
        self.net = First_CNN(self.fea_size).cuda()
        self.optimizer = optim.Adam
                
    def Begin(self):
        dataset = DataSet.load_dataset("phm_data")
        train_iter = self._preprocess(dataset,'train')
        test_iter = self._preprocess(dataset,'test')
        optimizer = self.optimizer(self.net.parameters(),lr=self.lr)

        for e in range(1, self.epochs+1):
            train_loss = 0
            for _ in range(self.batches):
                train_loss += self._fit(optimizer, train_iter)

            train_loss /= self.batches
            val_loss = self._evaluate(test_iter)

            print("[Epoch:%d][train_loss:%.4e][val_loss:%.4e]" 
                % (e,train_loss,val_loss))

    def _preprocess(self, dataset, select):
        if select == 'train':
            _select = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        elif select == 'test':
            _select = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                        'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                        'Bearing3_3']
        else:
            raise ValueError('wrong select!')
        data = dataset.get_value('data',condition={'bearing_name':_select})
        rul = dataset.get_value('RUL',condition={'bearing_name':_select})
        

        for i in range(len(rul)):
            rul[i] = np.arange(data[i].shape[0])[::-1] + rul[i]
            rul[i] = (1 - np.exp(-rul[i] / 500)).reshape([-1,1])
            temp_one_data = data[i].transpose(0,2,1)
            temp_one_data = self._fft(temp_one_data)
            length_data = temp_one_data.shape[2]
            temp_one_data = (temp_one_data - np.repeat(np.min(temp_one_data, axis=2, keepdims=True),length_data,axis=2)) \
                / np.repeat((np.max(temp_one_data,axis=2,keepdims=True) - np.min(temp_one_data,axis=2,keepdims=True)),length_data,axis=2)
            data[i] = temp_one_data

        return [data, rul]

    def _fft(self, data):
        fft_data = np.fft.fft(data,axis=2)/data.shape[2]
        fft_data = (np.abs(fft_data))**2
        fft_data = fft_data[:,:,1:1281]
        return fft_data

    def _fit(self, optimizer, train_iter):
        self.net.train()
        batch_data = []
        batch_rul = []
        train_num = [self.batch_size//len(train_iter[0])]*len(train_iter[0])
        for x in random.sample(range(0,len(train_iter[0])),self.batch_size % len(train_iter[0])):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            temp_idx = np.random.randint(0,train_iter[0][i].shape[0],size=x)
            batch_data.append(train_iter[0][i][temp_idx])
            batch_rul.append(train_iter[1][i][temp_idx])

        batch_data = np.concatenate(batch_data,axis=0)
        batch_rul = np.concatenate(batch_rul, axis=0)

        batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
        batch_rul = Variable(torch.from_numpy(batch_rul).type(torch.FloatTensor)).cuda()

        output,_ = self.net(batch_data)
        loss = nn.functional.mse_loss(output, batch_rul)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()        #empty useless variable
        return loss.data.cpu().numpy()

    def _evaluate(self, test_iter):
        self.net.eval()
        batch_data = []
        batch_rul = []
        test_num = [self.batch_size//len(test_iter[0])]*len(test_iter[0])
        for x in random.sample(range(0,len(test_iter[0])),self.batch_size % len(test_iter[0])):
            test_num[x] += 1

        for i,x in enumerate(test_num):
            temp_idx = np.random.randint(0,test_iter[0][i].shape[0],size=x)
            batch_data.append(test_iter[0][i][temp_idx])
            batch_rul.append(test_iter[1][i][temp_idx])

        batch_data = np.concatenate(batch_data,axis=0)
        batch_rul = np.concatenate(batch_rul, axis=0)

        batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
        batch_rul = Variable(torch.from_numpy(batch_rul).type(torch.FloatTensor)).cuda()

        output,_ = self.net(batch_data)
        loss = nn.functional.mse_loss(output, batch_rul)
        return loss.data.cpu().numpy()

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = Process()
    p.Begin()