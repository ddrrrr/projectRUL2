import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet
import pickle
import pandas as pd
import math
from collections import OrderedDict

class RUL_Net(nn.Module):
    def __init__(self,out_size):
        super(RUL_Net,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(7,16,1,bias=False),
            nn.Conv2d(16,32,4,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.AvgPool2d(2),
            nn.Conv2d(32,64,3,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.AvgPool2d(2),
            # nn.Dropout2d(),
            nn.Conv2d(64,128,3,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.AvgPool2d(2),
            # nn.Dropout2d(),
            nn.Conv2d(128,256,[3,4]),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.AvgPool2d(2),
        )
        self.FC = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(256,128,bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,out_size),
            nn.ReLU()
        )

    def forward(self,x):
        # x = nn.functional.dropout2d(x)
        x = self.cnn(x)
        x = x.view(x.size(0),-1)
        x = self.FC(x)
        return x

class RUL_Net2(nn.Module):
    def __init__(self):
        super(RUL_Net2,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(7*64,128,1,bias=False),
            # nn.Conv1d(256, 128, 5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128,128, 3,1,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128,256,3,1,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256,256,3,1,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)

            # nn.Conv3d(7*64,256,1,bias=False),
            # nn.BatchNorm3d(256),
            # nn.ReLU(),
            # nn.Conv3d(256,256,[1,3,3]),
            # nn.BatchNorm3d(256),
            # nn.ReLU(),
            # nn.Conv3d(256,512,[1,3,3]),
            # nn.BatchNorm3d(512),
            # nn.ReLU(),
        )
        self.FC = nn.Sequential(
            nn.Linear(512*4,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,1),
            nn.ReLU()
        )
        self.Atten = nn.Linear(128,1)

    def forward(self, x):
        x = x.view(x.size(0),7*64,-1)
        atten = self.Atten(x)
        atten = nn.functional.softmax(atten,dim=2)
        atten = atten.view(atten.size(0),7*64,1).repeat([1,1,x.size(2)])
        x = x*atten

        # x = x.view(x.size(0),7*64,4,5,5)
        x = self.cnn(x)
        x = x.view(x.size(0),-1)
        x = self.FC(x)
        return x

class RULPredict():
    def __init__(self):
        self.epochs = 200
        self.batches = 50
        self.batch_size = 32
        self.lr = 1e-3
        self.optimizer = optim.Adam
        self.rul_size = 1
        self.position_encoding_size = 8
        self.network = RUL_Net(self.rul_size).cuda()
        # self.network = RUL_Net2().cuda()

    def _preprocess(self, dataset, select):
        if select == 'train':
            _select = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        elif select == 'test':
            _select = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                        'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                        'Bearing3_3']
        else:
            raise ValueError('wrong select!')

        data = dataset.get_value('feature',condition={'bearing_name':_select})
        rul = dataset.get_value('rul',condition={'bearing_name':_select})
        rul_encoding = dataset.get_value('rul_encoding',condition={'bearing_name':_select})

        output = [data,rul,rul_encoding]
        return output

    def Begin(self):
        dataset = DataSet.load_dataset("phm_feature")
        train_iter = self._preprocess(dataset,'train')
        test_iter = self._preprocess(dataset,'test')
        optimizer = self.optimizer(self.network.parameters(),lr=self.lr)

        log = OrderedDict()
        log['train_loss'] = []
        log['train_rul_loss'] = []
        log['val_loss'] = []
        log['val_rul_loss'] = []

        for e in range(1, self.epochs+1):
            train_loss = 0
            train_rul_loss = 0
            for _ in range(self.batches):
                temp_train_loss = self._fit(train_iter,optimizer)
                train_loss += temp_train_loss[0]
                train_rul_loss += temp_train_loss[1]
            train_loss /= self.batches
            train_rul_loss /= self.batches

            val_loss = self._evaluate(test_iter)

            print("[Epoch:%d][train_loss:%.4e][train_rul_loss:%.4e] \
                    [val_loss:%.4e][val_rul_loss:%.4e]" 
                % (e,train_loss,train_rul_loss,
                    val_loss[0],val_loss[1]))
            log['train_loss'].append(float(train_loss))
            log['train_rul_loss'].append(float(train_rul_loss))
            log['val_loss'].append(float(val_loss[0]))
            log['val_rul_loss'].append(float(val_loss[1]))
            pd.DataFrame(log).to_csv('./model/log.csv',index=False)
            if float(val_loss[0]) == min(log['val_loss']):
                torch.save(self.network, './model/best_rul_model')
            torch.save(self.network, './model/newest_rul_model')

    def _fit(self, train_iter, optimizer):
        self.network.train()
        batch_data = []
        batch_rul = []
        batch_rul_encoding = []

        train_num = [self.batch_size//len(train_iter[0])]*len(train_iter[0])
        for x in random.sample(range(0,len(train_iter[0])),self.batch_size % len(train_iter[0])):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            random_end = np.random.randint(round(train_iter[0][i].shape[0]*0.5), round(train_iter[0][i].shape[0]*1.0), size=x)
            random_start = np.random.randint(0,round(train_iter[0][i].shape[0]*0.4), size=x)
            random_len = random_end - random_start
            for j in range(random_len.shape[0]):
                one_feed_data = self._get_one_feed_data(train_iter[0][i], random_start[j], random_len[j])
                # one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                # one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])],axis=0)
                batch_data.append(one_feed_data)
                batch_rul.append(train_iter[1][i][random_end[j]])
                batch_rul_encoding.append(train_iter[2][i][random_end[j],:].reshape([1,-1]))

        batch_data = np.concatenate(batch_data,axis=0)
        batch_rul = np.array(batch_rul)
        batch_rul_encoding = np.concatenate(batch_rul_encoding,axis=0)

        batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
        batch_rul_encoding = Variable(torch.from_numpy(batch_rul_encoding).type(torch.FloatTensor)).cuda()

        optimizer.zero_grad()
        output = self.network(batch_data)
        loss, rul_loss = self._custom_loss(output, batch_rul_encoding, batch_rul)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()        #empty useless variable
        return loss.data, rul_loss

    def _evaluate(self, test_iter):
        self.network.eval()
        batch_data = []
        batch_rul = []
        batch_rul_encoding = []

        train_num = [self.batch_size//len(test_iter[0])]*len(test_iter[0])
        for x in random.sample(range(0,len(test_iter[0])),self.batch_size % len(test_iter[0])):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            random_len = np.sort(np.random.randint(round(test_iter[0][i].shape[0]*0.8), test_iter[0][i].shape[0], size=x))
            for one_len in random_len:
                one_feed_data = self._get_one_feed_data(test_iter[0][i],0,one_len)
                # one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                # one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])])
                batch_data.append(one_feed_data)
                batch_rul.append(test_iter[1][i][one_len])
                batch_rul_encoding.append(test_iter[2][i][one_len,:].reshape([1,-1]))

        batch_data = np.concatenate(batch_data,axis=0)
        batch_rul = np.array(batch_rul)
        batch_rul_encoding = np.concatenate(batch_rul_encoding,axis=0)

        batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
        batch_rul_encoding = Variable(torch.from_numpy(batch_rul_encoding).type(torch.FloatTensor)).cuda()

        output = self.network(batch_data)
        loss, rul_loss = self._custom_loss(output, batch_rul_encoding, batch_rul)
        return loss, rul_loss

    def _custom_loss(self, output, batch_rul_encoding, batch_rul):
        rul_encoding_loss = nn.functional.mse_loss(output, batch_rul_encoding)
        # rul_encoding_loss = torch.mean((output - batch_rul_encoding)**2 * ((1-batch_rul_encoding)*0.5+0.5))
        real_rul_mse = 0
        return rul_encoding_loss, real_rul_mse

    def test_all(self,select):
        self.network = torch.load('./model/newest_rul_model')
        self.network.eval()
        dataset = DataSet.load_dataset("phm_feature")
        test_iter = self._preprocess(dataset,select)
        result = []
        for x in test_iter[0]:
            one_result_list = []
            for i in range(32,x.shape[0],self.batch_size):
                batch_data_list = []
                for j in range(min(self.batch_size,x.shape[0]-i)):
                    one_feed_data = self._get_one_feed_data(x,0,i+j)
                    # one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                    # one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])],axis=0)
                    batch_data_list.append(one_feed_data)

                batch_data = np.concatenate(batch_data_list,axis=0)
                batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
                output = self.network(batch_data)

                # real_rul = self.rul_decoder(output[:,1:])
                # real_rul = (torch.asin(real_rul) + math.pi/2) / math.pi * 10000
                # output = torch.cat([output,real_rul],dim=1)
                one_result_list.append(output.data.cpu().numpy())
            result.append(np.concatenate(one_result_list))

        with open('rul_target.pkl', 'wb') as f:
            pickle.dump(test_iter,f)
        with open('rul_result.pkl', 'wb') as f:
            pickle.dump(result,f)
        print('test result has been saved')

    def _get_one_feed_data(self, indata, startidx, limitlen):
        data = []
        limitlen = limitlen
        for i in range(6,-1,-1):
            position = -np.ones([32])
            temp_GCD = limitlen // (2**i)
            if temp_GCD > 0:
                temp_idx = np.arange(temp_GCD,max(-1,temp_GCD-32),-1)
                position[0:temp_idx.shape[0]] = temp_idx[::-1]*(2**i)
                temp_data = indata[temp_idx[::-1]*(2**i) + startidx,:]
                temp_data = np.concatenate([temp_data,np.zeros([32-temp_data.shape[0],temp_data.shape[1]])],axis=0)
                temp_data = self._position_encoding(temp_data,position)
                data.append(temp_data.reshape([1,1,32,-1]))
            else:
                temp_data = np.zeros([1,1,32,indata.shape[1]+self.position_encoding_size])
                # temp_data = np.zeros([1,1,32,indata.shape[1]])
                data.append(temp_data)
                # idx.append(temp_idx[::-1]*(2**i))

        data = np.concatenate(data,axis=1)
        # idx = np.concatenate(idx,axis=0)
        return data

    def _position_encoding(self, data, position = None):
        pe = np.zeros([data.shape[0], self.position_encoding_size])
        if isinstance(position,np.ndarray):
            position = position.reshape([-1,1])
        else:
            position = np.arange(0, data.shape[0]).reshape([-1,1])
        div_term = np.exp(np.arange(0, self.position_encoding_size, 2) *
                             -(np.log(10000.0) / self.position_encoding_size))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe[(position==-1).reshape([-1,]),:] = 0
        return np.concatenate([data,pe],axis=1)

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = RULPredict()
    p.Begin()
    p.test_all('test')