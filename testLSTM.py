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

class testLSTM(nn.Module):
    def __init__(self, in_size):
        super(testLSTM,self).__init__()
        self.net = nn.LSTM(in_size,128,2,batch_first=True)
        self.FC = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128,1),
            nn.ReLU()
        )

    def forward(self, x, x_len):
        batch_size = x.size(0)
        x = nn.utils.rnn.pack_padded_sequence(x,x_len,batch_first=True,enforce_sorted=False)
        out,h = self.net(x)
        h = h[0]
        out = self.FC(h[-1,:,:].view(batch_size,-1))
        return out

class Process():
    def __init__(self):
        self.hidden_size = 128
        self.epochs = 500
        self.batch_size = 128
        self.lr = 1e-3
        self.optimizer = optim.Adam
        self.max_data_len = 64*7
        self.feature_size = 128
        self.position_encoding_size = 8
        self.network = testLSTM(self.feature_size+self.position_encoding_size).cuda()

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

    def Begin(self):
        dataset = DataSet.load_dataset("phm_feature")
        train_iter = self._preprocess(dataset,'test')
        test_iter = self._preprocess(dataset,'train')
        optimizer = self.optimizer(self.network.parameters(),lr=self.lr)

        log = OrderedDict()
        log['train_rul_loss'] = []
        log['val_rul_loss'] = []

        for e in range(1,self.epochs+1):
            train_loss = self._fit(train_iter,optimizer)
            val_loss = self._evaluate(test_iter)

            print("[Epoch:%d][train_loss:%.4e] \
                    [val_loss:%.4e]" 
                % (e,train_loss,
                    val_loss))
            log['train_rul_loss'].append(float(train_loss))
            log['val_rul_loss'].append(float(val_loss))
            pd.DataFrame(log).to_csv('./model/log.csv',index=False)
            torch.save(self.network, './model/newest_rul_model')

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
                batch_len = []
                for j in range(min(self.batch_size,x.shape[0]-i)):
                    one_feed_data,one_len = self._get_one_feed_data(x,0,i+j)
                    # one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                    # one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])],axis=0)
                    batch_data_list.append(Variable(torch.from_numpy(one_feed_data).type(torch.FloatTensor)).cuda())
                    batch_len.append(one_len)

                batch_len = np.array(batch_len)

                batch_data = nn.utils.rnn.pad_sequence(batch_data_list,batch_first=True)
                batch_len = Variable(torch.from_numpy(batch_len).type(torch.FloatTensor)).cuda()
                output = self.network(batch_data,batch_len)

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

        shang = limitlen // 64
        yushu = limitlen % 64
        if yushu >= 32:
            shang += 1
        
        idx = np.concatenate([
            np.arange(shang) * 64,
            np.arange((shang-1)*64,limitlen)
        ],axis=0)

        position = idx
        data = indata[idx.astype(np.int),]
        data = self._position_encoding(data,position)
        return data, data.shape[0]

    def _fit(self, train_iter, optimizer):
        self.network.train()
        batch_data = []
        batch_rul = []
        batch_rul_encoding = []
        batch_sequence_len = []

        train_num = [self.batch_size//len(train_iter[0])]*len(train_iter[0])
        for x in random.sample(range(0,len(train_iter[0])),self.batch_size % len(train_iter[0])):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            random_end = np.random.randint(round(train_iter[0][i].shape[0]*0.7), round(train_iter[0][i].shape[0]*1), size=x)
            random_start = np.random.randint(0,round(train_iter[0][i].shape[0]*0.2), size=x)
            random_len = random_end - random_start
            for j in range(random_len.shape[0]):
                one_feed_data,one_len = self._get_one_feed_data(train_iter[0][i], random_start[j], random_len[j])
                batch_data.append(Variable(torch.from_numpy(one_feed_data).type(torch.FloatTensor)).cuda())
                batch_rul.append(train_iter[1][i][random_end[j]])
                batch_rul_encoding.append(train_iter[2][i][random_end[j],:].reshape([1,-1]))
                batch_sequence_len.append(one_len)

        batch_rul = np.array(batch_rul)
        batch_rul_encoding = np.concatenate(batch_rul_encoding,axis=0)
        batch_sequence_len = np.array(batch_sequence_len)

        batch_data = nn.utils.rnn.pad_sequence(batch_data,batch_first=True)
        batch_rul_encoding = Variable(torch.from_numpy(batch_rul_encoding).type(torch.FloatTensor)).cuda()
        batch_sequence_len = Variable(torch.from_numpy(batch_sequence_len).type(torch.FloatTensor)).cuda()

        optimizer.zero_grad()
        output = self.network(batch_data,batch_sequence_len)
        loss = self._custom_loss(output, batch_rul_encoding, batch_rul)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()        #empty useless variable
        return loss.data.cpu().numpy()

    def _evaluate(self, test_iter):
        self.network.eval()
        batch_data = []
        batch_rul = []
        batch_rul_encoding = []
        batch_sequence_len = []

        train_num = [self.batch_size//len(test_iter[0])]*len(test_iter[0])
        for x in random.sample(range(0,len(test_iter[0])),self.batch_size % len(test_iter[0])):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            random_end = np.random.randint(round(test_iter[0][i].shape[0]*0.9), round(test_iter[0][i].shape[0]*1), size=x)
            random_start = np.zeros([x,])
            random_len = random_end - random_start
            for j in range(random_len.shape[0]):
                one_feed_data,one_len = self._get_one_feed_data(test_iter[0][i], random_start[j], random_len[j])
                batch_data.append(Variable(torch.from_numpy(one_feed_data).type(torch.FloatTensor)).cuda())
                batch_rul.append(test_iter[1][i][random_end[j]])
                batch_rul_encoding.append(test_iter[2][i][random_end[j],:].reshape([1,-1]))
                batch_sequence_len.append(one_len)

        batch_rul = np.array(batch_rul)
        batch_rul_encoding = np.concatenate(batch_rul_encoding,axis=0)
        batch_sequence_len = np.array(batch_sequence_len)

        batch_data = nn.utils.rnn.pad_sequence(batch_data,batch_first=True)
        batch_rul_encoding = Variable(torch.from_numpy(batch_rul_encoding).type(torch.FloatTensor)).cuda()
        batch_sequence_len = Variable(torch.from_numpy(batch_sequence_len).type(torch.FloatTensor)).cuda()

        output = self.network(batch_data,batch_sequence_len)
        loss = self._custom_loss(output, batch_rul_encoding, batch_rul)
        return loss.data.cpu().numpy()

    def _custom_loss(self, output, batch_rul_encoding, batch_rul):
        return nn.functional.mse_loss(output, batch_rul_encoding)

if __name__ == '__main__':
    torch.backends.cudnn.enabled=False
    process = Process()
    process.Begin()
    process.test_all('test')