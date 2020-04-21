import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet
from DIModel import Encoder, ResBlock
import pickle
import pandas as pd
from collections import OrderedDict

class BiLSTM(nn.Module):
    def __init__(self,in_size,hidden_size,n_layers=1):
        super(BiLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(in_size,hidden_size,n_layers,bidirectional=True)
        self.state_predict = nn.Sequential(
            nn.Linear(hidden_size*2,3),
            nn.Sigmoid()
        )
        self.rul_predict = nn.Sequential(
            nn.Linear(hidden_size*2,2),
            nn.ReLU()
        )

    def forward(self, x, seq_len=None):
        x,_ = self.rnn(x)
        # x = nn.utils.rnn.pack_padded_sequence(x,seq_len)
        x = x.permute(1,2,0)
        x = nn.MaxPool1d(7*32)(x)
        x = x.view(x.size(0),x.size(1)) # size: (b,hidden_size*2)
        state = self.state_predict(x)
        # rul = self.rul_predict(x)
        # output = torch.cat([state,rul],dim=1)
        output = state

        return output

class RULProdict():
    def __init__(self):
        self.hidden_size = 128
        self.epochs = 50
        self.batch_size = 64
        self.lr = 1e-3
        self.optimizer = optim.Adam
        self.max_data_len = 32*7
        self.feature_size = 64
        self.position_encoding_size = 8
        self.network = BiLSTM(self.feature_size + self.position_encoding_size,self.hidden_size).cuda()

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
        state = dataset.get_value('state',condition={'bearing_name':_select})
        posibility = dataset.get_value('posibility',condition={'bearing_name':_select})
        rul = dataset.get_value('rul',condition={'bearing_name':_select})

        output = [data,state,posibility,rul]
        return output

    def _GenHealthLabel(self, data, rul=0):
        temp_one_data = data
        length = temp_one_data.shape[2]

        temp_one_data = np.square(temp_one_data)
        temp_one_data = np.sum(temp_one_data,axis=2) / length
        temp_one_data = np.sqrt(temp_one_data)

        time_len = temp_one_data.shape[0]

        temp_mean = np.mean(temp_one_data[:time_len//2,:],axis=0)
        temp_std = np.std(temp_one_data[:time_len//2,:],axis=0)

        temp_healthlabel = np.zeros(temp_one_data.shape)
        temp_healthlabel[temp_one_data>(temp_mean+2.5*temp_std)] = 1
        temp_posibility = np.mean(temp_healthlabel, axis=1)
        temp_healthlabel = np.max(temp_healthlabel, axis=1)
        temp_rul = np.arange(temp_one_data.shape[0])[::-1] + rul
        temp_rul[temp_healthlabel==0] = -1
        temp_rul = np.sin(temp_rul * np.pi / 2 / 5000)

        return temp_one_data ,temp_healthlabel, temp_posibility, temp_rul

    def _fft(self, data):
        fft_data = np.fft.fft(data,axis=2)/data.shape[2]
        fft_data = (np.abs(fft_data))**2
        fft_data = fft_data[:,:,1:1281]
        return fft_data

    def _GenFeature(self, model_name):
        bearing_dataset = DataSet.load_dataset('phm_data')
        bearing_data = bearing_dataset.get_value('data')
        bearing_name = bearing_dataset.get_value('bearing_name')
        bearing_rul = bearing_dataset.get_value('RUL')

        feature_net = torch.load(model_name + '.pkl')
        feature_net.eval()
        feature_dataset = DataSet(name='phm_feature',index=['bearing_name','feature','state','posibility','rul'])

        for i in range(len(bearing_data)):
            temp_one_data = bearing_data[i].transpose(0,2,1)
            _,state,posibility,rul = self._GenHealthLabel(temp_one_data,bearing_rul[i])
            temp_one_data = self._fft(temp_one_data)
            length_data = temp_one_data.shape[2]
            temp_one_data = (temp_one_data - np.repeat(np.min(temp_one_data, axis=2, keepdims=True),length_data,axis=2)) \
                / np.repeat((np.max(temp_one_data,axis=2,keepdims=True) - np.min(temp_one_data,axis=2,keepdims=True)),length_data,axis=2)
            temp_one_data = Variable(torch.from_numpy(temp_one_data.copy()).type(torch.FloatTensor)).cuda() 

            temp_one_feature_list = []
            for j in range(temp_one_data.shape[0]//64):
                temp_one_feature_list.append(feature_net(temp_one_data[j*64:min(j*64+64,temp_one_data.shape[0]-1)]).data.cpu().numpy())
            
            temp_one_feature = np.vstack(temp_one_feature_list)
            temp_one_feature = self._position_encoding(temp_one_feature)
            feature_dataset.append([bearing_name[i],temp_one_feature,state,posibility,rul])

        feature_dataset.save()

    def _position_encoding(self, data):
        pe = np.zeros([data.shape[0], self.position_encoding_size])
        position = np.arange(0, data.shape[0]).reshape([-1,1])
        div_term = np.exp(np.arange(0, self.position_encoding_size, 2) *
                             -(np.log(10000.0) / self.position_encoding_size))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return np.concatenate([data,pe],axis=1)

    def Begin(self):
        dataset = DataSet.load_dataset("phm_feature")
        train_iter = self._preprocess(dataset,'train')
        test_iter = self._preprocess(dataset,'test')
        optimizer = self.optimizer(self.network.parameters(),lr=self.lr)

        log = OrderedDict()
        log['train_state_loss'] = []
        log['train_rul_loss'] = []
        log['val_state_loss'] = []
        log['val_rul_loss'] = []

        for e in range(1,self.epochs+1):
            train_loss = self._fit(train_iter,optimizer)
            val_loss = self._evaluate(test_iter)

            print("[Epoch:%d][train_state_loss:%.4e][train_rul_loss:%.4e][val_state_loss:%.4e][val_rul_loss:%.4e]"
                % (e,train_loss[0],train_loss[1],val_loss[0],val_loss[1]))
            log['train_state_loss'].append(float(train_loss[0]))
            log['train_rul_loss'].append(float(train_loss[1]))
            log['val_state_loss'].append(float(val_loss[0]))
            log['val_rul_loss'].append(float(val_loss[1]))
            pd.DataFrame(log).to_csv('./model/log.csv',index=False)
            # if float(val_loss) == min(log['train_loss']):
            #     torch.save(self.network, './model/rul_model')
            torch.save(self.network, './model/newest_rul_model')

    def test_all(self):
        self.network = torch.load('./model/newest_rul_model')
        self.network.eval()
        dataset = DataSet.load_dataset("phm_feature")
        test_iter = self._preprocess(dataset,'test')
        result = []
        for x in test_iter[0]:
            one_result_list = []
            for i in range(32,x.shape[0],self.batch_size):
                batch_data_list = []
                for j in range(min(self.batch_size,x.shape[0]-i)):
                    one_feed_data = self._get_one_feed_data(x,i+j)
                    one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                    one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])],axis=0)
                    batch_data_list.append(one_feed_data)

                batch_data = np.concatenate(batch_data_list,axis=1)
                batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
                output = self.network(batch_data)
                one_result_list.append(output.data.cpu().numpy())

            result.append(np.concatenate(one_result_list))

        with open('rul_target.pkl', 'wb') as f:
            pickle.dump(test_iter,f)
        with open('rul_result.pkl', 'wb') as f:
            pickle.dump(result,f)
        print('test result has been saved')

    def _get_one_feed_data(self, indata, limitlen):
        idx = []
        for i in range(6,-1,-1):
            temp_GCD = limitlen // (2**i)
            if temp_GCD > 0:
                temp_idx = np.arange(temp_GCD,max(-1,temp_GCD-32),-1)
                idx.append(temp_idx[::-1]*(2**i))

        idx = np.concatenate(idx,axis=0)
        return indata[idx,:]

    def _fit(self, train_iter, optimizer):
        self.network.train()
        batch_data = []
        batch_state = []
        batch_posibility = []
        batch_rul = []
        batch_sequence_len = []

        train_num = [self.batch_size//len(train_iter)]*len(train_iter)
        for x in random.sample(range(0,len(train_iter)),self.batch_size % len(train_iter)):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            random_len = np.sort(np.random.randint(32, train_iter[0][i].shape[0], size=x))
            for one_len in random_len:
                one_feed_data = self._get_one_feed_data(train_iter[0][i],one_len)
                one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])],axis=0)
                batch_data.append(one_feed_data)
                batch_state.append(train_iter[1][i][one_len])
                batch_posibility.append(train_iter[2][i][one_len])
                batch_rul.append(train_iter[3][i][one_len])
                batch_sequence_len.append(one_len)

        batch_data = np.concatenate(batch_data,axis=1)
        batch_state = np.array(batch_state)
        batch_posibility = np.array(batch_posibility)
        batch_rul = np.array(batch_rul)
        batch_sequence_len = np.array(batch_sequence_len)

        # 按长度进行从大到小排列
        new_idx = np.argsort(-batch_sequence_len)
        batch_data = batch_data[new_idx,:,:]
        batch_state = batch_state[new_idx]
        batch_posibility = batch_posibility[new_idx]
        batch_rul = batch_rul[new_idx]
        batch_sequence_len = batch_sequence_len[new_idx]

        batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
        batch_state = Variable(torch.from_numpy(batch_state).type(torch.FloatTensor)).cuda()
        batch_posibility = Variable(torch.from_numpy(batch_posibility).type(torch.FloatTensor)).cuda()
        batch_rul = Variable(torch.from_numpy(batch_rul).type(torch.FloatTensor)).cuda()

        optimizer.zero_grad()
        output = self.network(batch_data,batch_sequence_len)
        loss, state_loss, rul_loss = self._custom_loss(output, batch_state, batch_posibility, batch_rul)
        loss.backward()
        optimizer.step()
        return state_loss.data, rul_loss.data

    def _evaluate(self, test_iter):
        self.network.eval()
        batch_data = []
        batch_state = []
        batch_posibility = []
        batch_rul = []
        batch_sequence_len = []

        train_num = [self.batch_size//len(test_iter)]*len(test_iter)
        for x in random.sample(range(0,len(test_iter)),self.batch_size % len(test_iter)):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            random_len = np.sort(np.random.randint(32, test_iter[0][i].shape[0], size=x))
            for one_len in random_len:
                one_feed_data = self._get_one_feed_data(test_iter[0][i],one_len)
                one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])])
                batch_data.append(one_feed_data)
                batch_state.append(test_iter[1][i][one_len])
                batch_posibility.append(test_iter[2][i][one_len])
                batch_rul.append(test_iter[3][i][one_len])
                batch_sequence_len.append(one_len)

        batch_data = np.concatenate(batch_data,axis=1)
        batch_state = np.array(batch_state)
        batch_posibility = np.array(batch_posibility)
        batch_rul = np.array(batch_rul)

        batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
        batch_state = Variable(torch.from_numpy(batch_state).type(torch.FloatTensor)).cuda()
        batch_posibility = Variable(torch.from_numpy(batch_posibility).type(torch.FloatTensor)).cuda()
        batch_rul = Variable(torch.from_numpy(batch_rul).type(torch.FloatTensor)).cuda()

        output = self.network(batch_data, batch_sequence_len)
        loss, state_loss, rul_loss = self._custom_loss(output, batch_state, batch_posibility, batch_rul)
        return state_loss.data, rul_loss.data

    def _custom_loss(self,output, batch_state, batch_posibility, batch_rul):
        state_loss = nn.functional.binary_cross_entropy(output[:,0].view(-1),batch_state)
        predict_rul = (output[:,1]*output[:,2]).view(-1) + (1-output[:,1]).view(-1)*batch_rul
        rul_loss = torch.sum(((predict_rul-batch_rul)**2) * batch_state) / torch.sum(batch_state)
        posibility_loss = torch.mean(-torch.log(output[:,1]))
        all_loss = state_loss + rul_loss + posibility_loss
        return all_loss, state_loss, rul_loss


if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = RULProdict()
    p._GenFeature('20200413encoder')
    p.Begin()
    p.test_all()