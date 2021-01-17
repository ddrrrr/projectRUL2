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
import math
# from testdecoder import PositionDecoder
# from rul_encoder import RUL_Decoder, RUL_Encoder
# from binTransfer import Bin_Encoder, Bin_Decoder

class Attention(nn.Module):
    def __init__(self, in_size, out_size, if_res=False):
        super(Attention, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.if_res = if_res
        self.Qw = nn.Linear(self.in_size, self.out_size, bias=False)
        self.Kw = nn.Linear(self.in_size, self.out_size, bias=False)
        self.Vw = nn.Linear(self.in_size, self.out_size, bias=False)
        self.linear = nn.Sequential(
            nn.LayerNorm(self.out_size),
            nn.Linear(self.out_size,self.out_size),
            nn.ReLU(),
            nn.Linear(self.out_size,self.out_size),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = x.transpose(0,1)    # [B*T*H]
        # x = F.dropout(x,p=0.3)
        q = self.Qw(x)  
        k = self.Kw(x)
        v = self.Vw(x)
        attn = torch.bmm(q / self.out_size**0.5, k.transpose(1,2))  # [B*T*T]
        attn = F.softmax(attn,dim=-1)
        res = F.dropout(attn.bmm(v), p=0.5)  # [B*T*H]
        output = self.linear(res)
        if self.if_res:
            output += res
        return output.transpose(0,1)

class Last_Atten(nn.Module):
    def __init__(self, in_size, out_size):
        super(Last_Atten, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.Kw = nn.Linear(self.in_size, self.out_size, bias=False)
        self.Vw = nn.Linear(self.in_size, self.out_size, bias=False)
        self.q = nn.Parameter(torch.rand(self.out_size))
        stdv = 1. / math.sqrt(self.q.size(0))
        self.q.data.uniform_(-stdv, stdv)
        self.linear = nn.Sequential(
            nn.LayerNorm(self.out_size),
            nn.Linear(self.out_size,self.out_size),
            nn.ReLU(),
            nn.Linear(self.out_size,self.out_size),
            nn.Dropout(0.5)
        )
        # self.layernorm = nn.LayerNorm(self.out_size,eps=1e-6)

    def forward(self, x):
        x = x.transpose(0, 1)  # [B*T*H]
        q = self.q.repeat(x.size(0),1).unsqueeze(1) # [B*1*T]
        k = self.Kw(x)
        v = self.Vw(x)
        attn = torch.bmm(q / self.out_size**0.5,k.transpose(1,2)).squeeze(1)
        attn = F.softmax(attn,dim=1).unsqueeze(1)
        output = attn.bmm(v)
        output = self.linear(output)
        return output

    # def score(self, x):
    #     # [B*T*I]->[B*T*H]
    #     energy = self.attn(x)
    #     energy = energy.transpose(1, 2)  # [B*H*T]
    #     v = self.v.repeat(x.size(0), 1).unsqueeze(1)  # [B*1*H]
    #     energy = torch.bmm(v, energy)  # [B*1*T]
    #     return energy.squeeze(1)  # [B*T]

class BiLSTM(nn.Module):
    def __init__(self,in_size,hidden_size,rul_size,n_layers=1):
        super(BiLSTM,self).__init__()
        self.hidden_size = hidden_size
        # self.rnn = nn.LSTM(in_size,hidden_size,n_layers,bidirectional=True)
        self.atten = nn.Sequential(
            Attention(in_size,128,True),
            # Attention(128,128,True),
            # Attention(96,128,True),
            Attention(128,128,True),
            Attention(128,128,True),
            Last_Atten(128,128)
        )
        self.posibility_predict = nn.Sequential(
            nn.Linear(128,64),
            nn.PReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        self.rul_predict = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128,rul_size),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, seq_len=None):
        # x = self.dropout(x)
        rnn_output = self.atten(x)

        rnn_output = rnn_output.view(rnn_output.size(0),-1) # size: (b,hidden_size*2)
        posibility = self.posibility_predict(rnn_output)
        rul = self.rul_predict(rnn_output)
        output = torch.cat([posibility,rul],dim=1)

        return output

# class RegLSTM(nn.Module):
#     def __init__(self):
#         super(RegLSTM,self).__init__()


class RULProdict():
    def __init__(self):
        self.hidden_size = 128
        self.epochs = 2000
        self.batch_size = 64
        self.lr = 1e-3
        self.optimizer = optim.Adam
        self.max_data_len = 32*7
        self.feature_size = 32
        self.position_encoding_size = 8
        self.network = BiLSTM(self.feature_size+self.position_encoding_size,self.hidden_size,1).cuda()
        # self.rul_decoder = torch.load('PositionDecoder.pkl')
        # self.rul_decoder.eval()

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
        # state = dataset.get_value('state',condition={'bearing_name':_select})
        # posibility = dataset.get_value('posibility',condition={'bearing_name':_select})
        rul = dataset.get_value('rul',condition={'bearing_name':_select})
        rul_encoding = dataset.get_value('rul_encoding',condition={'bearing_name':_select})

        output = [data,rul,rul_encoding]
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
        # temp_rul[temp_healthlabel==0] = -1
        # temp_rul = np.sin(temp_rul * np.pi / 2 / 5000)

        # pe = np.sin(temp_rul / 3500 * np.pi - np.pi/2).reshape([-1,1])
        pe = (1 - np.exp(-temp_rul / 500)).reshape([-1,1])
        # pe = np.zeros([temp_rul.shape[0], 12])
        # for i,x in enumerate(temp_rul):
        #     pe[i,:] = Bin_Encoder(min(2**12-1,round(x)),12)
        
        # pe = np.zeros([temp_rul.shape[0], self.position_encoding_size])
        # position = temp_rul.reshape([-1,1])
        # div_term = np.exp(np.arange(0, self.position_encoding_size, 2) *
        #                      -(np.log(10000.0) / self.position_encoding_size))
        # pe[:, 0::2] = np.sin(position * div_term)
        # pe[:, 1::2] = np.cos(position * div_term)

        return temp_one_data ,temp_healthlabel, temp_posibility, temp_rul, pe

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
        feature_dataset = DataSet(name='phm_feature',index=['bearing_name','feature','state','posibility','rul','rul_encoding'])

        for i in range(len(bearing_data)):
            temp_one_data = bearing_data[i].transpose(0,2,1)
            _,state,posibility,rul,rul_encoding = self._GenHealthLabel(temp_one_data,bearing_rul[i])
            temp_one_data = self._fft(temp_one_data)
            length_data = temp_one_data.shape[2]
            temp_one_data = (temp_one_data - np.repeat(np.min(temp_one_data, axis=2, keepdims=True),length_data,axis=2)) \
                / np.repeat((np.max(temp_one_data,axis=2,keepdims=True) - np.min(temp_one_data,axis=2,keepdims=True)),length_data,axis=2)
            # temp_one_data = temp_one_data*2-1
            temp_one_data = Variable(torch.from_numpy(temp_one_data.copy()).type(torch.FloatTensor)).cuda() 

            temp_one_feature_list = []
            for j in range(temp_one_data.shape[0]//64+1):
                _,_,temp_one_fea = feature_net(temp_one_data[j*64:min(j*64+64,temp_one_data.shape[0]-1)])
                temp_one_feature_list.append(temp_one_fea.data.cpu().numpy())
            
            temp_one_feature = np.vstack(temp_one_feature_list)
            # temp_one_feature = temp_one_data.reshape([temp_one_data.shape[0],-1])
            # temp_one_feature = self._position_encoding(temp_one_feature)
            feature_dataset.append([bearing_name[i],temp_one_feature,state,posibility,rul,rul_encoding])

        feature_dataset.save()

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

            print("[Epoch:%d][train_loss:%.4e][train_po_loss:%.4e][train_rul_loss:%.4e] \
                    [val_loss:%.4e][val_po_loss:%.4e][val_rul_loss:%.4e]" 
                % (e,train_loss[0],train_loss[1],train_loss[2],
                    val_loss[0],val_loss[1],val_loss[2]))
            log['train_state_loss'].append(float(train_loss[0]))
            log['train_rul_loss'].append(float(train_loss[2]))
            log['val_state_loss'].append(float(val_loss[0]))
            log['val_rul_loss'].append(float(val_loss[2]))
            pd.DataFrame(log).to_csv('./model/log.csv',index=False)
            # if float(val_loss) == min(log['train_loss']):
            #     torch.save(self.network, './model/rul_model')
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
                for j in range(min(self.batch_size,x.shape[0]-i)):
                    one_feed_data = self._get_one_feed_data(x,0,i+j)
                    one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                    # one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])],axis=0)
                    batch_data_list.append(one_feed_data)

                batch_data = np.concatenate(batch_data_list,axis=1)
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
        position = -np.ones([64*7])
        for i in range(6,-1,-1):
            temp_GCD = limitlen // (2**i)
            if temp_GCD > 0:
                temp_idx = np.arange(temp_GCD,max(-1,temp_GCD-64),-1)
                position[(6-i)*64:(6-i)*64+temp_idx.shape[0]] = temp_idx[::-1]*(2**i)
                temp_data = indata[temp_idx[::-1]*(2**i) + startidx,:]
                temp_data = np.concatenate([temp_data,np.zeros([64-temp_data.shape[0],temp_data.shape[1]])],axis=0)
                data.append(temp_data)
            else:
                temp_data = np.zeros([64,indata.shape[1]])
                data.append(temp_data)
                # idx.append(temp_idx[::-1]*(2**i))

        data = np.concatenate(data,axis=0)
        data = self._position_encoding(data,position)
        # idx = np.concatenate(idx,axis=0)
        return data

    def _fit(self, train_iter, optimizer):
        self.network.train()
        batch_data = []
        batch_state = []
        batch_posibility = []
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
                one_feed_data = self._get_one_feed_data(train_iter[0][i], random_start[j], random_len[j])
                one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                # one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])],axis=0)
                batch_data.append(one_feed_data)
                # batch_state.append(train_iter[1][i][random_end[j]])
                # batch_posibility.append(train_iter[2][i][random_end[j]])
                batch_rul.append(train_iter[1][i][random_end[j]])
                batch_rul_encoding.append(train_iter[2][i][random_end[j],:].reshape([1,-1]))
                batch_sequence_len.append(random_end[j])

        batch_data = np.concatenate(batch_data,axis=1)
        # batch_state = np.array(batch_state)
        # batch_posibility = np.array(batch_posibility)
        batch_rul = np.array(batch_rul)
        batch_rul_encoding = np.concatenate(batch_rul_encoding,axis=0)
        batch_sequence_len = np.array(batch_sequence_len)

        # 按长度进行从大到小排列
        # new_idx = np.argsort(-batch_sequence_len)
        # batch_data = batch_data[:,new_idx,:]
        # batch_state = batch_state[new_idx]
        # batch_posibility = batch_posibility[new_idx]
        # batch_rul_encoding = batch_rul_encoding[new_idx]
        # batch_sequence_len = batch_sequence_len[new_idx]

        batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
        # batch_state = Variable(torch.from_numpy(batch_state).type(torch.FloatTensor)).cuda()
        # batch_posibility = Variable(torch.from_numpy(batch_posibility).type(torch.FloatTensor)).cuda()
        batch_rul_encoding = Variable(torch.from_numpy(batch_rul_encoding).type(torch.FloatTensor)).cuda()

        optimizer.zero_grad()
        output = self.network(batch_data,batch_sequence_len)
        loss, state_loss, rul_loss = self._custom_loss(output, batch_rul_encoding, batch_rul)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()        #empty useless variable
        return loss.data, state_loss.data, rul_loss

    def _evaluate(self, test_iter):
        self.network.eval()
        batch_data = []
        batch_state = []
        batch_posibility = []
        batch_rul = []
        batch_rul_encoding = []
        batch_sequence_len = []

        train_num = [self.batch_size//len(test_iter[0])]*len(test_iter[0])
        for x in random.sample(range(0,len(test_iter[0])),self.batch_size % len(test_iter[0])):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            random_len = np.sort(np.random.randint(round(test_iter[0][i].shape[0]*0.8), test_iter[0][i].shape[0], size=x))
            for one_len in random_len:
                one_feed_data = self._get_one_feed_data(test_iter[0][i],0,one_len)
                one_feed_data = np.reshape(one_feed_data,[one_feed_data.shape[0],1,-1])
                # one_feed_data = np.concatenate([one_feed_data,np.zeros([self.max_data_len - one_feed_data.shape[0], 1, self.feature_size + self.position_encoding_size])])
                batch_data.append(one_feed_data)
                # batch_state.append(test_iter[1][i][one_len])
                # batch_posibility.append(test_iter[2][i][one_len])
                batch_rul.append(test_iter[1][i][one_len])
                batch_rul_encoding.append(test_iter[2][i][one_len,:].reshape([1,-1]))
                batch_sequence_len.append(one_len)

        batch_data = np.concatenate(batch_data,axis=1)
        # batch_state = np.array(batch_state)
        # batch_posibility = np.array(batch_posibility)
        batch_rul = np.array(batch_rul)
        batch_rul_encoding = np.concatenate(batch_rul_encoding,axis=0)

        batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
        # batch_state = Variable(torch.from_numpy(batch_state).type(torch.FloatTensor)).cuda()
        # batch_posibility = Variable(torch.from_numpy(batch_posibility).type(torch.FloatTensor)).cuda()
        batch_rul_encoding = Variable(torch.from_numpy(batch_rul_encoding).type(torch.FloatTensor)).cuda()

        output = self.network(batch_data, batch_sequence_len)
        loss, state_loss, rul_loss = self._custom_loss(output, batch_rul_encoding, batch_rul)
        return loss, state_loss.data, rul_loss

    # def _cal_real_rul(self, rul_encoding_tensor):
    #     real_rul = self.rul_decoder(rul_encoding_tensor).data.cpu().numpy()
    #     real_rul = np.arcsin(real_rul) + np.pi/2
    #     real_rul = real_rul/np.pi * 10000
    #     return real_rul.reshape([-1])

    def _custom_loss(self,output, batch_rul_encoding, batch_rul):
        # posibility_loss = torch.mean(torch.cos(output[:,0]*math.pi/2))
        # posibility_loss = torch.mean(-torch.log(output[:,0]))
        # posibility_repeat = output[:,0].view(-1,1).repeat(1,self.position_encoding_size)
        # predict_rul = posibility_repeat * output[:,1:] + (1-posibility_repeat) * batch_rul_encoding
        # rul_encoding_loss = nn.functional.mse_loss(predict_rul,batch_rul_encoding)
        # rul_encoding_loss = nn.functional.mse_loss(output[:,1:],batch_rul_encoding)
        # sgn = torch.sign(output[:,1:] - batch_rul_encoding)
        # rul_encoding_loss = torch.mean((output[:,1:] - batch_rul_encoding)**abs(sgn-3) * ((1-batch_rul_encoding)*0.5+0.5)) 
        rul_encoding_loss = torch.mean((output[:,1:] - batch_rul_encoding)**2)

        posibility_loss = nn.functional.mse_loss(((batch_rul_encoding - output[:,1:]) / (output[:,1:]+1e-6)),output[:,0])

        l2_loss = Variable(torch.zeros(1)).cuda()
        for name, param in self.network.named_parameters():
            if 'bias' not in name:
                l2_loss += torch.norm(param)
        # rul_encoding_loss = torch.mean(torch.sum((predict_rul-batch_rul_encoding)**2,dim=1))
        # rul_encoding_square = output[:,1:]**2
        # rul_encoding_selfloss = Variable(torch.ones([rul_encoding_square.size(0), int(rul_encoding_square.size(1)/2)])).cuda() \
        #     - rul_encoding_square[:,0::2] - rul_encoding_square[:,1::2]
        # rul_encoding_selfloss = torch.mean(rul_encoding_selfloss**2)

        # rul_encoding_loss = torch.mean(torch.sum((output[:,1:] - batch_rul_encoding)**2,dim=1) * output[:,0].view(-1))
        cal_loss = 0 * posibility_loss + rul_encoding_loss + 1e-4*l2_loss

        # real_rul = self.rul_decoder(output[:,1:]).data.cpu().numpy()
        # real_rul = np.arcsin(real_rul) + np.pi/2
        # real_rul = real_rul/np.pi * 10000
        # real_rul_mse = np.mean((batch_rul - real_rul.reshape([-1,]))**2)
        real_rul_mse = 0

        # state_loss = nn.functional.binary_cross_entropy(output[:,0].view(-1),batch_state)
        # predict_rul = (output[:,1]*output[:,2]).view(-1) + (1-output[:,1]).view(-1)*batch_rul_encoding
        # rul_loss = torch.sum(((predict_rul-batch_rul_encoding)**2) * batch_state) / torch.sum(batch_state)
        # posibility_loss = torch.mean(-torch.log(output[:,1]))
        # all_loss = state_loss + rul_loss + posibility_loss
        return cal_loss, posibility_loss, real_rul_mse


if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = RULProdict()
    # p._GenFeature('20200427encoder')
    p.Begin()
    p.test_all('test')