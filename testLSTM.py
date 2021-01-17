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
import time

class FreFea(nn.Module):
    def __init__(self):
        super(FreFea, self).__init__()
        self.FC = nn.Sequential(
            # nn.BatchNorm1d(2),
            nn.Linear(2560,256),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,64),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(-1,2560)
        return self.FC(x)

class testLSTM(nn.Module):
    def __init__(self, in_size):
        super(testLSTM,self).__init__()
        self.hidden_size = 256
        self.net = nn.LSTM(in_size,self.hidden_size,1,batch_first=True,bidirectional=True,dropout=0.3)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(self.hidden_size*2,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,1),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(64,1),
            # nn.ReLU(),
        )
        self.po = nn.Sequential(
            nn.Linear(self.hidden_size*2,1),
            nn.ReLU()
        )
        # self.encoding = FreFea()

    # def _position_encoding(self, data, position):
    #     pe = torch.zeros([1,data.size(1), self.position_encoding_size]).cuda()
    #     div_term = torch.exp(torch.arange(0, self.position_encoding_size, 2).cuda() *
    #                             -(torch.log(torch.tensor(10000)).cuda() / self.position_encoding_size))
    #     pe[:,:,0::2] = torch.sin(position * div_term)
    #     pe[:,:,1::2] = torch.cos(position * div_term)
    #     return torch.cat([data,pe],dim=2)

    def forward(self, x, x_position, x_len):
        # batch_size = x.size(0)
        # x = nn.functional.dropout(x,p=0.3)
        # for i in range(len(x)):
        #     temp_x = self.encoding(x[i])
        #     # temp_x = self._position_encoding(temp_x, x_position[i])
        #     temp_x = torch.cat([temp_x,x_position[i]],dim=1)
        #     x[i] = temp_x

        x = nn.utils.rnn.pad_sequence(x,batch_first=True)
        # x = nn.functional.dropout(x,p=0.3)
        x = nn.utils.rnn.pack_padded_sequence(x,x_len,batch_first=True,enforce_sorted=False)
        out,h = self.net(x)
        output, out_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        output = output.transpose(1,2)
        output = self.maxpool(output)
        output = output.view(-1,self.hidden_size*2)
        # h = h[0]
        # h = h.view(-1,2,2,self.hidden_size)
        # h = h[:,-1,:,:]
        # output = h.view(-1,self.hidden_size*2)
        out = self.FC(output)
        # po = self.po(output)
        return out

class Process():
    def __init__(self):
        self.hidden_size = 128
        self.epochs = 200
        self.batch_size = 128
        self.batches = 30
        self.lr = 1e-3
        self.optimizer = optim.Adam
        self.max_data_len = 64*7
        self.feature_size = 32
        self.position_encoding_size = 8
        self.network = testLSTM(self.feature_size+self.position_encoding_size).cuda()
        self.embed_drop_rate = 0.25
        self.gamma = 1
        self.max_sequence_len = 128

    def _preprocess(self, dataset, select):
        if select == 'train':
            _select = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
            # _select = ['Bearing1_1','Bearing2_1','Bearing3_1']
        elif select == 'test':
            _select = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                        'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                        'Bearing3_3']
            # _select = ['Bearing1_2','Bearing2_2','Bearing3_2']
        else:
            raise ValueError('wrong select!')

        data = dataset.get_value('feature',condition={'bearing_name':_select})
        rul = dataset.get_value('rul',condition={'bearing_name':_select})
        rul_encoding = dataset.get_value('rul_encoding',condition={'bearing_name':_select})

        output = [data,rul,rul_encoding]
        return output

    def _preprocess_2_dataset(self, dataset1, dataset2, select):
        if select == 'train':
            _select = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        elif select == 'test':
            _select = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                        'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                        'Bearing3_3']
        else:
            raise ValueError('wrong select!')
        data1 = dataset1.get_value('feature',condition={'bearing_name':_select})
        data2 = dataset2.get_value('feature',condition={'bearing_name':_select})
        data = []
        for i in range(len(data1)):
            data.append(np.concatenate([data1[i],data2[i]],axis=-1))
        rul = dataset1.get_value('rul',condition={'bearing_name':_select})
        rul_encoding = dataset1.get_value('rul_encoding',condition={'bearing_name':_select})

        output = [data,rul,rul_encoding]
        return output

    # def _preprocess(self, dataset, select):
    #     if select == 'train':
    #         # _select = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
    #         _select = ['Bearing1_1','Bearing2_1','Bearing3_1']
    #     elif select == 'test':
    #         # _select = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
    #         #             'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
    #         #             'Bearing3_3']
    #         _select = ['Bearing1_2','Bearing2_2','Bearing3_2']
    #     else:
    #         raise ValueError('wrong select!')

    #     data = dataset.get_value('data',condition={'bearing_name':_select})
    #     rul = dataset.get_value('RUL',condition={'bearing_name':_select})

    #     rul_encoding = []
    #     for i,x in enumerate(data):
    #         temp_one_data = x.transpose(0,2,1)
    #         temp_one_data = self._fft(temp_one_data)
    #         length_data = temp_one_data.shape[2]
    #         temp_one_data = (temp_one_data - np.repeat(np.min(temp_one_data, axis=2, keepdims=True),length_data,axis=2)) \
    #             / np.repeat((np.max(temp_one_data,axis=2,keepdims=True) - np.min(temp_one_data,axis=2,keepdims=True)),length_data,axis=2)

    #         data[i] = temp_one_data

    #         temp_rul = np.arange(temp_one_data.shape[0])[::-1] + rul[i]
    #         rul[i] = temp_rul

    #         pe = (np.log(temp_rul+1)).reshape([-1,1])
    #         rul_encoding.append(pe)

    #     output = [data,rul,rul_encoding]
    #     return output

    def _fft(self, data):
        fft_data = np.fft.fft(data,axis=2)/data.shape[2]
        fft_data = (np.abs(fft_data))**2
        fft_data = fft_data[:,:,1:1281]
        return fft_data

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
        # return pe

    def Begin(self):
        dataset = DataSet.load_dataset("phm_feature_16bitShuff")
        other_dataset = DataSet.load_dataset('phm_feature_16bitCOD')
        # self.network.encoding = torch.load('20200527encoder.pkl')
        # train_iter = self._preprocess(dataset,'train')
        # test_iter = self._preprocess(dataset,'test')
        train_iter = self._preprocess_2_dataset(dataset,other_dataset,'train')
        test_iter = self._preprocess_2_dataset(dataset,other_dataset,'test')
        optimizer = self.optimizer(self.network.parameters(),lr=self.lr,weight_decay=1e-4)
        train_idx = [2750,830,870,750,490,1450]

        # self.network = torch.load('./model/newest_rul_model')

        log = OrderedDict()
        log['train_rul_loss'] = []
        log['val_rul_loss'] = []

        for e in range(1,self.epochs+1):
            train_loss = 0
            for _ in range(self.batches):
                train_loss += self._fit(train_iter,optimizer,train_idx)
            val_loss = 0
            for _ in range(8):
                val_loss += self._evaluate(test_iter)
            val_loss /= 8
            # self.gamma *= 1.005
            # self.embed_drop_rate += 0.25/self.epochs

            print("[Epoch:%d][train_loss:%.4e] \
                    [val_loss:%.4e]" 
                % (e,train_loss/self.batches,
                    val_loss))
            log['train_rul_loss'].append(float(train_loss/self.batches))
            log['val_rul_loss'].append(float(val_loss))
            pd.DataFrame(log).to_csv('./model/log.csv',index=False)
            torch.save(self.network, './model/newest_rul_model')
            if float(val_loss) == min(log['val_rul_loss']):
                torch.save(self.network, './model/best_rul_model')

            if (e>10 and e%5==0):
                for i in range(len(train_idx)):
                    if train_idx[i] >=round(train_iter[0][i].shape[0]*0.2):
                        train_idx[i] = max(0,train_idx[i] - random.randint(0,round(train_iter[0][i].shape[0]*0.03)))

    def test_all(self,select):
        self.network = torch.load('./model/best_rul_model')
        self.network.eval()
        # dataset = DataSet.load_dataset("phm_feature")
        # test_iter = self._preprocess(dataset,select)
        dataset = DataSet.load_dataset("phm_feature_16bitShuff")
        other_dataset = DataSet.load_dataset('phm_feature_16bitCOD')
        test_iter = self._preprocess_2_dataset(dataset,other_dataset,select)

        result = []
        for x in test_iter[0]:
            one_result_list = []
            for temp_index in range(0,-1,-1):
                if x.shape[0] // 2**temp_index < 10:
                    continue
                temp_result_list = []
                temp_result_list.append(np.zeros([10*2**temp_index,1]))
                temp_idx = np.arange(x.shape[0]//2**temp_index)*2**temp_index
                for i in range(10,temp_idx.shape[0],self.batch_size):
                    batch_data_list = []
                    batch_len = []
                    batch_position = []
                    for j in range(min(self.batch_size,temp_idx.shape[0]-i)):
                        one_feed_data,one_len,position = self._get_one_feed_data(x,0,i+j,False)
                        # batch_idx = temp_idx[max(i+j-self.max_sequence_len,0):i+j]
                        # one_feed_data = x[batch_idx.astype(np.int),]
                        # one_feed_data = self._position_encoding(one_feed_data,batch_idx)
                        # one_len = one_feed_data.shape[0]

                        batch_data_list.append(Variable(torch.from_numpy(one_feed_data).type(torch.FloatTensor)).cuda())
                        batch_position.append(Variable(torch.from_numpy(position).type(torch.FloatTensor)).cuda())
                        batch_len.append(one_len)

                    batch_len = np.array(batch_len)
                    # batch_data = nn.utils.rnn.pad_sequence(batch_data_list,batch_first=True)
                    batch_len = Variable(torch.from_numpy(batch_len).type(torch.FloatTensor)).cuda()
                    output = self.network(batch_data_list,batch_position,batch_len)
                    # output = np.concatenate([output.data.cpu().numpy(),po.data.cpu().numpy()],axis=1)
                    temp_result_list.append(output.data.cpu().numpy().repeat(2**temp_index,axis=0))

                temp_result = np.concatenate(temp_result_list)
                one_result_list.append(temp_result[:x.shape[0]])
            result.append(np.concatenate(one_result_list,axis=1))

        with open('rul_target.pkl', 'wb') as f:
            pickle.dump(test_iter,f)
        with open('rul_result.pkl', 'wb') as f:
            pickle.dump(result,f)
        print('test result has been saved')

    def _test_time(self):
        self.network = torch.load('./model/best_rul_model')
        self.network.eval()
        dataset = DataSet.load_dataset("phm_feature_16bitShuff")
        other_dataset = DataSet.load_dataset('phm_feature_16bitCOD')
        test_iter = self._preprocess_2_dataset(dataset,other_dataset,'train')
        one_feed_data,one_len,position = self._get_one_feed_data(test_iter[0][0],0,2800,False)

        time_c = 0
        size = 100
        for i in range(size):

            time_s = time.time()
            test_one_feed_data=Variable(torch.from_numpy(one_feed_data).type(torch.FloatTensor)).cuda()
            test_position=Variable(torch.from_numpy(position).type(torch.FloatTensor)).cuda()
            test_one_len=Variable(torch.from_numpy(np.array(one_len).reshape([1])).type(torch.FloatTensor)).cuda()
            
            output = self.network([test_one_feed_data],test_position,test_one_len)
            time_e = time.time()
            print(time_e - time_s)
            time_c += time_e - time_s
        print('time_mean',time_c/size)


    def _get_one_feed_data(self, indata, startidx, limitlen, is_drop=True):
        # if is_drop and random.random()>0.5:
        #     indata = indata[::2].copy()
        #     startidx = int(round(startidx/2))
        #     limitlen = int(round(limitlen/2))
        #     if limitlen + startidx >= indata.shape[0]:
        #         limitlen = indata.shape[0] - startidx -1

        # 1,2,3,5,7,11各取长度不超过32的序列
        # temp_idx = []
        # count=0
        # for i in [1,2,3,5,7,11]:
        #     one_temp_idx = np.arange(count,count + 24*i,i)
        #     temp_idx.append(one_temp_idx)
        #     count = one_temp_idx[-1] + i
        # temp_idx = np.concatenate(temp_idx)
        # temp_idx = temp_idx[temp_idx<limitlen]
        # temp_idx = limitlen - temp_idx[::-1]
        # position = temp_idx
        # 1,2,4,8,16,32各取长度不超过32的序列

        # temp_idx = []
        # for i in range(6):
        #     temp_idx.append(np.arange(16*(2**i-1),16*(2**(i+1)-1),2**i))
        # temp_idx = np.concatenate(temp_idx)
        # temp_idx = temp_idx[temp_idx<limitlen]
        # temp_idx = limitlen - temp_idx[::-1]
        # position = temp_idx
        # 64,32,16,8,4,2,1各取长度不超过64的序列
        # temp_idx = []
        # for i in range(6,-1,-1):
        #     temp_GCD = limitlen // (2**i)
        #     if temp_GCD > 0:
        #         temp_idx.append(np.arange(temp_GCD,max(-1,temp_GCD-32),-1)[::-1]*(2**i))
        # temp_idx = np.concatenate(temp_idx)
        # position = temp_idx

        temp_idx = []
        count = limitlen
        for i in range(7):
            count = count - count % 2**i
            if count == 0:
                break
            temp_one_idx = -np.arange(0,min(16*2**i,count),2**i)[::-1]+count
            count = temp_one_idx[0]
            temp_idx.append(temp_one_idx)
        temp_idx.append(np.int32(np.zeros(1)))
        temp_idx = np.concatenate(temp_idx[::-1])
        position = temp_idx
        # 前面数据以64为间隔，后面取连续数据
        # data = []
        # limitlen = limitlen

        # shang = limitlen // 64
        # yushu = limitlen % 64
        # if yushu >= 32:
        #     shang += 1
        
        # temp_idx = np.concatenate([
        #     np.arange(shang) * 64,
        #     np.arange((shang-1)*64,limitlen)
        # ],axis=0)

        # position = temp_idx
        # idx += startidx
        # data = indata[idx.astype(np.int),]
        # data = self._position_encoding(data,position)

        # 随机取间隔为64,32,16,8,4,2,1的序列
        # max_index = 0
        # for i in range(4,-1,-1):
        #     temp_GCD = limitlen // (2**i)
        #     if temp_GCD >= 10:
        #         max_index = i
        #         break
        # temp_index = np.random.randint(0,max_index+1)
        # # temp_index = 0
        # temp_idx = np.arange(0,min(limitlen, self.max_sequence_len*(2**temp_index)),2**temp_index)
        # temp_idx = -temp_idx[::-1] + limitlen
        # position = temp_idx

        # 取距离最短的序列
        # idx = [0]
        # count = 0
        # while count < limitlen-1:
        #     temp_data = indata[count+1+startidx:min(indata.shape[0],count+64)+startidx,:].copy()
        #     temp_data -= indata[count+startidx,:].copy().reshape([1,-1]).repeat(temp_data.shape[0],axis=0)
        #     temp_data = np.mean(temp_data**2,axis=1)
        #     count += temp_data.argmin() +1
        #     idx.append(count)
        # temp_idx = np.array(idx)
        # position = temp_idx
        # if is_drop:
        #     random_idx = np.random.randint(0,limitlen-1,temp_idx.shape[0])
        #     random_m = np.random.random(temp_idx.shape[0])
        #     temp_idx[random_m<self.embed_drop_rate] = random_idx[random_m<self.embed_drop_rate]
        temp_idx += startidx
        data = indata[temp_idx.astype(np.int),].copy()
        data = self._position_encoding(data,position)
        if is_drop:
            radom_m = np.random.random(data.shape[0])
            data[radom_m<self.embed_drop_rate,:] = 0

        return data, data.shape[0], position

    def _fit(self, train_iter, optimizer,train_idx):
        self.network.train()
        batch_data = []
        batch_rul = []
        batch_rul_encoding = []
        batch_sequence_len = []
        batch_position = []

        train_num = [self.batch_size//len(train_iter[0])]*len(train_iter[0])
        for x in random.sample(range(0,len(train_iter[0])),self.batch_size % len(train_iter[0])):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            # random_end = np.random.randint(round(train_iter[0][i].shape[0]*0.6), round(train_iter[0][i].shape[0]*1-1), size=x)
            random_end = np.random.randint(train_idx[i],train_iter[0][i].shape[0],size=x)
            random_start = np.random.randint(0,max(10,train_idx[i]-32), size=x)
            random_len = random_end - random_start
            for j in range(random_len.shape[0]):
                one_feed_data,one_len,one_position = self._get_one_feed_data(train_iter[0][i], random_start[j], random_len[j])
                batch_data.append(Variable(torch.from_numpy(one_feed_data).type(torch.FloatTensor)).cuda())
                batch_position.append(Variable(torch.from_numpy(one_position).type(torch.FloatTensor)).cuda())
                batch_rul.append(train_iter[1][i][random_end[j]])
                batch_rul_encoding.append(train_iter[2][i][random_end[j]].reshape([1,-1]))
                batch_sequence_len.append(one_len)

        batch_rul = np.array(batch_rul).reshape([-1,1])
        batch_rul_encoding = np.concatenate(batch_rul_encoding,axis=0)
        rul_random = np.random.randn(batch_rul_encoding.shape[0]).reshape([-1,1]) *0.1
        batch_rul_encoding += rul_random
        batch_sequence_len = np.array(batch_sequence_len)

        # batch_data = nn.utils.rnn.pad_sequence(batch_data,batch_first=True)
        batch_rul_encoding = Variable(torch.from_numpy(batch_rul_encoding).type(torch.FloatTensor)).cuda()
        batch_sequence_len = Variable(torch.from_numpy(batch_sequence_len).type(torch.FloatTensor)).cuda()
        batch_rul = Variable(torch.from_numpy(batch_rul).type(torch.FloatTensor)).cuda()

        optimizer.zero_grad()
        output = self.network(batch_data,batch_position,batch_sequence_len)
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
        batch_position = []

        train_num = [self.batch_size//len(test_iter[0])]*len(test_iter[0])
        for x in random.sample(range(0,len(test_iter[0])),self.batch_size % len(test_iter[0])):
            train_num[x] += 1

        for i,x in enumerate(train_num):
            random_end = np.random.randint(round(test_iter[0][i].shape[0]*0.9), round(test_iter[0][i].shape[0]*1), size=x)
            random_start = np.zeros([x,])
            random_len = random_end - random_start
            for j in range(random_len.shape[0]):
                one_feed_data,one_len,one_position = self._get_one_feed_data(test_iter[0][i], int(random_start[j]), random_len[j],False)
                batch_data.append(Variable(torch.from_numpy(one_feed_data).type(torch.FloatTensor)).cuda())
                batch_position.append(Variable(torch.from_numpy(one_position).type(torch.FloatTensor)).cuda())
                batch_rul.append(test_iter[1][i][random_end[j]])
                batch_rul_encoding.append(test_iter[2][i][random_end[j]].reshape([1,-1]))
                batch_sequence_len.append(one_len)

        batch_rul = np.array(batch_rul).reshape([-1,1])
        batch_rul_encoding = np.concatenate(batch_rul_encoding,axis=0)
        batch_sequence_len = np.array(batch_sequence_len)

        # batch_data = nn.utils.rnn.pad_sequence(batch_data,batch_first=True)
        batch_rul_encoding = Variable(torch.from_numpy(batch_rul_encoding).type(torch.FloatTensor)).cuda()
        batch_sequence_len = Variable(torch.from_numpy(batch_sequence_len).type(torch.FloatTensor)).cuda()
        batch_rul = Variable(torch.from_numpy(batch_rul).type(torch.FloatTensor)).cuda()

        output = self.network(batch_data, batch_position,batch_sequence_len)
        loss = self._custom_loss(output, batch_rul_encoding, batch_rul)
        return loss.data.cpu().numpy()

    def _custom_loss(self, output, batch_rul_encoding, batch_rul):
        # posibility_loss = torch.mean(-torch.log(output[1]))
        # predict_rul = output[1] * output[0] + (1-output[1]) * batch_rul_encoding
        # loss = nn.functional.mse_loss(predict_rul, batch_rul_encoding) + self.gamma * posibility_loss
        loss = nn.functional.mse_loss(output , batch_rul_encoding)
        # real_wucha  = torch.abs(((10**output[0] - 1) - batch_rul) / (10**output[0] - 1+1e-1))
        # po_loss = torch.mean(nn.functional.relu(real_wucha - output[1]) + output[1])
        # loss += self.gamma * po_loss 
        return loss

if __name__ == '__main__':
    torch.backends.cudnn.enabled=False
    process = Process()
    # process.Begin() 
    # process.test_all('train')
    process._test_time()