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
import time

class TimeEncoder(nn.Module):
    def __init__(self,fea_size):
        super(TimeEncoder, self).__init__()
        self.fea_size = fea_size
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(2),
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
            nn.Tanh(),
        )

    def forward(self, x):
        # x [b*2*L*DataLen]
        return self.encoder(x).view(x.size(0),self.fea_size,-1).transpose(2,1)

class FreEncoder(nn.Module):
    def __init__(self,fea_size):
        super(FreEncoder, self).__init__()
        self.fea_size = fea_size
        self.encoder = nn.Sequential(
            nn.Conv1d(2560,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Conv1d(512,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Conv1d(128,fea_size,1),
            nn.BatchNorm1d(fea_size),
            nn.Tanh(),
        )
    def forward(self,x):
        # x [b*2*L*DataLen]
        x = x.transpose(2,3)
        x = x.contiguous().view(x.size(0),2560,-1)
        x = self.encoder(x)
        x = x.transpose(1,2)
        return x


class CDC(nn.Module):
    def __init__(self, timestep, batch_size, seq_len, fea_size):
        super(CDC,self).__init__()
        self.batch_size = batch_size
        self.fea_size = fea_size
        self.timestep = timestep
        self.seq_len = seq_len
        self.encoder = TimeEncoder(fea_size)
        # self.encoder = FreEncoder(fea_size)
        self.rnn_hidden_size = 256
        self.gru = nn.GRU(self.fea_size,self.rnn_hidden_size,1,bidirectional=False,batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(self.rnn_hidden_size,self.fea_size) for i in range(self.timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        self.sgru = nn.GRU(fea_size,self.rnn_hidden_size,1,batch_first=True)
        self.slinear = nn.Sequential(
            nn.Linear(self.rnn_hidden_size,1),
            nn.Sigmoid(),
        )

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

    def cal_shuffle_loss(self,z):
        shuffle_z = torch.empty((z.size(0),z.size(1),z.size(2))).float().cuda()
        for i in range(z.size(0)):
            idx = torch.randperm(z.size(1))
            shuffle_z[i] = z[i,idx,:].view(z.size(1),z.size(2))

        cat_z = torch.cat([z,shuffle_z],dim=0)
        cat_z = nn.functional.dropout(cat_z)
        target = torch.cat([torch.ones(z.size(0),1).cuda(),torch.zeros(z.size(0),1).cuda()],dim=0)
        idx = torch.randperm(cat_z.size(0))
        cat_z = cat_z[idx]
        target = target[idx]
        out,_ = self.sgru(cat_z,torch.zeros(1, cat_z.size(0), self.rnn_hidden_size).cuda())
        out = out[:,-1,:].view(cat_z.size(0),self.rnn_hidden_size)
        out = self.slinear(out)

        loss = nn.functional.binary_cross_entropy(out,target)
        acc = torch.sum(torch.eq(torch.round(out),target)).float() / out.size(0)

        return acc.item(), loss

    # def init_hidden(self, batch_size, use_gpu=True):
    #     if use_gpu: return torch.zeros(1, batch_size, 64).cuda()
    #     else: return torch.zeros(1, batch_size, 64)
    def forward(self, x):
        batch = x.size(0)
        t_samples = torch.randint(self.seq_len - self.timestep, size=(1,)).long()
        z = self.encoder(x)
        s_acc,s_loss = self.cal_shuffle_loss(z)
        # z = z.transpose(1,2)    # b*L*128
        nce = 0
        encode_samples = torch.empty((self.timestep, batch, self.fea_size)).float().cuda()
        for i in range(1,self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,self.fea_size)
        forward_seq = z[:,:t_samples+1,:]
        forward_seq = nn.functional.dropout(forward_seq)
        output,h = self.gru(forward_seq,torch.zeros(1, self.batch_size, self.rnn_hidden_size).cuda())
        c_t = output[:,t_samples,:].view(batch,self.rnn_hidden_size)
        pred = torch.empty((self.timestep, batch, self.fea_size)).float().cuda()
        for i in range(self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in range(self.timestep):
            total = torch.mm(nn.functional.relu(encode_samples[i]), nn.functional.relu(torch.transpose(pred[i],0,1)))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total),dim=0),torch.arange(0,batch).cuda()))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        accuracy = (accuracy + s_acc)/2
        nce += s_loss

        return accuracy, nce

class Shuff(nn.Module):
    def __init__(self,fea_size):
        super(Shuff,self).__init__()
        self.fea_size = fea_size
        self.hidden_size = 256
        self.gru = nn.GRU(fea_size,self.hidden_size,1,batch_first=True)
        self.encoder = FreEncoder(fea_size)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size,1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        z = self.encoder(x)
        shuffle_z = torch.empty((z.size(0),z.size(1),z.size(2))).float().cuda()
        for i in range(z.size(0)):
            idx = torch.randperm(z.size(1))
            shuffle_z[i] = z[i,idx,:].view(z.size(1),z.size(2))

        cat_z = torch.cat([z,shuffle_z],dim=0)
        cat_z = nn.functional.dropout(cat_z)
        target = torch.cat([torch.ones(z.size(0),1).cuda(),torch.zeros(z.size(0),1).cuda()],dim=0)
        idx = torch.randperm(cat_z.size(0))
        cat_z = cat_z[idx]
        target = target[idx]
        out,_ = self.gru(cat_z,torch.zeros(1, cat_z.size(0), self.hidden_size).cuda())
        out = out[:,-1,:].view(cat_z.size(0),self.hidden_size)
        out = self.linear(out)

        loss = nn.functional.binary_cross_entropy(out,target)
        acc = torch.sum(torch.eq(torch.round(out),target)).float() / out.size(0)
        return acc.item(),loss

class Shuff2(nn.Module):
    def __init__(self,fea_size,seq_len):
        super(Shuff2,self).__init__()
        self.fea_size = fea_size
        self.hidden_size = 256
        self.seq_len = seq_len
        self.D = nn.Sequential(
            nn.Conv1d(self.seq_len,64,3,1,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,1,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128,self.hidden_size,3,1,1),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # self.encoder = FreEncoder(fea_size)
        self.encoder = TimeEncoder(fea_size)
        self.linear = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.hidden_size*self.fea_size//8,1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        z = self.encoder(x)
        shuffle_z = torch.empty((z.size(0),z.size(1),z.size(2))).float().cuda()
        for i in range(z.size(0)):
            idx = torch.randperm(z.size(1))
            shuffle_z[i] = z[i,idx,:].view(z.size(1),z.size(2))

        cat_z = torch.cat([z,shuffle_z],dim=0)
        # cat_z = nn.functional.dropout(cat_z)
        target = torch.cat([torch.ones(z.size(0),1).cuda(),torch.zeros(z.size(0),1).cuda()],dim=0)
        idx = torch.randperm(cat_z.size(0))
        cat_z = cat_z[idx]
        target = target[idx]
        cat_z = nn.functional.dropout(cat_z)
        out = self.D(cat_z)
        out = out.view(cat_z.size(0),-1)
        out = self.linear(out)

        loss = nn.functional.binary_cross_entropy(out,target)
        acc = torch.sum(torch.eq(torch.round(out),target)).float() / out.size(0)
        return acc.item(),loss


class Process():
    def __init__(self):
        self.dataset = DataSet.load_dataset(name = 'phm_data')
        self.lr = 1e-3
        self.optimizer = optim.Adam
        self.epochs = 200
        self.batches = 50
        self.batch_size = 17
        self.seq_len = 32
        self.fea_size = 16
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']

        self.network = CDC(8,self.batch_size,self.seq_len,self.fea_size).cuda()
        # self.network = Shuff(self.fea_size).cuda()
        # self.network = Shuff2(self.fea_size,self.seq_len).cuda()

    def Begin(self):
        train_iter = self._preprocess('all')
        test_iter = self._preprocess('test')
        optimizer = self.optimizer(self.network.parameters(),lr=self.lr,
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
        train_idx = [2750,830,870,750,490,1450]
        for x in (test_iter):
            train_idx.append(x.shape[0]-20)

        for e in range(1, self.epochs+1):
            train_loss = 0
            train_acc = 0
            for _ in range(self.batches):
                temp_acc, temp_loss = self._cal_one_batch(train_iter, isTrain=True, opti=optimizer,train_idx=train_idx)
                train_acc += temp_acc
                train_loss += temp_loss
            test_acc, test_loss = self._cal_one_batch(test_iter)

            print("[Epoch:%d][train_loss:%.4e][train_acc:%.4f][val_loss:%.4e][val_acc:%.4f]"
                % (e, train_loss/self.batches, train_acc/self.batches, test_loss, test_acc))

            torch.save(self.network.encoder,datetime.datetime.now().strftime("%Y%m%d") + 'encoder.pkl')

            if (e>10 and e%5==0):
                for i in range(len(train_idx)):
                    if train_idx[i] >=0:
                        train_idx[i] = max(0,train_idx[i] - random.randint(0,round(train_iter[i].shape[0]*0.03)))

    def _preprocess(self, select):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.test_bearings})
        elif select == 'all':
            temp_data = self.dataset.get_value('data')
        else:
            raise ValueError('error selection for features!')
            
        for i,x in enumerate(temp_data):
            temp_one_data = x.transpose(0,2,1)
            # temp_one_data = self._fft(temp_one_data)
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

    def _cal_one_batch(self, data_iter, isTrain=False, opti=None,train_idx=None):
        
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
                temp_idx = np.random.randint(max(self.seq_len,train_idx[i]),data_iter[i].shape[0],size=x)
            else:
                temp_idx = np.random.randint(self.seq_len,data_iter[i].shape[0],size=x)

            for one_idx in temp_idx:
                if isTrain:
                    # temp_jiange = np.random.randint(1,one_idx//self.seq_len+1,size=self.seq_len)
                    temp_jiange = np.sort(np.int32(np.round(np.random.sample(self.seq_len)*one_idx)))
                    temp_one_data = data_iter[i][temp_jiange,:,:].copy()
                else:
                    temp_one_data = data_iter[i][one_idx-self.seq_len:one_idx,:,:].copy()
                #     temp_jiange=1
                # temp_one_data = data_iter[i][one_idx-self.seq_len*temp_jiange:one_idx:temp_jiange,:,:].copy()
                
                # daoshu = data_iter[i].shape[0]-one_idx
                # shunshu = (one_idx-self.seq_len*temp_jiange)/data_iter[i].shape[0]
                # if isTrain :
                #     if daoshu<20:
                #         random_num = np.random.rand(1)
                #         if random_num > 0.5:
                #             random_idx = np.random.randint(0,len(train_idx))
                #             temp_one_data[-1] = data_iter[random_idx][-daoshu].copy()
                #     if shunshu < 0.1:
                #         random_num = np.random.rand(1)
                #         if random_num > 0.5:
                #             random_idx = np.random.randint(0,len(train_idx))
                #             temp_one_data[0] = data_iter[random_idx][int(round(data_iter[random_idx].shape[0]*shunshu))].copy()
                        
                batch_data.append(temp_one_data.reshape([1,2,self.seq_len,data_len]))

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
            pe = (np.log(temp_one_rul+1)).reshape([-1,1])
            temp_one_data = Variable(torch.from_numpy(temp_one_data.copy()).type(torch.FloatTensor)).cuda() 
            temp_one_feature_list = []
            for j in range(temp_one_data.shape[0]//64+1):
                temp_one_fea = feature_net(temp_one_data[j*64:min(j*64+64,temp_one_data.shape[0]-1)].unsqueeze(2))
                temp_one_feature_list.append(temp_one_fea.data.cpu().numpy().reshape([temp_one_fea.shape[0],-1]))
            temp_one_fea = np.concatenate(temp_one_feature_list,axis=0)

            feature_dataset.append([bearing_name[i],temp_one_fea,temp_one_rul,pe])
        
        feature_dataset.save()
    def _test_gen_time(self,model_name):
        bearing_data = self._preprocess('all')
        bearing_name = self.dataset.get_value('bearing_name')
        bearing_rul = self.dataset.get_value('RUL')

        feature_net = torch.load(model_name + '.pkl')
        feature_net.eval()

        one_data = bearing_data[0]
        one_data = one_data[0].reshape([-1,one_data.shape[1],one_data.shape[2]])

        time_c = 0
        size = 100
        for i in range(size):
            test_one_data = one_data[0].reshape([-1,one_data.shape[1],one_data.shape[2]])
            time_s = time.time()
            test_one_data = Variable(torch.from_numpy(test_one_data.copy()).type(torch.FloatTensor)).cuda() 
            temp_one_fea = feature_net(test_one_data.unsqueeze(2))
            time_e = time.time()
            print(time_e - time_s)
            time_c += time_e - time_s
        print('time_mean',time_c/size)
        


if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = Process()
    # p.Begin()
    # p._gen_feature('20200602encoder')
    p._test_gen_time('20200602encoder')