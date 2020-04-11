import datetime
import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(2,8,65),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(8,12,33),
            nn.ReLU(),
            nn.MaxPool1d(4),
            # nn.Conv1d(96,128,17),
            # nn.ReLU(),
            # nn.MaxPool1d(4),
            nn.Conv1d(12,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool1d(2),    # b*32*17
        )
        self.FC = nn.Sequential(
            nn.Linear(17*16,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )

    def forward(self, x):
        x = self.CNN(x)
        x = x.view(x.size(0),-1)
        out = self.FC(x)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,34*16),
        )
        self.CNN = nn.Sequential(
            nn.Conv1d(16,128,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(128,96,5,1,2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(96,64,5,1,2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64,64,33),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            # nn.ConvTranspose1d(64,64,5,1,2),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64,2,65),
            nn.ReLU(),
        )

    def forward(self, x_source, x_target):
        x_c = torch.cat([x_source, x_target], 1)
        x_c = self.FC(x_c)
        x_c = x_c.view(x_c.size(0),16,34)
        out = self.CNN(x_c)
        return out


class DIModel(nn.Module):
    def __init__(self):
        super(DIModel,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x_source, x_target):
        x_s = self.encoder(x_source)

        x_t = self.encoder(x_target)

        out = self.decoder(x_s, x_t)
        return out

class Process():
    def __init__(self):
        self.dataset = DataSet.load_dataset(name = 'phm_data')
        self.lr = 0.00005
        self.epochs = 50
        self.batches = 50
        self.batch_size = 64
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']

    def train(self):
        train_data = self._preprocess('train')
        test_data = self._preprocess('test')
        net = DIModel().cuda()
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        for e in range(1, self.epochs+1):
            train_loss = 0
            val_loss = 0
            for _ in range(1, self.batches+1):
                train_num = [self.batch_size//len(train_data)]*len(train_data)
                for x in random.sample(range(0,len(train_data)),self.batch_size % len(train_data)):
                    train_num[x] += 1
                train_idx = []
                train_idx_p = []
                for i,x in enumerate(train_num):
                    append_idx = np.random.randint(0,train_data[i].shape[0]-2,size=x)
                    train_idx.append(append_idx)
                    max_zhishu = int(min(np.floor(np.log2(train_data[i].shape[0] - np.max(append_idx) - 1)) , 6))
                    append_zhishu = np.random.randint(0,max_zhishu,size=x)
                    append_idx_p = append_idx + np.power(2,append_zhishu)
                    train_idx_p.append(append_idx_p)

                train_source = np.vstack([train_data[i][train_idx[i]] for i in range(len(train_idx))])
                train_target = np.vstack([train_data[i][train_idx_p[i]] for i in range(len(train_idx_p))])
                train_loss += self._fit(net,optimizer,[train_source,train_target])

                test_num = [self.batch_size//len(test_data)]*len(test_data)
                for x in random.sample(range(0,len(test_data)),self.batch_size % len(test_data)):
                    test_num[x] += 1
                test_idx = [np.random.randint(0,test_data[i].shape[0]-2,size=x) for i,x in enumerate(test_num)]
                test_source = np.vstack([test_data[i][test_idx[i]] for i in range(len(test_idx))])
                test_target = np.vstack([test_data[i][test_idx[i]+1] for i in range(len(test_idx))])
                val_loss += self._evaluate(net,[test_source,test_target])
            print("[Epoch:%d][train_loss:%.4e][val_loss:%.4e]"
                % (e,train_loss/self.batches,val_loss/self.batches))


        # test
        train_num = [self.batch_size//len(train_data)]*len(train_data)
        for x in random.sample(range(0,len(train_data)),self.batch_size % len(train_data)):
            train_num[x] += 1

        train_idx = []
        train_idx_p = []
        for i,x in enumerate(train_num):
            append_idx = np.random.randint(0,train_data[i].shape[0]-2,size=x)
            train_idx.append(append_idx)
            max_zhishu = int(min(np.floor(np.log2(train_data[i].shape[0] - np.max(append_idx) - 1)) , 6))
            append_zhishu = np.random.randint(0,max_zhishu,size=x)
            append_idx_p = append_idx + np.power(2,append_zhishu)
            train_idx_p.append(append_idx_p)
        
        train_source = np.vstack([train_data[i][train_idx[i]] for i in range(len(train_idx))])
        train_target = np.vstack([train_data[i][train_idx_p[i]] for i in range(len(train_idx))])
        self._test(net,[train_source,train_target])

        self.save_model(net.encoder,datetime.datetime.now().strftime("%Y%m%d") + "encoder")
            

    def  _preprocess(self, select):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        else:
            raise ValueError('error selection for features!')


        for i,x in enumerate(temp_data):
            temp_one_data = x.transpose(0,2,1)
            temp_one_data = self._fft(temp_one_data)
            length_data = temp_one_data.shape[2]
            temp_one_data = (temp_one_data - np.repeat(np.min(temp_one_data, axis=2, keepdims=True),length_data,axis=2)) \
                / np.repeat((np.max(temp_one_data,axis=2,keepdims=True) - np.min(temp_one_data,axis=2,keepdims=True)),length_data,axis=2)

            temp_data[i] = temp_one_data

        return temp_data

    def _fit(self, model, optimizer, train_iter):
        model.train()
        for i in range(len(train_iter)):
            train_iter[i] = Variable(torch.from_numpy(train_iter[i].copy()).type(torch.FloatTensor)).cuda()

        output = model(train_iter[0],train_iter[1])
        loss = torch.nn.functional.mse_loss(output,train_iter[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()        #empty useless variable
        return loss.data.cpu().numpy()

    def _evaluate(self, model, test_iter):
        model.eval()
        for i in range(len(test_iter)):
            test_iter[i] = Variable(torch.from_numpy(test_iter[i].copy()).type(torch.FloatTensor)).cuda()

        output = model(test_iter[0],test_iter[1])
        loss = torch.nn.functional.mse_loss(output,test_iter[1])
        torch.cuda.empty_cache()        #empty useless variable
        return loss.data.cpu().numpy()

    def _test(self, model, test_iter):
        model.eval()
        target = test_iter[1]
        for i in range(len(test_iter)):
            test_iter[i] = Variable(torch.from_numpy(test_iter[i].copy()).type(torch.FloatTensor)).cuda() 
        output = model(test_iter[0],test_iter[1])
        output = output.data.cpu().numpy()
        np.save("result.npy",output)
        np.save("target.npy",target)

    def _fft(self, data):
        fft_data = np.fft.fft(data,axis=2)/data.shape[2]
        fft_data = (np.abs(fft_data))**2
        fft_data = fft_data[:,:,1:1281]
        return fft_data

    def save_model(self, model, name):
        # 保存
        torch.save(model, name + '.pkl')
        # # 加载
        # model = torch.load('\model.pkl')

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = Process()
    p.train()