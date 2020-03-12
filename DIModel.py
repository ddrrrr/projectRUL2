import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataset import DataSet

class DIModel(nn.Module):
    def __init__(self):
        super(DIModel,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2,16,65),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool1d(4),    # b*64*39
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64,64,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64,32,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            nn.Conv1d(32,16,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            nn.ConvTranspose1d(16,2,65),
            nn.Sigmoid(),
        )

    def forward(self, x_source, x_target):
        x_s = self.encoder(x_source)
        x_t = self.encoder(x_target)
        x_c = torch.cat([x_s, x_t], 2)
        out = self.decoder(x_c)
        return out

class Process():
    def __init__(self):
        self.dataset = DataSet.load_dataset(name = 'phm_data')
        self.lr = 0.001
        self.epochs = 50
        self.batches = 100
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
                train_idx = [np.random.randint(0,train_data[i].shape[0]-2,size=x) for i,x in enumerate(train_num)]
                train_source = np.vstack([train_data[i][train_idx[i]] for i in range(len(train_idx))])
                train_target = np.vstack([train_data[i][train_idx[i]+1] for i in range(len(train_idx))])
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
            

    def  _preprocess(self, select):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        else:
            raise ValueError('error selection for features!')

        for i,x in enumerate(temp_data):
            temp_one_data = x.transpose(0,2,1)
            temp_one_data = (temp_one_data - np.repeat(np.min(temp_one_data, axis=2, keepdims=True),2560,axis=2)) \
                / np.repeat((np.max(temp_one_data,axis=2,keepdims=True) - np.min(temp_one_data,axis=2,keepdims=True)),2560,axis=2)

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

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = Process()
    p.train()