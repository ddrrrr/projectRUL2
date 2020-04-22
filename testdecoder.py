import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

class PositionDecoder(nn.Module):
    def __init__(self, in_len):
        super(PositionDecoder,self).__init__()
        self.in_len = in_len
        self.net = nn.Sequential(
            nn.Linear(self.in_len,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.net(x)
        # out = torch.asin(out) + math.pi/2
        # out = out / math.pi * self.max_len
        return out

class Process():
    def __init__(self):
        self.epochs = 500
        self.batch_size = 128
        self.batches = 100
        self.lr = 1e-3
        self.optimizer = optim.Adam
        self.encoder_len = 16
        self.max_len = 10000
        self.net = PositionDecoder(self.encoder_len).cuda()

    def _GenBatchData(self):
        pe = np.zeros([self.batch_size, self.encoder_len])
        position = np.random.randint(0,self.max_len,self.batch_size).reshape([-1,1])
        div_term = np.exp(np.arange(0, self.encoder_len, 2) *
                             -(np.log(self.max_len) / self.encoder_len))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe, position

    def _GetRealLoss(self, position, output):
        output = np.arcsin(output) + np.pi/2
        output = output/np.pi * self.max_len
        loss = np.mean((position - output)**2)
        return loss

    def Begin(self):
        self.net.train()
        optimizer = self.optimizer(self.net.parameters(),lr=self.lr)
        min_loss = 1e7
        for e in range(1,self.epochs+1):
            epoch_loss = 0
            real_loss = 0
            for _ in range(1,self.batches):
                batch_data, position = self._GenBatchData()

                batch_label = np.sin(position/self.max_len*np.pi - np.pi/2)
                batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()
                batch_label = Variable(torch.from_numpy(batch_label).type(torch.FloatTensor)).cuda()

                optimizer.zero_grad()
                output = self.net(batch_data)
                loss = nn.functional.mse_loss(batch_label,output)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.data.cpu().numpy()
                real_loss += self._GetRealLoss(position, output.data.cpu().numpy())

            epoch_loss /= self.batches
            real_loss /= self.batches
            if real_loss < min_loss:
                self.SaveNet()
                min_loss = real_loss
            print("[Epoch:%d][train_loss:%.4e][real_loss:%.4e]" % (e,epoch_loss,real_loss))


    def SaveNet(self):
        torch.save(self.net, 'PositionDecoder.pkl')


if __name__ == '__main__':
    torch.backends.cudnn.enabled=False
    p = Process()
    p.Begin()


    