import numpy as np 
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

class RUL_Encoder(nn.Module):
    def __init__(self, encoder_len):
        super(RUL_Encoder, self).__init__()
        self.encoder_len = encoder_len
        self.net = nn.Sequential(
            nn.Linear(1,128),
            nn.PReLU(),
            nn.Linear(128,256),
            nn.PReLU(),
            nn.Linear(256,self.encoder_len),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class RUL_Decoder(nn.Module):
    def __init__(self, encoder_len):
        super(RUL_Decoder, self).__init__()
        self.encoder_len = encoder_len
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.encoder_len, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Process():
    def __init__(self):
        self.epochs = 500
        self.batch_size = 128
        self.batches = 100
        self.lr = 1e-4
        self.optimizer = optim.Adam
        self.encoder_len = 16
        self.max_len = 10000
        self.encoder = RUL_Encoder(self.encoder_len)
        self.decoder = RUL_Decoder(self.encoder_len)
        self.net = nn.Sequential(self.encoder,self.decoder).cuda()

    def Begin(self):
        self.net.train()
        optimizer = self.optimizer(self.net.parameters(),lr=self.lr)
        min_loss = 1
        for e in range(1,self.epochs+1):
            epoch_loss = 0
            for _ in range(1,self.batches):
                batch_data = np.random.rand(self.batch_size).reshape([-1,1])

                batch_data = Variable(torch.from_numpy(batch_data).type(torch.FloatTensor)).cuda()

                optimizer.zero_grad()
                output = self.net(batch_data)
                loss = nn.functional.mse_loss(batch_data,output)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.data.cpu().numpy()

            epoch_loss /= self.batches
            if epoch_loss < min_loss:
                self.SaveNet()
                min_loss = epoch_loss
            print("[Epoch:%d][train_loss:%.4e]" % (e,epoch_loss))

    def SaveNet(self):
        torch.save(self.encoder, 'RUL_Encoder.pkl')
        torch.save(self.decoder, 'RUL_Decoder.pkl')

if __name__ == '__main__':
    torch.backends.cudnn.enabled=False
    p = Process()
    p.Begin()