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

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        m['relu1'] = nn.ReLU(inplace=True)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x):
        residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.CNN_1 = nn.Sequential(
            nn.Conv1d(2,64,11,1,5),
            nn.ReLU()
        )
        self.CNN_2 = nn.Sequential(
            nn.Conv1d(64,128,11,1,5),
            nn.ReLU()
        )
        self.FC = nn.Sequential(
            nn.Linear(20*128,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,64),
            nn.Sigmoid()
        )

    def forward(self, x):
        back_1 = self.CNN_1(x)
        x = nn.functional.max_pool1d(back_1,8)
        back_2 = self.CNN_2(x)
        x = nn.functional.max_pool1d(back_2,8)
        x = x.view(x.size(0),-1)
        fea = self.FC(x)
        return back_1,back_2,fea

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.CNN_1 = nn.Sequential(
            nn.Conv1d(128,2,11,1,5),
            nn.ReLU()
        )
        self.CNN_2 = nn.Sequential(
            nn.Conv1d(256,64,11,1,5),
            nn.ReLU()
        )
        self.FC = nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,20*128),
            nn.ReLU()
        )

    def forward(self, fea, back_1, back_2):
        fea = self.FC(fea)
        fea = fea.view(fea.size(0),128,-1)
        fea = nn.functional.upsample(fea,[20*8])
        fea = self.CNN_2(torch.cat([fea,back_2],1))
        fea = nn.functional.upsample(fea,[20*8*8])
        fea = self.CNN_1(torch.cat([fea,back_1],1))
        return fea

class DIModel(nn.Module):
    def __init__(self):
        super(DIModel,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x_source, x_target, x_s_label, x_t_label, label_all):
        back_1_s,back_2_s,fea_s = self.encoder(x_source)
        back_1_t,back_2_t,fea_t = self.encoder(x_target)

        out_s = self.decoder(fea_s, back_1_t, back_2_t)
        out_t = self.decoder(fea_t, back_1_s, back_2_s)

        kl_loss = torch.zeros(1).cuda()
        for i in range(label_all):
            for j in range(label_all):
                if i!=j:
                    temp_fea_1 = torch.cat([fea_s[x_s_label==i],fea_t[x_t_label==i]],dim=0)
                    temp_fea_2 = torch.cat([fea_s[x_s_label==j],fea_t[x_t_label==j]],dim=0)
                    kl_loss += nn.functional.kl_div(temp_fea_1.log(),temp_fea_2)
        kl_loss /= label_all*(label_all)/2
        return out_s, out_t, kl_loss

class TimeEncoder(nn.Module):
    def __init__(self):
        super(TimeEncoder,self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(2,8,16,16),
            nn.PReLU(),
            nn.Conv1d(8,32,8,8),
            nn.PReLU(), # 32*20
        )
        self.FC = nn.Sequential(
            nn.Linear(32*20, 128),
            nn.Tanh(),
            # nn.Dropout(0.3),
            # nn.Linear(256,64),
            # nn.Tanh()
        )

    def forward(self, x):
        nn.functional.dropout(x,p=0.3)
        x = self.CNN(x)
        x = x.view(x.size(0),-1)
        out = self.FC(x)
        return out

class TimeDecoder(nn.Module):
    def __init__(self):
        super(TimeDecoder,self).__init__()
        self.CNN = nn.Sequential(
            nn.ConvTranspose1d(32,32,8,8),
            nn.PReLU(),
            nn.ConvTranspose1d(32,2,16,16),
            nn.Tanh()
        )
        self.FC = nn.Sequential(
            nn.Linear(128*2,20*32),
            nn.PReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(512,20*32),
            # nn.PReLU()
        )
    def forward(self, x_source, x_target):
        x_c = torch.cat([x_source, x_target], 1)
        x_c = self.FC(x_c)
        x_c = x_c.view(x_c.size(0),32,-1)
        x_c = self.CNN(x_c)
        out = x_c
        # out = self.CNN(x_c)
        return out

class TimeDIModel(nn.Module):
    def __init__(self):
        super(TimeDIModel,self).__init__()
        self.encoder = TimeEncoder()
        self.decoder = TimeDecoder()

    def forward(self, x_source, x_target):
        x_s = self.encoder(x_source)

        x_t = self.encoder(x_target)

        out = self.decoder(x_s, x_t)
        return out

class STFTEncoder(nn.Module):
    def __init__(self, fea_size):
        super(STFTEncoder,self).__init__()
        self.fea_size = fea_size
        self.CNN = nn.Sequential(
            # nn.BatchNorm2d(2),
            nn.Conv2d(2,32,4,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(32,64,4,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(64,96,3,2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(96,128,5),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        self.FC = nn.Sequential(
            nn.Dropout(),
            nn.Linear(5*5*64,self.fea_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.CNN(x)
        # x = x.view(x.size(0),-1)
        # out = self.FC(x)
        return x

class STFTDecoder(nn.Module):
    def __init__(self, fea_size):
        super(STFTDecoder,self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(fea_size*2,512),
            nn.ReLU(),
            nn.Linear(512,4*4*64),
            nn.ReLU()
        )
        self.CNN = nn.Sequential(
            # nn.Conv2d(256,96,1),
            nn.ConvTranspose2d(256,196,3),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            nn.ConvTranspose2d(196,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(128,64,3,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(64,64,4,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(64,2,4,2),
            nn.ReLU(),
        )

    def forward(self, x_s_fea, x_t_fea):
        x = torch.cat([x_s_fea, x_t_fea],1)
        # x = self.FC(x)
        # x = x.view(x.size(0),64,4,4)
        out = self.CNN(x)
        return out

class STFTDIModel(nn.Module):
    def __init__(self, fea_size):
        super(STFTDIModel,self).__init__()
        self.fea_size = fea_size
        self.encoder = STFTEncoder(self.fea_size)
        self.decoder = STFTDecoder(self.fea_size)

    def forward(self, x_s, x_t, x_s_label, x_t_label, label_all):
        x_s_fea = self.encoder(x_s)
        x_t_fea = self.encoder(x_t)

        out = self.decoder(x_s_fea, x_t_fea)

        kl_loss = torch.zeros(1).cuda()
        for i in range(label_all):
            for j in range(label_all):
                if i!=j:
                    temp_fea_1 = torch.cat([x_s_fea[x_s_label==i],x_t_fea[x_t_label==i]],dim=0)
                    temp_fea_2 = torch.cat([x_s_fea[x_s_label==j],x_t_fea[x_t_label==j]],dim=0)
                    kl_loss += nn.functional.kl_div(temp_fea_1.log(),temp_fea_2)
        kl_loss /= label_all*(label_all-1)
        return out, kl_loss

class STFTUnet(nn.Module):
    def __init__(self):
        super(STFTUnet, self).__init__()
        self.encoder_CNN_1 = nn.Sequential(
            nn.Conv2d(2,32,4,2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.encoder_CNN_2 = nn.Sequential(
            nn.Conv2d(32,64,4,2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.encoder_CNN_3 = nn.Sequential(
            nn.Conv2d(64,96,3,2),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.encoder_CNN_4 = nn.Sequential(
            nn.Conv2d(96,128,5),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        self.decoder_CNN_4 = nn.Sequential(
            nn.ConvTranspose2d(128,96,5),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.decoder_CNN_3 = nn.Sequential(
            nn.ConvTranspose2d(96*2,64,3,2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder_CNN_2 = nn.Sequential(
            nn.ConvTranspose2d(64*2,32,4,2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder_CNN_1 = nn.Sequential(
            nn.ConvTranspose2d(32*2,2,4,2),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )

    def forward(self, x_source, x_t_fea):
        x_fea_1 = self.encoder_CNN_1(x_source)
        x_fea_2 = self.encoder_CNN_2(x_fea_1)
        x_fea_3 = self.encoder_CNN_3(x_fea_2)
        x_fea_4 = self.encoder_CNN_4(x_fea_3)

        # out = torch.cat([x_t_fea,x_fea_4],dim=1)
        out = x_fea_4 * x_t_fea
        out = self.decoder_CNN_4(out)
        out = torch.cat([out, x_fea_3],dim=1)
        out = self.decoder_CNN_3(out)
        out = torch.cat([out, x_fea_2],dim=1)
        out = self.decoder_CNN_2(out)
        out = torch.cat([out, x_fea_1],dim=1)
        out = self.decoder_CNN_1(out)

        return out

class STFTEncoder2(nn.Module):
    def __init__(self):
        super(STFTEncoder2,self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(2,32,4,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,96,3,2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96,128,5),
            nn.BatchNorm2d(128),
            nn.Softmax()
        )
    def forward(self, x):
        return self.CNN(x)

class STFTDIModel2(nn.Module):
    def __init__(self):
        super(STFTDIModel2,self).__init__()
        self.encoder = STFTEncoder2()
        self.Unet = STFTUnet()

    def forward(self, x_source, x_target):
        x_t_fea = self.encoder(x_target)
        x_trans = self.Unet(x_source, x_t_fea)
        return x_trans


class Process():
    def __init__(self):
        self.dataset = DataSet.load_dataset(name = 'phm_data')
        self.lr = 2e-3
        self.epochs = 1500
        self.batches = 50
        self.batch_size = 32
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']

    def train(self):
        train_data = self._preprocess('train')
        test_data = self._preprocess('test')
        # net = DIModel().cuda()
        # net = STFTDIModel(128).cuda()
        # net = TimeDIModel().cuda()
        net = STFTDIModel2().cuda()
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        for e in range(1, self.epochs+1):
            train_loss = [0,0]
            val_loss = [0,0]
            for _ in range(1, self.batches+1):
                label_all = len(train_data)
                train_num = [self.batch_size//len(train_data)]*len(train_data)
                batch_s_label = -np.ones(self.batch_size)
                batch_t_label = -np.ones(self.batch_size)
                for x in random.sample(range(0,len(train_data)),self.batch_size % len(train_data)):
                    train_num[x] += 1
                train_idx = []
                train_idx_p = []
                label_count = 0
                for i,x in enumerate(train_num):
                    temp_idx = np.random.randint(0,train_data[i].shape[0]-1,size=x)
                    train_idx.append(temp_idx)
                    train_idx_p.append(temp_idx+1)

                    # append_idx = np.random.randint(0,train_data[i].shape[0]-1,size=x)
                    # train_idx.append(append_idx)
                    # append_idx_p = np.random.randint(0,train_data[i].shape[0],size=x)
                    # train_idx_p.append(append_idx_p)

                    batch_s_label[label_count:label_count+self.batch_size//len(train_data)] = i
                    batch_t_label[label_count:label_count+self.batch_size//len(train_data)] = i
                    label_count += x

                train_source = np.vstack([train_data[i][train_idx[i]] for i in range(len(train_idx))])
                train_target = np.vstack([train_data[i][train_idx_p[i]] for i in range(len(train_idx_p))])
                temp_train_loss = self._fit(net,optimizer,[train_source,train_target], [batch_s_label,batch_t_label], label_all)
                train_loss[0] += temp_train_loss[0]
                train_loss[1] += temp_train_loss[1]

                # evaluate
                label_all = len(test_data)
                test_num = [self.batch_size//len(test_data)]*len(test_data)
                batch_s_label = -np.ones(self.batch_size)
                batch_t_label = -np.ones(self.batch_size)
                for x in random.sample(range(0,len(test_data)),self.batch_size % len(test_data)):
                    test_num[x] += 1
                test_idx = []
                test_idx_p = []
                label_count = 0
                for i,x in enumerate(test_num):
                    temp_idx = np.random.randint(0,test_data[i].shape[0]-1,size=x)
                    test_idx.append(temp_idx)
                    test_idx_p.append(temp_idx+1)

                    # append_idx = np.random.randint(0,test_data[i].shape[0],size=x)
                    # test_idx.append(append_idx)
                    # append_idx_p = np.random.randint(0,test_data[i].shape[0],size=x)
                    # test_idx_p.append(append_idx_p)

                    batch_s_label[label_count:label_count+self.batch_size//len(test_data)] = i
                    batch_t_label[label_count:label_count+self.batch_size//len(test_data)] = i
                    label_count += x



                # test_idx = [np.random.randint(0,test_data[i].shape[0],size=x) for i,x in enumerate(test_num)]
                # test_idx_p = [np.random.randint(0,test_data[i].shape[0],size=x) for i,x in enumerate(test_num)]
                test_source = np.vstack([test_data[i][test_idx[i]] for i in range(len(test_idx))])
                test_target = np.vstack([test_data[i][test_idx_p[i]] for i in range(len(test_idx_p))])
                temp_val_loss = self._evaluate(net,[test_source,test_target], [batch_s_label,batch_t_label], label_all)
                val_loss[0] += temp_val_loss[0]
                val_loss[1] += temp_val_loss[1]

            print("[Epoch:%d][train_loss:%.4e][train_KL_loss:%.4e][val_loss:%.4e][val_KL_loss:%.4e]"
                % (e,train_loss[0]/self.batches, train_loss[1]/self.batches, 
                val_loss[0]/self.batches, val_loss[1]/self.batches))

        self.save_model(net.encoder,datetime.datetime.now().strftime("%Y%m%d") + "encoder")

        # test
        label_all = len(train_data)
        train_num = [self.batch_size//len(train_data)]*len(train_data)
        batch_s_label = -np.ones(self.batch_size)
        batch_t_label = -np.ones(self.batch_size)
        for x in random.sample(range(0,len(train_data)),self.batch_size % len(train_data)):
            train_num[x] += 1

        train_idx = []
        train_idx_p = []
        label_count = 0
        for i,x in enumerate(train_num):
            temp_idx = np.random.randint(0,train_data[i].shape[0]-1,size=x)
            train_idx.append(temp_idx)
            train_idx_p.append(temp_idx+1)
            # append_idx = np.random.randint(0,train_data[i].shape[0]-2,size=x)
            # train_idx.append(append_idx)
            # append_idx_p = np.random.randint(0,train_data[i].shape[0],size=x)
            # train_idx_p.append(append_idx_p)
            batch_s_label[label_count:label_count+self.batch_size//len(train_data)] = i
            batch_t_label[label_count:label_count+self.batch_size//len(train_data)] = i
            label_count += x

        
        train_source = np.vstack([train_data[i][train_idx[i]] for i in range(len(train_idx))])
        train_target = np.vstack([train_data[i][train_idx_p[i]] for i in range(len(train_idx))])
        self._test(net,[train_source,train_target],[batch_s_label, batch_t_label], label_all)

        # self.save_model(net.encoder,datetime.datetime.now().strftime("%Y%m%d") + "encoder")
            

    def  _preprocess(self, select):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        elif select == 'all':
            temp_data = self.dataset.get_value('data')
        else:
            raise ValueError('error selection for features!')


        for i,x in enumerate(temp_data):
            temp_one_data = x.transpose(0,2,1)
            # temp_one_data = self._fft(temp_one_data)
            temp_one_data = self._stft(temp_one_data)
            # temp_one_data *= 50
            # length_data = temp_one_data.shape[2]
            # temp_one_data = (temp_one_data - np.repeat(np.min(temp_one_data, axis=2, keepdims=True),length_data,axis=2)) \
            #     / np.repeat((np.max(temp_one_data,axis=2,keepdims=True) - np.min(temp_one_data,axis=2,keepdims=True)),length_data,axis=2)
            
            # temp_one_data = temp_one_data*2-1

            temp_data[i] = temp_one_data

        return temp_data

    def _fit(self, model, optimizer, train_iter, label_iter, label_all):
        model.train()
        for i in range(len(train_iter)):
            train_iter[i] = Variable(torch.from_numpy(train_iter[i].copy()).type(torch.FloatTensor)).cuda()
            label_iter[i] = Variable(torch.from_numpy(label_iter[i].copy()).type(torch.FloatTensor)).cuda()

        # output = model(train_iter[0],train_iter[1],label_iter[0],label_iter[1],label_all)
        output = model(train_iter[0],train_iter[1])
        # loss = torch.nn.functional.mse_loss(output[0],train_iter[0]) + 2e-3* output[1]
        loss = torch.nn.functional.mse_loss(output,train_iter[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()        #empty useless variable

        # result = output[0].data.cpu().numpy()
        # if np.std(result < 1e-8):
        #     print('result is 0')
        # return loss.data.cpu().numpy(), output[2].data.cpu().numpy()
        return loss.data.cpu().numpy(),0

    def _evaluate(self, model, test_iter, label_iter, label_all):
        model.eval()
        for i in range(len(test_iter)):
            test_iter[i] = Variable(torch.from_numpy(test_iter[i].copy()).type(torch.FloatTensor)).cuda()
            label_iter[i] = Variable(torch.from_numpy(label_iter[i].copy()).type(torch.FloatTensor)).cuda()

        # output = model(test_iter[0],test_iter[1],label_iter[0],label_iter[1],label_all)
        # loss = torch.nn.functional.mse_loss(output[0],test_iter[0]) +  2e-3* output[1]
        output = model(test_iter[0],test_iter[1])
        loss = torch.nn.functional.mse_loss(output,test_iter[1])
        torch.cuda.empty_cache()        #empty useless variable
        # return loss.data.cpu().numpy(), output[2].data.cpu().numpy()
        return loss.data.cpu().numpy(),0

    def _test(self, model, test_iter, label_iter, label_all):
        model.eval()
        # target = np.concatenate([test_iter[0],test_iter[1]],axis=0)
        for i in range(len(test_iter)):
            test_iter[i] = Variable(torch.from_numpy(test_iter[i].copy()).type(torch.FloatTensor)).cuda() 
            label_iter[i] = Variable(torch.from_numpy(label_iter[i].copy()).type(torch.FloatTensor)).cuda()

        # output,_ = model(test_iter[0],test_iter[1],label_iter[0],label_iter[1],label_all)
        output = model(test_iter[0],test_iter[1])
        
        # output = np.concatenate([output[0].data.cpu().numpy(), output[1].data.cpu().numpy()],axis=0)
        np.save("result.npy",output.data.cpu().numpy())
        np.save("target.npy",test_iter[1].data.cpu().numpy())

    def _fft(self, data):
        fft_data = np.fft.fft(data,axis=2)/data.shape[2]
        fft_data = (np.abs(fft_data))**2
        fft_data = fft_data[:,:,1:1281]
        return fft_data

    def _stft(self, data):
        stft_data = np.zeros([data.shape[0],2,50,50])
        for i in range(data.shape[0]):
            for j in range(2):
                _,_,z = scipy.signal.stft(data[i,j,:],fs=0.1,nperseg=256,noverlap=204)
                z = np.abs(z[:50,:50])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                stft_data[i,j,:,:] = z
        return stft_data

    def save_model(self, model, name):
        # 保存
        torch.save(model, name + '.pkl')
        # # 加载
        # model = torch.load('\model.pkl')

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
            temp_one_data = self._stft(temp_one_data)
            # temp_one_data = self._fft(temp_one_data)
            # length_data = temp_one_data.shape[2]
            # temp_one_data = (temp_one_data - np.repeat(np.min(temp_one_data, axis=2, keepdims=True),length_data,axis=2)) \
            #     / np.repeat((np.max(temp_one_data,axis=2,keepdims=True) - np.min(temp_one_data,axis=2,keepdims=True)),length_data,axis=2)
            # temp_one_data = temp_one_data*2-1
            temp_one_data = Variable(torch.from_numpy(temp_one_data.copy()).type(torch.FloatTensor)).cuda() 

            temp_one_feature_list = []
            for j in range(temp_one_data.shape[0]//64+1):
                temp_one_fea = feature_net(temp_one_data[j*64:min(j*64+64,temp_one_data.shape[0]-1)])
                temp_one_feature_list.append(temp_one_fea.data.cpu().numpy().reshape([temp_one_fea.shape[0],-1]))
            
            temp_one_feature = np.vstack(temp_one_feature_list)
            # temp_one_feature = temp_one_data.reshape([temp_one_data.shape[0],-1])
            # temp_one_feature = self._position_encoding(temp_one_feature)
            feature_dataset.append([bearing_name[i],temp_one_feature,state,posibility,rul,rul_encoding])

        feature_dataset.save()

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
        # pe = (1 - np.exp(-temp_rul / 500)).reshape([-1,1])
        pe = (np.log10(temp_rul+1)).reshape([-1,1])
        # pe = (-np.log10(temp_rul+1)+3).reshape([-1,1])
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

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    p = Process()
    p.train()
    p._GenFeature('20200513encoder')