import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime

from . import transforms, io, dynamics, utils

sz = 3

def convbatchrelu(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )  

def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )  

def batchconv0(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )  

class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj  = batchconv0(in_channels, out_channels, 1)
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz))
                
    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz))
                
    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x

class downsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase)-1):
            if residual_on:
                self.down.add_module('res_down_%d'%n, resdown(nbase[n], nbase[n+1], sz))
            else:
                self.down.add_module('conv_down_%d'%n, convdown(nbase[n], nbase[n+1], sz))
            
    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n>0:
                y = self.maxpool(xd[n-1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd
    
class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        self.conv = batchconv(in_channels, out_channels, sz)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels*2)
        else:
            self.full = nn.Linear(style_channels, out_channels)
        
    def forward(self, style, x, y=None):
        if y is not None:
            x = x + y
        feat = self.full(style)
        y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y
    
class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.proj  = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y)
        x = x + self.conv[3](style, self.conv[2](style, x))
        return x
    
class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation))
        
    def forward(self, x, y, style):
        x = self.conv[1](style, self.conv[0](x), y)
        return x
    
class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        #self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        #style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2],x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5

        return style
    
class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, concatenation=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1,len(nbase)):
            if residual_on:
                self.up.add_module('res_up_%d'%(n-1), 
                    resup(nbase[n], nbase[n-1], nbase[-1], sz, concatenation))
            else:
                self.up.add_module('conv_up_%d'%(n-1), 
                    convup(nbase[n], nbase[n-1], nbase[-1], sz, concatenation))

    def forward(self, style, xd):
        x = self.up[-1](xd[-1], xd[-1], style)
        for n in range(len(self.up)-2,-1,-1):
            x= self.upsampling(x)
            x = self.up[n](x, xd[n], style)
        return x
    
class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz, residual_on=True, style_on=True, concatenation=False):
        super(CPnet, self).__init__()
        self.nbase = nbase
        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        self.style_on = style_on
        
    def forward(self, data):
        T0    = self.downsample(data)
        style = self.make_style(T0[-1])
        style0 = style 
        if not self.style_on:
            style = style * 0
        T0 = self.upsample(style, T0)
        T0    = self.output(T0)
        return T0, style0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))


class CellposeModel(): 
    def __init__(self, device=None): 
        nchan = 2
        nbase = [nchan, 32, 64, 128, 256]
        net = CPnet(nbase, 3, sz)
        if device is None:
            self.device = torch.device('cuda')
        else:
            self.device = device
        self.diam_mean = 30
        self.net = CPnet(nbase, 3, 3).to(self.device)

    def loss_fn(self, lbl, y):
        criterion  = nn.MSELoss(reduction='mean')
        criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
        veci = 5. * torch.from_numpy(lbl[:,1:]).float().to(self.device)
        lbl  = torch.from_numpy(lbl[:,0]>0.5).float().to(self.device)
        loss = criterion(y[:,:-1], veci) / 2 + criterion2(y[:,-1] , lbl)
        return loss
    
    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, pretrained_model=None, 
              save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, weight_decay=0.00001, batch_size=8, rescale=True):
        
        nimg = len(train_data)

        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels,
                                                                                                   test_data, test_labels,
                                                                                                   channels, normalize)

        # check if train_labels have flows
        train_flows = dynamics.labels_to_flows(train_labels, files=train_files)
        if run_test:
            test_flows = dynamics.labels_to_flows(test_labels, files=test_files)
        else:
            test_flows = None
        
        netstr='cellpose'
        
        d = datetime.datetime.now()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = 0.00001
        self.momentum = 0.95

        nimg = len(train_data)

        # compute average cell diameter
        if rescale:
            diam_train = np.array([utils.diameters(train_labels[k][0])[0] for k in range(len(train_labels))])
            diam_train[diam_train<5] = 5.
            if test_data is not None:
                diam_test = np.array([utils.diameters(test_labels[k][0])[0] for k in range(len(test_labels))])
                diam_test[diam_test<5] = 5.
            scale_range = 0.5
        else:
            scale_range = 1.0

        net_type='cellpose'

        nchan = train_data[0].shape[0]
        print('>>>> training network with %d channel input <<<<'%nchan)
        print('>>>> saving every %d epochs'%save_every)
        print('>>>> median diameter = %d'%self.diam_mean)
        print('>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f'%(self.learning_rate, self.batch_size, self.weight_decay))
        print('>>>> ntrain = %d'%nimg)
        if test_data is not None:
            print('>>>> ntest = %d'%len(test_data))
        print(train_data[0].shape)

        optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate,
                            momentum=self.momentum, weight_decay=self.weight_decay)

        eta = np.linspace(0, self.learning_rate, 10)
        tic = time.time()

        lavg, nsum = 0., 0

        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path, 'models/')

            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            print('WARNING: no save_path given, model not saving')

        ksave = 0
        rsc = 1.0

        self.net.train()

        for iepoch in range(self.n_epochs):
            np.random.seed(iepoch)
            rperm = np.random.permutation(nimg)
            if iepoch<len(eta):
                LR = eta[iepoch]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR
            for ibatch in range(0,nimg,batch_size):
                if rescale:
                    diam_batch = diam_train[rperm[ibatch:ibatch+batch_size]]
                    rsc = diam_batch / self.diam_mean
                else:
                    rsc = np.ones(len(rperm[ibatch:ibatch+batch_size]), np.float32)

                imgi, lbl, scale = transforms.random_rotate_and_resize(
                                        [train_data[i] for i in rperm[ibatch:ibatch+batch_size]],
                                        Y=[train_labels[i][1:] for i in rperm[ibatch:ibatch+batch_size]],
                                        rescale=rsc, scale_range=scale_range)
                X = torch.from_numpy(imgi).float().to(self.device)
                
                optimizer.zero_grad()
                self.net.train()
                y, style = self.net(X)
                loss = self.loss_fn(lbl,y)
                loss.backward()
                train_loss = loss.item()
                if iepoch>0:
                    optimizer.step()
                lavg += train_loss * len(imgi)
                nsum += len(imgi)
            if iepoch>self.n_epochs-100 and iepoch%10==1:
                LR = LR/2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR

            if iepoch%10==0 or iepoch<10:
                lavg = lavg / nsum
                if test_data is not None:
                    lavgt = 0.
                    nsum = 0
                    np.random.seed(42)
                    rperm = np.arange(0, len(test_data), 1, int)
                    for ibatch in range(0,len(test_data),batch_size):
                        if rescale:
                            rsc = diam_test[rperm[ibatch:ibatch+batch_size]] / self.diam_mean
                        else:
                            rsc = np.ones(len(rperm[ibatch:ibatch+batch_size]), np.float32)
                        imgi, lbl, scale = transforms.random_rotate_and_resize(
                                            [test_data[i] for i in rperm[ibatch:ibatch+batch_size]],
                                            Y=[test_labels[i][1:] for i in rperm[ibatch:ibatch+batch_size]],
                                            scale_range=0., rescale=rsc)
                        X    = torch.from_numpy(imgi).float().to(self.device)
                        self.net.eval()
                        y, style = self.net(X)
                        loss = self.loss_fn(lbl, y)
                        test_loss = loss.item()
                        lavgt += test_loss * len(imgi)
                        nsum += len(imgi)
                    print('Epoch %d, Time %4.1fs, Loss %2.4f, Loss Test %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavg, lavgt/nsum, LR))
                else:
                    print('Epoch %d, Time %4.1fs, Loss %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavg, LR))
                lavg, nsum = 0., 0

            if save_path is not None:
                if iepoch==self.n_epochs-1 or iepoch%save_every==1:
                    # save model at the end
                    file = '{}_{}_{}'.format(net_type, file_label, d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                    ksave += 1
                    print('saving network parameters')
                    torch.save(self.net.state_dict(), os.path.join(file_path, file))
        return os.path.join(file_path, file)
