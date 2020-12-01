import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from . import transforms, io

sz = 3

def convbatchrelu(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )  

def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )  

def batchconv0(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )  

class resdown(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj  = batchconv0(in_channels, out_channels, sz)
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz))
                
    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class downsample(nn.Module):
    def __init__(self, nbase, sz):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase)-1):
            self.down.add_module('res_down_%d'%n, resdown(nbase[n], nbase[n+1], sz))
            
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
    def __init__(self, in_channels, out_channels, style_channels, sz):
        super().__init__()
        self.conv = batchconv(in_channels, out_channels, sz)
        self.full = nn.Linear(style_channels, out_channels)
        
    def forward(self, style, x, y=None):
        if y is not None:
            print(x.shape, y.shape)
            x = x + y
        feat = self.full(style)
        y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y
    
class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.proj  = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y)
        x = x + self.conv[3](style, self.conv[2](style, x))
        return x
    
    
class make_style(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        style = self.pool_all(x0)
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1).unsqueeze(1)**.5

        return style
    
class upsample(nn.Module):
    def __init__(self, nbase, sz):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1,len(nbase)):
            self.up.add_module('res_up_%d'%(n-1), resup(nbase[n], nbase[n-1], nbase[-1], sz))
                
    def forward(self, style, xd):
        x = self.up[-1](xd[-1], xd[-1], style)
        for n in range(len(self.up)-2,-1,-1):
            x= self.upsampling(x)
            x = self.up[n](x, xd[n], style)
        return x
    
class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz):
        super(CPnet, self).__init__()
        self.nbase = nbase
        self.downsample = downsample(nbase, sz)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz)
        self.make_style = make_style()
        self.output = batchconv(nbaseup[0], nout, 1)
        
    def forward(self, data):
        T0    = self.downsample(data)
        style = self.make_style(T0[-1])
        T0 = self.upsample(style, T0)
        T0    = self.output(T0)
        return T0, style


class CellposeModel(): 
    def __init__(self): 
        nchan = 2
        nbase = [nchan, 32, 64, 128, 256]
        net = CPnet(nbase, 3, sz)
        self.device = torch.device('cuda')

    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        criterion  = nn.MSELoss()
        criterion2 = nn.BCELoss()
        veci = 5. * torch.from_numpy(lbl[:,1:]).float().to(self.device)
        lbl  = torch.from_numpy(lbl[:,0]>.5).float().to(self.device)
        loss = criterion(y[:,:-1] , veci) + criterion2(y[:,-1] , lbl)
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
        self.weight_decay = weight_decay
        self.momentum = 0.9

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

        nchan = train_data[0].shape[0]
        print('>>>> training network with %d channel input <<<<'%nchan)
        print('>>>> saving every %d epochs'%save_every)
        print('>>>> median diameter = %d'%self.diam_mean)
        print('>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f'%(self.learning_rate, self.batch_size, self.weight_decay))
        print('>>>> ntrain = %d'%nimg)
        if test_data is not None:
            print('>>>> ntest = %d'%len(test_data))
        print(train_data[0].shape)

        trainer = optim.SGD(self.net.parameters(), lr=self.learning_rate,
                            momentum=self.momentum, wd=self.weight_decay)

        eta = np.linspace(0, self.learning_rate, 10)
        tic = time.time()

        lavg, nsum = 0, 0

        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path, 'models/')

            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            print('WARNING: no save_path given, model not saving')

        ksave = 0
        rsc = 1.0

        for iepoch in range(self.n_epochs):
            np.random.seed(iepoch)
            rperm = np.random.permutation(nimg)
            if iepoch<len(eta):
                LR = eta[iepoch]
                trainer.set_learning_rate(LR)
            for ibatch in range(0,nimg,batch_size):
                if rescale:
                    diam_batch = diam_train[rperm[ibatch:ibatch+batch_size]]
                    rsc = diam_batch / self.diam_mean
                else:
                    rsc = np.ones(len(rperm[ibatch:ibatch+batch_size]), np.float32)

                imgi, lbl, scale = transforms.random_rotate_and_resize(
                                        [train_data[i] for i in rperm[ibatch:ibatch+batch_size]],
                                        Y=[train_labels[i][1:] for i in rperm[ibatch:ibatch+batch_size]],
                                        rescale=rsc, scale_range=scale_range, unet=self.unet)
                if self.unet and lbl.shape[1]>1 and rescale:
                    #lbl[:,1] *= scale[0]**2
                    lbl[:,1] /= diam_batch[:,np.newaxis,np.newaxis]**2
                X = torch.from_numpy(imgi).float().to(device)
                with mx.autograd.record():
                    y, style = self.net(X)
                    loss = self.loss_fn(lbl, y)

                loss.backward()
                train_loss = loss.sum().item
                lavg += train_loss
                nsum+=len(loss)
                if iepoch>0:
                    trainer.step(batch_size)
            if iepoch>self.n_epochs-100 and iepoch%10==1:
                LR = LR/2
                trainer.set_learning_rate(LR)

            if iepoch%10==0 or iepoch<10:
                lavg = lavg / nsum
                if test_data is not None:
                    lavgt = 0
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
                                            scale_range=0., rescale=rsc, unet=self.unet)
                        if self.unet and lbl.shape[1]>1:
                            lbl[:,1] *= scale[0]**2
                        X    = torch.from_numpy(imgi).float().to(device)
                        y, style = self.net(X)
                        loss = self.loss_fn(lbl, y)
                        lavgt += loss.sum().item
                        nsum+=len(loss)
                    print('Epoch %d, Time %4.1fs, Loss %2.4f, Loss Test %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavg, lavgt/nsum, LR))
                else:
                    print('Epoch %d, Time %4.1fs, Loss %2.4f, LR %2.4f'%
                            (iepoch, time.time()-tic, lavg, LR))
                lavg, nsum = 0, 0

            if save_path is not None:
                if iepoch==self.n_epochs-1 or iepoch%save_every==1:
                    # save model at the end
                    file = '{}_{}_{}'.format(self.net_type, file_label, d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                    ksave += 1
                    print('saving network parameters')
                    self.net.save_parameters(os.path.join(file_path, file))
        return os.path.join(file_path, file)
