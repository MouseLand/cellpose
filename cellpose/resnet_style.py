from mxnet import gluon, nd
from mxnet.gluon import nn
import numpy as np

nfeat = 128
sz  = [3, 3, 3, 3, 3]
sz2 = [3, 3, 3, 3, 3]
szf = [1]


def total_variation_loss(x):
    """ regularize convolutional masks (not currently in use) """
    a = nd.square(x[:, :, :-1, :-1] - x[:, :, 1:, :-1])
    b = nd.square(x[:, :, :-1, :-1] - x[:, :, :-1, 1:])
    return nd.sum(nd.mean(nd.power(a + b, 1.25), axis=(2,3)))

def convbatchrelu(nconv, sz):
    conv = nn.HybridSequential()
    with conv.name_scope():
        conv.add(
                nn.Conv2D(nconv, kernel_size=sz, padding=sz//2),
                nn.BatchNorm(axis=1),
                nn.Activation('relu'),
        )
    return conv

def batchconv(nconv, sz):
    conv = nn.HybridSequential()
    with conv.name_scope():
        conv.add(
                nn.BatchNorm(axis=1),
                nn.Activation('relu'),
                nn.Conv2D(nconv, kernel_size=sz, padding=sz//2),
        )
    return conv

def batchconv0(nconv, sz):
    conv = nn.HybridSequential()
    with conv.name_scope():
        conv.add(
                nn.BatchNorm(axis=1),
                nn.Conv2D(nconv, kernel_size=sz, padding=sz//2),
        )
    return conv

class resdown(nn.HybridBlock):
    def __init__(self, nconv, **kwargs):
        super(resdown, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            for t in range(4):
                self.conv.add( batchconv(nconv, 3))
            self.proj  = batchconv0(nconv, 1)

    def hybrid_forward(self, F, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class downsample(nn.HybridBlock):
    def __init__(self, nbase, **kwargs):
        super(downsample, self).__init__(**kwargs)
        with self.name_scope():
            self.down = nn.HybridSequential()
            for n in range(len(nbase)):
                self.down.add(resdown(nbase[n]))

    def hybrid_forward(self, F, x):
        xd = []
        for n in range(len(self.down)):
            if n>0:
                y = F.Pooling(xd[n-1], kernel=(2,2), stride=(2,2), pool_type='max')
            else:
                y = x
            xd.append(self.down[n](y))
        return xd

class batchconvstyle(nn.HybridBlock):
    def __init__(self, nconv, **kwargs):
        super(batchconvstyle, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = batchconv(nconv, 3)
            self.full = nn.Dense(nconv)

    def hybrid_forward(self, F, style, x):
        feat = self.full(style)
        y = F.broadcast_add(x, feat.expand_dims(-1).expand_dims(-1))
        y = self.conv(y)
        return y

class resup(nn.HybridBlock):
    def __init__(self, nconv, **kwargs):
        super(resup, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            self.conv.add(batchconv(nconv,3))
            for n in range(3):
                self.conv.add(batchconvstyle(nconv))
            self.proj  = batchconv0(nconv, 1)
            #self.proj2  = batchconv0(nconv, 1)

    def hybrid_forward(self, F, x, y, style):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y)
        x = x + self.conv[3](style, self.conv[2](style, x))
        return x

class upsample(nn.HybridBlock):
    def __init__(self, nbase, **kwargs):
        super(upsample, self).__init__(**kwargs)
        with self.name_scope():
            self.up = nn.HybridSequential()
            for n in range(len(nbase)):
                self.up.add(resup(nbase[n]))

    def hybrid_forward(self, F, style, xd):
        x= self.up[-1](xd[-1], xd[-1], style)
        for n in range(len(self.up)-2,-1,-1):
            x= F.UpSampling(x, scale=2, sample_type='nearest')
            x = self.up[n](x, xd[n], style)
        return x

class make_style(nn.HybridBlock):
    def __init__(self,  **kwargs):
        super(make_style, self).__init__(**kwargs)
        with self.name_scope():
            self.pool_all = nn.GlobalAvgPool2D()
            self.flatten = nn.Flatten()

    def hybrid_forward(self, F, x0):
        style = self.pool_all(x0)
        style = self.flatten(style)
        style = F.broadcast_div(style , F.sum(style**2, axis=1).expand_dims(1)**.5)

        return style

class CPnet(gluon.HybridBlock):
    def __init__(self, nbase, nout, **kwargs):
        super(CPnet, self).__init__(**kwargs)
        with self.name_scope():
            #self.conv1 = convbatchrelu(16, 3)
            self.downsample = downsample(nbase)
            self.upsample = upsample(nbase)
            self.output = batchconv(nout, 1)
            self.make_style = make_style()

    def hybrid_forward(self, F, data):
        #data     = self.conv1(data)
        T0    = self.downsample(data)
        style = self.make_style(T0[-1])
        T0    = self.upsample(style, T0)
        T0    = self.output(T0)

        return T0, style
