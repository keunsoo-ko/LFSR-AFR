import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils import init_kernel_AR, warp, conv2
import numpy as np

class Blend(nn.Module):
    def __init__(self):
        super(Blend, self).__init__()
        self.c = 1
        self.numb = 4
        self.k_size = 3
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Init FeatureExtractor
        self.FE = dict()
        for i in range(self.numb):
            self.FeatureExtract(i)

        # Init Refine Network
        self.RF = dict()
        self.RefineNet()

        self.Pr = nn.Parameter(torch.from_numpy(init_kernel_AR()).cuda())

        self._initialize_weights()
        self._get_params()

    def forward(self, x, v, flows):
        inputs = x
        # Extract Feature
        x = self.FE_forward(x)
        warped = []
        inputs_ = []
        for i in range(self.numb):
            if v == 1:
              if i > 1:
                warped.append(x[i])
                inputs_.append(inputs[i])
              else:
                warped.append(warp(x[i], flows[i]))
                inputs_.append(warp(inputs[i], flows[i]))
            elif v == 2:
              if (i == 1 or i == 3):
                warped.append(x[i])
                inputs_.append(inputs[i])
              else:
                warped.append(warp(x[i], flows[i]))
                inputs_.append(warp(inputs[i], flows[i]))
            else:
                warped.append(warp(x[i], flows[i]))
                inputs_.append(warp(inputs[i], flows[i]))
        
        x = torch.cat(warped, 1)
        inputs = torch.cat(inputs_, 1)
        b, c, h, w = inputs.shape

        pruning_kernel = self.Pr[v]
        weights = pruning_kernel.view(32*4, 32*4, 1, 1)
        x = F.conv2d(x, weights)
        # Refined(Mix) Feature & Get outputs
        ex = self.RefineNet_forward(x)

        x = F.pad(inputs, (self.k_size//2, self.k_size//2, self.k_size//2, self.k_size//2))
        patches = x.unfold(2, self.k_size, 1).unfold(3, self.k_size, 1)
        patches = patches.permute(0, 1, 4, 5, 2, 3).reshape(b, -1, h, w)
        output = patches.view(b, 1, c*(self.k_size**2), h, w)\
                     * ex.view(b, self.c, c*(self.k_size**2), h, w)
        return output.sum(2)

    def _initialize_weights(self):
        for name in self.FE:
            init.orthogonal_(self.FE[name].weight, init.calculate_gain('leaky_relu'))

        for name in self.RF:
            if name in 'output':
                init.orthogonal_(self.RF[name].weight)
            else:
                init.orthogonal_(self.RF[name].weight, init.calculate_gain('leaky_relu'))

    def _get_params(self):
        self.params = []
        for name in self.FE:
            self.params += list(self.FE[name].parameters())
        for name in self.RF:
            self.params += list(self.RF[name].parameters())
        #for name in self.SU_:
        self.params += list([self.Pr])

    def FeatureExtract(self, iter):
        self.FE['%d_conv1' % iter] = conv2(self.c, 32)
        self.FE['%d_conv2' % iter] = conv2(32, 32)

    def RefineNet(self):
        def RDB(name, iter):
            name = name + '_%d'
            for i in range(2):
                self.RF[name % i + '_conv1'] = conv2(64*(1+i+iter), 64)
            self.RF[name % i + '_conv'] = conv2(64*(2+i+iter), 64, kernel=1)

        self.RF['concat1'] = conv2(32*self.numb, 32*(int(self.numb**0.5)))
        self.RF['concat2'] = conv2(32*(int(self.numb**0.5)), 64)

        for i in range(2):
            RDB('RDN%d'%i, i)

        self.RF['concat3'] = conv2(192, 64, kernel=1)
        self.RF['conv1'] = conv2(64, 64)
        self.RF['conv2'] = conv2(64, 64, kernel=1)
        self.RF['conv3'] = conv2(64, 64)
        self.RF['output'] = conv2(64, self.c*(self.c*4)*9, kernel=1)

    def FE_forward(self, x):
        y = []
        for i in range(self.numb):
            x_ = x[i]
            for j in range(2):
                j += 1
                x_ = self.relu(self.FE['%d_conv%d' % (i, j)](x_))
            y.append(x_)
        return y#torch.cat(y, 1)

    def RefineNet_forward(self, x):
        def RDB(x, name):
            name = name + '_%d'
            res = x
            for i in range(2):
                x = self.relu(self.RF[name % i + '_conv1'](res))
                res = torch.cat((res, x), 1)
            x = self.relu(self.RF[name % i + '_conv'](res))
            return x

        for i in range(2):
            i += 1
            x = self.relu(self.RF['concat%d'%i](x))
        res = x
        for i in range(2):
            x = RDB(res, 'RDN%d'%i)
            res = torch.cat((res, x), 1)
        x = self.relu(self.RF['concat3'](res))
        for i in range(2):
            i += 1
            x = self.relu(self.RF['conv%d'%i](x))

        x = self.relu(self.RF['conv3'](x))
        return self.RF['output'](x)
