import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils import init_kernel_SR, mask, conv2, warp

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c = 1
        self.factor = 2
        self.numb = 9
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Init FeatureExtractor
        self.FE = dict()
        for i in range(self.numb):
            self.FeatureExtract(i)

        # Init Estimate Optical Flow Net
        self.FL = dict()
        self.Flow()

        # Init Refine Network
        self.RF = dict()
        self.RefineNet()

        self.Pr = nn.Parameter(torch.from_numpy(init_kernel_SR()).cuda())
        self.mask = nn.Parameter(torch.from_numpy(mask()).cuda())
        self.mask.requires_grad = False

        self._initialize_weights()
        self._get_params()

    def forward(self, x, v):
        # Estimation Optical Flow
        flows = []
        basis = x[..., self.numb//2]
        for i in range(self.numb):
            if i == self.numb // 2:
                flows.append(None)
                continue
            flows.append(self.Flow_forward(torch.cat((basis, x[..., i]), 1)))

        # Extract Feature
        x = self.FE_forward(x)

        # Warp Feature & Concat Feature
        
        x_ = x[..., self.numb//2]
        
        for i in range(self.numb):
            if i == self.numb // 2:
                continue
            warped = warp(x[..., i], flows[i])

            x_ = torch.cat((x_, warped), 1)
        pruning_kernel = self.Pr[v[0][1]] * self.mask
        weights = pruning_kernel.view(32*9, 32*9, 1, 1)
        x_ = F.conv2d(x_, weights)
        # Refined(Mix) Feature & Get outputs
        x = self.RefineNet_forward(x_)
        return x

    def _initialize_weights(self):
        for name in self.FE:
            init.orthogonal_(self.FE[name].weight, init.calculate_gain('leaky_relu'))

        for name in self.FL:
            if name in 'conv3':
                init.orthogonal_(self.FL[name].weight)
            else:
                init.orthogonal_(self.FL[name].weight, init.calculate_gain('leaky_relu'))

        for name in self.RF:
            if name in 'output':
                init.orthogonal_(self.RF[name].weight)
            else:
                init.orthogonal_(self.RF[name].weight, init.calculate_gain('leaky_relu'))

    def _get_params(self):
        self.params = []
        for name in self.FE:
            self.params += list(self.FE[name].parameters())
        for name in self.FL:
            self.params += list(self.FL[name].parameters())
        for name in self.RF:
            self.params += list(self.RF[name].parameters())
        #for name in self.SU_:
        self.params += list([self.Pr])

    def FeatureExtract(self, iter):
        self.FE['%d_conv1' % iter] = conv2(self.c, 32)
        self.FE['%d_conv2' % iter] = conv2(32, 32)

    def Flow(self):
        def conv_level(level):
            c2 = 16*(2**level)
            if level == 0:
                c1 = self.c * 2
            else:
                c1 = c2

            self.FL['%d_level1' % level] = conv2(c1, c2)
            self.FL['%d_level2' % level] = conv2(c2, c2)
            self.FL['%d_level3' % level] = conv2(c2, c2*2, stride=2)

        for i in range(2):
            conv_level(i)
        c = 16*(2**(i+1))
        self.FL['conv1'] = conv2(c, c)
        self.FL['conv2'] = conv2(c, c)
        self.FL['conv3'] = conv2(c, 2)

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
        self.RF['conv2'] = conv2(64, 8*self.factor**2, kernel=1)
        self.pixel_shuffle = nn.PixelShuffle(self.factor)
        self.RF['conv3'] = conv2(8, 64)
        self.RF['output'] = conv2(64, self.c, kernel=1)

    def FE_forward(self, x):
        y = []
        for i in range(self.numb):
            x_ = x[..., i]
            for j in range(2):
                j += 1
                x_ = self.relu(self.FE['%d_conv%d' % (i, j)](x_))
            y.append(x_)
        return torch.stack(y, 4)

    def Flow_forward(self, x):
        _, _, h, w = x.size()
        for i in range(2):
            for j in range(3):
                j += 1
                x = self.relu(self.FL['%d_level%d' % (i, j)](x))
        for i in range(2):
            i += 1
            x = self.relu(self.FL['conv%d'%i](x))

        return nn.Upsample(size=(h, w))(self.FL['conv3'](x))

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

        x = self.pixel_shuffle(x)
        x = self.relu(self.RF['conv3'](x))
        return self.RF['output'](x)

    def select_forward(self, x, id):
        B, C, H, W = x.size()

        id = id.view(B, 1, -1, 1, 1)
        id = id.repeat(1, 1, 1, H, W)

        xx = torch.arange(0, W).view(1, 1, -1).repeat(C, H, 1).view(1, 1, C, H, W).repeat(B, 1, 1, 1, 1)
        yy = torch.arange(0, H).view(1, -1, 1).repeat(C, 1, W).view(1, 1, C, H, W).repeat(B, 1, 1, 1, 1)
        cc = id

        grid = torch.cat((xx, yy), 1).float()

        vgrid = Variable(grid).cuda()

        vgrid[:, 1] = 2.0 * vgrid[:, 1].clone() / max(H - 1, 1) - 1.0
        vgrid[:, 0] = 2.0 * vgrid[:, 0].clone() / max(W - 1, 1) - 1.0
        vgrid = torch.cat((vgrid, cc), 1)
        vgrid_ = vgrid.permute(0, 2, 3, 4, 1)
        x = x.view(1, 1, C, H, W)
        output = nn.functional.grid_sample(x, vgrid_).view(1, C, H, W)
        return output
