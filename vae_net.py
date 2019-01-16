# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Decoder(nn.Module):
    def __init__(self, dim=24):
        super(Decoder, self).__init__()
        self._name = 'mnistG'
        self.dim = dim
        #self.in_shape = int(np.sqrt(self.dim))
        #self.shape = (self.in_shape, self.in_shape, 1)
        preprocess = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(self.dim, 4 * 4 * 4 * self.dim), dim=None),
                # nn.Linear(self.dim, 4 * 4 * 4 * self.dim),
                nn.ReLU(True),
                )
        self.ups1 = nn.Upsample(scale_factor=2, mode='bilinear')#nn.UpsamplingBilinear2d(scale_factor=2) #nn.UpsamplingNearest2d(scale_factor=2)
        self.block1 = nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(4*self.dim, 2*self.dim, 5, dilation=1,  padding=2)),
                #nn.Conv2d(4 * self.dim, 2 * self.dim, 3, dilation=1, padding=1),
                nn.ReLU(True),
                )
        self.ups2 = nn.Upsample(scale_factor=2, mode='bilinear')#nn.UpsamplingBilinear2d(scale_factor=2) #nn.UpsamplingNearest2d(scale_factor=2)
        self.block2 = nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(2*self.dim, 1*self.dim, 5, dilation=1, padding=2)),
                #nn.Conv2d(2 * self.dim, 1 * self.dim, 3, dilation=1, padding=1),
                nn.ReLU(True),
                )
        self.ups3 = nn.Upsample(scale_factor=2, mode='bilinear')#nn.UpsamplingBilinear2d(scale_factor=2) #nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_out = nn.utils.weight_norm(nn.Conv2d(1*self.dim, 1, 5, dilation=1, stride=1, padding=2))

        #self.deconv_out = nn.Conv2d(1 * self.dim, 1, 3, dilation=1, stride=1, padding=1)
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, inpt, doprint=False):
        if doprint:
            inpt = Variable(torch.rand((1, self.dim)))
        output = self.preprocess(inpt)
        if doprint:
            print(output.size())
        #output = F.dropout(output, p=0.3, training=self.training)
        #output = output.view(-1, 4*self.dim, 7, 7)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.ups1(output)
        if doprint:
            print('ups1', output.size())
        output = self.block1(output)
        if doprint:
            print('block1', output.size())


        #output = F.dropout(output, p=0.3, training=self.training)
        output = output[:, :, :7, :7]
        output = self.ups2(output)
        if doprint:
            print('ups2', output.size())
        output = self.block2(output)
        if doprint:
            print('block 2', output.size())

        output = self.ups3(output)
        if doprint:
            print('ups3', output.size())
        #output = F.dropout(output, p=0.3, training=self.training)
        output = self.deconv_out(output)
        if doprint:
            print('deconv', output.size())
        output = self.sigmoid(output)
        return output.view(-1, 1, 28, 28)

#https://github.com/neale/Adversarial-Autoencoder/blob/master/generators.py
class Encoder(nn.Module):
    #can't turn dropout off completely because otherwise the loss -> NaN....
    #batchnorm does not seem to help things...
    def __init__(self, dim=24):
        super(Encoder, self).__init__()
        self._name = 'mnistE'
        self.shape = (1, 28, 28)
        self.dim = dim
        self.dropout = 0.03125 #0.125 #
        convblock = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.utils.weight_norm(nn.Conv2d(1, 1 * self.dim, 5, dilation=1, stride=1, padding=2)),
                #nn.Conv2d(1, 1 * self.dim, 3, dilation=1, stride=1, padding=1),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.utils.weight_norm(nn.Conv2d(1*self.dim, 2*self.dim, 5, dilation=1, stride=1, padding=2)),
                #nn.Conv2d(1 * self.dim, 2 * self.dim, 3, dilation=1, stride=1, padding=1),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.utils.weight_norm(nn.Conv2d(2*self.dim, 4*self.dim, 5, dilation=1,  stride=1, padding=2)),
                #nn.Conv2d(2 * self.dim, 4 * self.dim, 3, dilation=1, stride=2, padding=1),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                nn.MaxPool2d(2)
                )
        self.main = convblock
        self.output = convblock = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.utils.weight_norm(nn.Linear(4 * 3 * 3 * self.dim, self.dim), dim=None),
                #nn.ReLU(True),
                #nn.utils.weight_norm(nn.Linear(4 * 4 * 4 * self.dim, self.dim), dim=None)
        )

        #self.output = nn.utils.weight_norm(nn.Linear(4*4*4*self.dim, self.dim), dim=None)
        #self.output = nn.Linear(4 * 4 * 4 * self.dim, self.dim)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(input.size(0), -1)
        out = self.output(out)
        return out.view(-1, self.dim)



class VariationalEncoder(nn.Module):
    #can't turn dropout off completely because otherwise the loss -> NaN....
    #batchnorm does not seem to help things...
    def __init__(self, dim=24):
        super(VariationalEncoder, self).__init__()
        self._name = 'mnistE'
        self.shape = (1, 28, 28)
        self.dim = dim
        self.dropout = 0.03125
        convblock = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.utils.weight_norm(nn.Conv2d(1, 1 * self.dim, 5, dilation=1, stride=1, padding=2)),
                #nn.Conv2d(1, 1 * self.dim, 3, dilation=1, stride=1, padding=1),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.utils.weight_norm(nn.Conv2d(1*self.dim, 2*self.dim, 5, dilation=1, stride=1, padding=2)),
                #nn.Conv2d(1 * self.dim, 2 * self.dim, 3, dilation=1, stride=1, padding=1),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.utils.weight_norm(nn.Conv2d(2*self.dim, 4*self.dim, 5, dilation=1,  stride=1, padding=2)),
                #nn.Conv2d(2 * self.dim, 4 * self.dim, 3, dilation=1, stride=2, padding=1),
                nn.Dropout(p=self.dropout),
                nn.ReLU(True),
                nn.MaxPool2d(2)
                )

        self.main = convblock

        #self.get_mu = nn.Linear(4*4*4*self.dim, self.dim)
        self.get_mu = nn.utils.weight_norm(nn.Linear(4 * 3 * 3 * self.dim, self.dim))
        self.get_logvar = nn.utils.weight_norm(nn.Linear(4 * 3 * 3 * self.dim, self.dim))
        #self.get_logvar = nn.Linear(4*4*4*self.dim, self.dim)


    def reparameterize(self, mu, logvar, do_reparameterize=False):
        if self.training and do_reparameterize:
            #std = logvar.mul(0.5).exp_()
            std = torch.exp(0.5*logvar)
            #std = 0.5*torch.ones_like(mu)
            eps = Variable(std.data.new(std.size()).normal_())
            return mu + eps*std
        else:
            return mu


    def forward(self, input, do_reparameterize=True):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(out.size(0), -1)
        mu = self.get_mu(out)
        logvar = self.get_logvar(out)
        z = self.reparameterize(mu, logvar, do_reparameterize=do_reparameterize)
        return z.view(z.size(0), -1), logvar