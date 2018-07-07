# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self._name = 'mnistG'
		self.dim = 24
		#self.in_shape = int(np.sqrt(self.dim))
		#self.shape = (self.in_shape, self.in_shape, 1)
		preprocess = nn.Sequential(
				nn.Linear(self.dim, 4*4*4*self.dim),
				nn.ReLU(True),
				)
		'''		
		block1 = nn.Sequential(
				nn.ConvTranspose2d(8*self.dim, 4*self.dim, 5),
				nn.ReLU(True),
				)
		block2 = nn.Sequential(
				nn.ConvTranspose2d(4*self.dim, 2*self.dim, 5),
				nn.ReLU(True),
				)
		deconv_out = nn.ConvTranspose2d(2*self.dim, 1, 8, stride=2)
		'''
		self.ups1 = nn.UpsamplingBilinear2d(scale_factor=2)	
		block1 = nn.Sequential(
				#nn.BatchNorm2d(4*self.dim),
				nn.Conv2d(4*self.dim, 2*self.dim, 5, dilation=1,  padding=2),
				nn.ReLU(True),
				)
		self.ups2 = nn.UpsamplingBilinear2d(scale_factor=2)		
		block2 = nn.Sequential(
				#nn.BatchNorm2d(2*self.dim),
				nn.Conv2d(2*self.dim, 1*self.dim, 5, dilation=1, padding=2),
				nn.ReLU(True),
				)
		self.ups3 = nn.UpsamplingBilinear2d(scale_factor=2)		
		deconv_out = nn.Conv2d(1*self.dim, 1, 7,dilation=1, stride=1, padding=3)
		self.block1 = block1
		self.block2 = block2
		self.deconv_out = deconv_out
		self.preprocess = preprocess
		self.sigmoid = nn.Sigmoid()

	def forward(self, input, doprint=False):
		if doprint:
			input = Variable(torch.rand((1,12)))
		output = self.preprocess(input)
		if doprint:
			print(output.size())
		#output = F.dropout(output, p=0.3, training=self.training)
		output = output.view(-1, 4*self.dim, 4, 4)
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
	def __init__(self):
		super(Encoder, self).__init__()
		self._name = 'mnistE'
		self.shape = (1, 28, 28)
		self.dim = 24
		convblock = nn.Sequential(
				#nn.BatchNorm2d(1),
				nn.Conv2d(1, 1*self.dim, 5, dilation=1,  stride=2, padding=2),
				nn.Dropout(p=0.03125),
				nn.ReLU(True),
				#nn.BatchNorm2d(self.dim),
				nn.Conv2d(1*self.dim, 2*self.dim, 5, dilation=1,  stride=2, padding=2),
				nn.Dropout(p=0.03125),
				nn.ReLU(True),
				#nn.BatchNorm2d(2*self.dim),
				nn.Conv2d(2*self.dim, 4*self.dim, 5, dilation=1,  stride=2, padding=2),
				nn.Dropout(p=0.03125),
				nn.ReLU(True),
				#nn.BatchNorm2d(4*self.dim),
				)
		self.main = convblock
		self.output = nn.Linear(4*4*4*self.dim, self.dim)

	def forward(self, input):
		input = input.view(-1, 1, 28, 28)
		out = self.main(input)
		out = out.view(-1, 4*4*4*self.dim)
		out = self.output(out)
		return out.view(-1, self.dim)

		

class VariationalEncoder(nn.Module):
	#can't turn dropout off completely because otherwise the loss -> NaN....
	#batchnorm does not seem to help things...
	def __init__(self):
		super(VariationalEncoder, self).__init__()
		self._name = 'mnistE'
		self.shape = (1, 28, 28)
		self.dim = 24
		
		self.dropout1 = nn.Dropout(p=0.25)
		self.conv1 = nn.Conv2d(1, 1*self.dim, 5, dilation=1,  stride=2, padding=2)
		self.activation1 = nn.ReLU(True)
		self.dropout2 = nn.Dropout(p=0.25)
		self.conv2 = nn.Conv2d(1*self.dim, 2*self.dim, 5, dilation=1,  stride=2, padding=2)
		self.activation2 = nn.ReLU(True)
		self.dropout3 = nn.Dropout(p=0.25)
		self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 5, dilation=1,  stride=2, padding=2)
		self.activation3 = nn.ReLU(True)
		self.dropout4 = nn.Dropout(p=0.25)
		
		
		self.get_mu = nn.Linear(4*4*4*self.dim, self.dim)
		self.get_logvar = nn.Linear(4*4*4*self.dim, self.dim)

	def convblock(self, input):
		x = self.dropout1(input)
		x = self.conv1(x)
		x = self.activation1(x)
		x = self.dropout2(x)
		x = self.conv2(x)
		x = self.activation2(x)
		x = self.dropout3(x)
		x = self.conv3(x)
		x = self.activation3(x)	
		x = self.dropout4(x)	
		return x
	
	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu
			
	def forward(self, input, doprint=False):
		if doprint:
			input = Variable(torch.random(1, 1, 28, 28))
		input = input.view(-1, 1, 28, 28)
		out = self.convblock(input)
		out = out.view(-1, 4*4*4*self.dim)
		if doprint:
			print('out', out.size())
		mu = self.get_mu(out)
		if doprint:
			print('mu', mu.size())
		logvar = self.get_logvar(out)
		logvar = logvar.view(logvar.size(0), -1)
		z = self.reparameterize(mu, logvar)
		return z.view(z.size(0), -1)
		
		
class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

		self.fc1 = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, 20)
		self.fc22 = nn.Linear(400, 20)
		self.fc3 = nn.Linear(20, 400)
		self.fc4 = nn.Linear(400, 784)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def encode(self, x):
		h1 = self.relu(self.fc1(x))
		return self.fc21(h1), self.fc22(h1)

	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu

	def decode(self, z):
		h3 = self.relu(self.fc3(z))
		return self.sigmoid(self.fc4(h3))

	def forward(self, x):
		mu, logvar = self.encode(x.view(-1, 784))
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar		 