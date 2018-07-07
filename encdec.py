import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import torchvision.datasets as dsets
from torchvision import transforms


import contrastive
import net

from scipy import ndimage


mb_size = 128
#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


train = dsets.MNIST(
	root='../data/',
	train=True,
	transform = transforms.Compose([transforms.ToTensor()]),
	# transform=transforms.Compose([
	#	  transforms.ToTensor(),
	#	  transforms.Normalize((0.1307,), (0.3081,))
	# ]),
	download=True
)
test = dsets.MNIST(
	root='../data/',
	train=False,
	transform = transforms.Compose([transforms.ToTensor()])
)

train_iter = torch.utils.data.DataLoader(train, batch_size=mb_size, shuffle=True)	
val_iter = torch.utils.data.DataLoader(test, batch_size=mb_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test, batch_size=mb_size, shuffle=True)

use_cuda = True
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
	train,
	batch_size=mb_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(
	test,
	batch_size=mb_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
	test,
	batch_size=mb_size, shuffle=False, **kwargs)


lr = 1e-3
cnt = 0
z_dim = 32

contrastivec = contrastive.ContrastiveLoss()

enc = net.Encoder()
dec = net.Decoder()

enc.cuda()
dec.cuda()


def reset_grad():
	enc.zero_grad()
	dec.zero_grad()

def augment(X):
	#print(X.numpy().shape)
	for i in range(X.size(0)):
		r = ndimage.rotate(X[i, 0].numpy(), np.random.randint(-5, 6), reshape=False)
		#print(r.shape)
		shiftx = np.random.randint(-2,2)
		shifty = np.random.randint(-2,2)
		if shiftx > 0:
			r[shiftx:] = r[0:-shiftx]
		if shiftx < 0:
			r[0:r.shape[0]+shiftx] = r[-shiftx:]	
		if shifty > 0:
			r[:, shifty:] = r[:, 0:-shifty]
		if shifty < 0:
			r[:, 0:r.shape[0]+shifty] = r[:, -shifty:]
			
		X[i, 0] = torch.from_numpy(r)
	
	return X	
	

enc_solver = optim.RMSprop([p for p in enc.parameters()]+[p for p in dec.parameters()], lr=lr)

epoch_len = 64
max_veclen = 0.0
patience = 12*epoch_len
patience_duration = 0



for it in range(1000000):

	if it % epoch_len == 0:
		vec_len = 0.0

	batch_idx, (X, labels) = next(enumerate(train_loader))
	X_aug = augment(X)
	X = Variable(X).cuda()
	X_aug = Variable(X_aug).cuda()
	labels = torch.zeros((mb_size, 1))
	labels = Variable(labels).cuda()

	# Dicriminator forward-loss-backward-update
	mu = enc(X)
	X2 = dec(mu)
	X2d = X2.detach()
	mu2 = enc(X2)
	mu2d = enc(X2d)

	mu_aug = enc(X_aug)
	
	enc_loss = contrastivec(mu[::2], mu[1::2], 0.0*labels[::2])
	enc_loss += contrastivec(mu_aug, mu2, 1.0-0.0*labels)
	enc_loss += 2.0*contrastivec(mu, mu2d, 0.0*labels)
	
	vec_len += torch.mean(torch.sqrt(torch.mean((mu2-(torch.mean(mu2, 0)).repeat(mb_size, 1))**2, 1)))
	
	
	if vec_len.data.cpu().numpy()[0]/epoch_len > max_veclen:
		max_veclen = 1.0*vec_len.data.cpu().numpy()[0]/epoch_len
		patience_duration = 0
	else:
		patience_duration += 1	
	if patience_duration > patience:
		break
	
	enc_loss.backward()
	enc_solver.step()


	# Housekeeping - reset gradient
	reset_grad()

	# Print and plot every now and then
	if it % (epoch_len) == 0:
		plt.close('all')
		#print('Iter-{}; enc_loss: {}; dec_loss: {}'
		#	  .format(it, enc_loss.data.cpu().numpy(), dec_loss.data.cpu().numpy()))

		print('Iter-{}; enc_loss: {}; vec_len: {}, {}'
			  .format(it, enc_loss.data.cpu().numpy(), vec_len.data.cpu().numpy(), max_veclen))

		#print('Iter-{}; enc_loss: {};'
		#	  .format(it, enc_loss.data.cpu().numpy())
			  			  
		samples = X2.data[0:8]
		samples = samples.cpu().numpy()
		originals = X.data[0:8]
		originals = originals.cpu().numpy()
	
		samples = np.append(samples, originals, axis=0)

		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
		plt.pause(0.02)

		if not os.path.exists('out/'):
			os.makedirs('out/')

		#plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
		cnt += 1
		
