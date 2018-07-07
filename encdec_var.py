import matplotlib
#matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np


import os
from torch.autograd import Variable
#from tensorflow.examples.tutorials.mnist import input_data
import torchvision.datasets as dsets
from torchvision import transforms


import contrastive
import net

from scipy import ndimage

plt.close('all')
#fig = plt.gcf()
fig = plt.figure(figsize=(4, 6))
fig.show()
fig.canvas.draw()


def makeplot(fig, samples):
    #fig = plt.figure(figsize=(4, 6))
    gs = gridspec.GridSpec(4, 6)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    fig.canvas.draw()

mb_size = 128

train = dsets.MNIST(
    root='../data/',
    train=True,
    transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()]),

    #transform = transforms.Compose([transforms.ToTensor()]),

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

enc = net.VariationalEncoder()
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
min_veclen = np.inf
patience = 16 #*epoch_len
patience_duration = 0

transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees=7, translate=(2.0/28, 2.0/28)),  transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees=7, translate=(2.0/28, 2.0/28))])


for it in range(1000000):
    if patience_duration > patience:
        break
    if it % epoch_len == 0:
        vec_len = 0.0

    batch_idx, (X, labels) = next(enumerate(train_loader))
    #X_aug = augment(X)

    X_aug = 1.0*X
    #for i in range(X.size(0)):
    #	X_aug[i] = transform(X[i])

    #X_aug = np.fromiter((transform(X[i]) for i in range(X.size(0))), X.numpy().dtype)
    #X_aug = torch.from_numpy(X_aug)

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

    vec_len += torch.mean(torch.sqrt(torch.mean((mu2-(torch.mean(mu2, 0)).repeat(mb_size, 1))**2, 1))).data.cpu().numpy()





    enc_loss.backward()
    enc_solver.step()


    # Housekeeping - reset gradient
    reset_grad()

    # Print and plot every now and then
    if (it+1) % (epoch_len) == 0:
        print(enc.dropout2.p)
        if vec_len/epoch_len < min_veclen:
            patience_duration = 0
            min_veclen = vec_len/epoch_len
            enc.dropout1.p = 0.125
            enc.dropout2.p = 0.125
            enc.dropout3.p = 0.125
            enc.dropout4.p = 0.125

        if vec_len/epoch_len > max_veclen:
            max_veclen = 1.0*vec_len/epoch_len
            patience_duration = 0
            torch.save(enc, 'enc_model.pt')
            torch.save(dec, 'dec_model.pt')

            enc.dropout1.p = 0.0
            enc.dropout2.p = 0.0
            enc.dropout3.p = 0.0
            enc.dropout4.p = 0.0
        else:
            patience_duration += 1




        #print('Iter-{}; enc_loss: {}; dec_loss: {}'
        #	  .format(it, enc_loss.data.cpu().numpy(), dec_loss.data.cpu().numpy()))

        print('Iter-{}; enc_loss: {}; vec_len: {}, {}'
              .format(it, enc_loss.data.cpu().numpy(), vec_len/epoch_len, max_veclen))

        #print('Iter-{}; enc_loss: {};'
        #	  .format(it, enc_loss.data.cpu().numpy())

        #samples = X2.data[0:12]
        #samples = samples.cpu().numpy()
        #originals = X.data[0:12]
        #originals = originals.cpu().numpy()

        samples = X2.data[0:12].cpu().numpy()
        originals =  X.data[0:12].cpu().numpy()

        samples = np.append(samples, originals, axis=0)

        #plt.close('all')
        #plot(samples)
        #plt.pause(0.02)

        makeplot(fig, samples)
        plt.pause(0.001)

        #plt.plot(samples.reshape((6*samples.shape[2], 4*samples.shape[3])))
        #plt.gca().clear()
        #plot(samples)
        #plt.imshow(np.transpose(samples, (2,3,1,0)).reshape((6*samples.shape[2], 4*samples.shape[3])))
        #plt.plot(samples)
        #fig.canvas.draw()
        #plt.pause(0.001)

        if not os.path.exists('out/'):
            os.makedirs('out/')

        #plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1

enc = torch.load('enc_model.pt')
dec = torch.load('dec_model.pt')

mu = enc(X[0:12])
X2 = dec(mu)

samples = np.append(X2.data.cpu().numpy(), X[0:12].data.cpu().numpy(), axis=0)

