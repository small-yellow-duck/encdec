import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.optim as optim
import numpy as np


import os
from torch.autograd import Variable
import torchvision.datasets as dsets
from torchvision import transforms


import contrastive
import net as net



use_cuda = True
print(f'use_cuda {use_cuda}')

mb_size = 128
lr = 1.0e-4
cnt = 0
z_dim = 32 #24

plt.close('all')
#fig = plt.gcf()
fig = plt.figure(figsize=(4, 4))
fig.show()
fig.canvas.draw()


def makeplot(fig, samples):
    #fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    fig.canvas.draw()


train = dsets.MNIST(
    root='../data/',
    train=True,
    transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()]),
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



contrastiveloss = contrastive.ContrastiveLoss()
#KLloss = contrastive.KL()

#enc = net.VariationalEncoder(dim=z_dim)
enc = net.Encoder(dim=z_dim)
#dec = net.Decoder(output_dim=(28, 28))
dec = net.Decoder(dim=z_dim)

if use_cuda:
    enc.cuda()
    dec.cuda()


def reset_grad():
    enc.zero_grad()
    dec.zero_grad()

def plot(samples):
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

    return fig


enc_solver = optim.RMSprop([p for p in enc.parameters()]+[p for p in dec.parameters()], lr=lr)

epoch_len = 64 #4 #
max_veclen = 0.0
min_veclen = np.inf
patience = 16 #*epoch_len
patience_duration = 0
vec_len = 0.0
loss = 0.0

transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees=7, translate=(2.0/28, 2.0/28)),  transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees=7, translate=(2.0/28, 2.0/28))])

mask = torch.ones((mb_size, 1, 28, 28))
mask[::3, :, 0:14, :] = 0.0*mask[::3, :, 0:14, :]
mask[1::3, :, :, 0:14] = 0.0*mask[1::3, :, :, 0:14]
mask = Variable(mask)
if use_cuda:
    mask = mask.cuda()


for it in range(1000000):
    if patience_duration > patience:
        break
    if it % epoch_len == 0:
        vec_len = 0.0

    batch_idx, (X, labels) = next(enumerate(train_loader))

    #degrees = np.random.randint(-180, 180)
    #transform_fwd = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation((degrees, degrees+1)),  transforms.ToTensor()])
    #transform_bwd = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation((-degrees, -degrees+1)),  transforms.ToTensor()])


    #X_aug = augment(X)

    X = Variable(X)
    #X_aug = Variable(X_aug)

    if use_cuda:
        X = X.cuda()
        #X_aug = X_aug.cuda()
    labels = torch.zeros((mb_size, 1))
    labels = Variable(labels)

    if use_cuda:
        labels = labels.cuda()

    # Dicriminator forward-loss-backward-update
    mu = enc(X)
    #mu_aug = enc(X_aug)
    #mu, logsigma = enc(X)
    #mu_aug = enc(mask*(X.flip(2 + it % 2)))
    X2 = dec(mu)
    #X2_aug = dec(mu_aug)
    #X2_aug = (mask*(dec(mu_aug)).flip(2 + it % 2))
    #mu2_aug = enc(X2_aug)
    X2d = X2.detach()
    #mu2, logsigma2 = enc(X2)
    mu2 = enc(X2)
    #mu2_aug = enc(X2_aug)
    mu2d = enc(X2d)

    enc_loss = contrastiveloss(mu[::2], mu[1::2], 0.0 * labels[::2])
    enc_loss += contrastiveloss(mu, mu2, 1.0 - 0.0 * labels)
    enc_loss += 2.0 * contrastiveloss(mu, mu2d, 0.0 * labels)


    #vec_len += torch.mean(torch.sqrt(torch.mean((mu2 - (torch.mean(mu2, 0)).repeat(mb_size, 1)) ** 2, 1))).data.cpu().numpy()

    vec_len += 0.5 * (torch.mean(torch.pow(mu, 2)) + torch.mean(torch.pow(mu2, 2))).data.cpu().numpy()




    enc_loss.backward()
    enc_solver.step()

    loss += enc_loss.data.cpu().numpy()


    # Housekeeping - reset gradient
    reset_grad()



    # Print and plot every now and then
    if it % (epoch_len) == 0:
        #plt.close('all')
        #print('Iter-{}; enc_loss: {}; dec_loss: {}'
        #	  .format(it, enc_loss.data.cpu().numpy(), dec_loss.data.cpu().numpy()))

        vec_len = vec_len/epoch_len
        loss = loss / epoch_len
        print('Iter-{}; enc_loss: {}; vec_len: {}, {}'
              .format(it, loss, vec_len, max_veclen))
        vec_len = 0.0
        loss = 0.0


        samples = X2.data[0:8]
        samples = samples.cpu().numpy()
        originals = X.data[0:8]
        originals = originals.cpu().numpy()

        #print(samples.shape)
        #print(originals.shape)
        samples = np.append(samples, originals, axis=0)

        makeplot(fig, samples)
        plt.pause(0.001)




        if not os.path.exists('out/'):
            os.makedirs('out/')

        #plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1

enc = torch.load('enc_model.pt')
dec = torch.load('dec_model.pt')

mu = enc(X[0:12])
X2 = dec(mu)

samples = np.append(X2.data.cpu().numpy(), X[0:12].data.cpu().numpy(), axis=0)

