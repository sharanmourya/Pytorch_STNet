import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import scipy.io as sio 
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from collections import OrderedDict

img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
encoded_dim = 512 #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
img_size = 32
num_heads = 4 # for multi-head attention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
depth = 1 # No of STB 
qkv_bias=True
window = 8 # window size for LSA
envir = 'indoor'

class GroupAttention(nn.Module):

    def __init__(self, num_heads=4, qkv_bias=False):
        super(GroupAttention, self).__init__()

        self.num_heads = num_heads
        head_dim = img_size // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(img_size, img_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(img_size, img_size)
        self.ws = window

    def forward(self, x):
        B, C, H, W = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, C, h_group, self.ws, W)
        qkv = self.qkv(x).reshape(B, C, total_groups, -1, 3, self.num_heads, self.ws // self.num_heads).permute(4, 0, 1, 2, 5, 3, 6)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        return x

class GlobalAttention(nn.Module):

    def __init__(self, num_heads=4, qkv_bias=False):
        super().__init__()

        self.dim = img_size
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim//window, self.dim//window * 2, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        self.sr = nn.Conv2d(2, 2, kernel_size=window, stride=window)
        self.norm = nn.LayerNorm(self.dim//window)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, -1, self.dim//window, self.dim//window).permute(0,1,3,2,4)
        x_ = self.sr(x).reshape(B, C, -1, self.dim//window, self.dim//window)
        x_ = self.norm(x_)
        kv = self.kv(x_).reshape(B, C, -1, 2, self.dim//window, self.dim//window).permute(3,0,1,4,2,5)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)

        return x

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.cc1 = nn.Linear(img_size, img_size)
        self.cc2 = nn.Linear(img_size, img_size)
        self.act = nn.GELU()

    def forward(self, x):

        x = self.cc1(x)
        x = self.act(x)
        x = self.cc2(x)

        return x


class WTL(nn.Module):
    def __init__(self, num_heads, qkv_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(img_size, eps=1e-6)
        self.attn1 = GroupAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
        )
        self.attn2 = GlobalAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm4 = nn.LayerNorm(img_size, eps=1e-6)
        self.mlp1 = MLP()
        self.mlp2 = MLP()

    def forward(self, x):

        x = x + self.attn1(self.norm1(x))
        x = x + self.mlp1(self.norm2(x))
        x = x + self.attn2(self.norm3(x))
        x = x + self.mlp2(self.norm4(x))

        return x

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            img_size=img_size,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
    ):
        super().__init__()


        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.conv1 = nn.Conv2d(2,16, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(16,2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(2*img_size*img_size, encoded_dim)

    def forward(self, x):

        n_samples = x.shape[0]
        x = self.conv1(x)
        x = self.conv5(x)
        X = x 

        for block in self.blocks:
            x = block(x)
        x = self.norm3(x)
        x = self.convT(x) 
        x = X + self.conv4(x)
        x = self.norm2(x)
        x = x.reshape(n_samples,2*img_size*img_size)
        x = self.fc(x)
        return x


class Decoder(nn.Module):   
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(encoded_dim, img_channels*img_size*img_size)
        self.act = nn.Sigmoid()
        self.conv5 = nn.Conv2d(2,2, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)

        self.dense_layers = nn.Sequential(
            nn.Linear(encoded_dim, img_total)
        )

        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)

    def forward(self, x):
        img = self.dense_layers(x)
        img = img.view(-1, img_channels, img_height, img_width)

        out = self.decoder_feature(img)
        x = self.conv5(img)

        for block in self.blocks:
            x = block((x+out))

        x = self.norm2(x)
        x = self.convT(x)
        x = self.conv4(x) 

        for block in self.blocks:
            x = block((x+out))

        x = self.norm3(x)

        x = self.act(x) 

        return x

encoder = Encoder()
encoder.to(device)
print(encoder)

decoder = Decoder()
decoder.to(device)
print(decoder)

print('Data loading begins.....')

if envir == 'indoor':
    mat = sio.loadmat('../data/DATA_Htrainin.mat') 
    x_train = mat['HT'] 
    mat = sio.loadmat('../data/DATA_Hvalin.mat')
    x_val = mat['HT'] 
    mat = sio.loadmat('../data/DATA_Htestin.mat')
    x_test = mat['HT'] 

elif envir == 'outdoor':
    mat = sio.loadmat('../data/DATA_Htrainout.mat') 
    x_train = mat['HT'] 
    mat = sio.loadmat('../data/DATA_Hvalout.mat')
    x_val = mat['HT'] 
    mat = sio.loadmat('../data/DATA_Htestout.mat')
    x_test = mat['HT']

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))

print('Data loading done!')


x_train = x_train.to(device, dtype=torch.float)

x_test = x_test.to(device, dtype=torch.float)

def train_autoencoder(uncompressed_images, opt_enc, opt_dec):
    opt_enc.zero_grad()
    opt_dec.zero_grad()

    compressed_data = encoder.forward(uncompressed_images)

    reconstructed_images = decoder.forward(compressed_data)

    loss = nn.MSELoss()
    grads = loss(uncompressed_images, reconstructed_images)
    grads.backward()
    opt_enc.step()
    opt_dec.step()
    return grads.item()

def fit(epochs, lr, start_idx=1):
    
    losses_dec = []
    losses_auto = []

    opt_enc = Adam(encoder.parameters(), lr, betas=(0.5, 0.999))
    opt_dec = Adam(decoder.parameters(), lr, betas=(0.5, 0.999))
    
    reps = int(len(x_train) / (batch_size))

    for epoch in range(epochs):
        x_train_idx = torch.randperm(x_train.size()[0])
        for i in range(reps):
            loss_auto= train_autoencoder(x_train[x_train_idx[i*batch_size:(i+1)*batch_size]], opt_enc, opt_dec)
            if i % 600 == 0:               
                print('epoch',epoch+1,'/',epochs,'batch:',i+1,'/',reps, "loss_auto: {:.12f}".format(loss_auto))
            losses_auto.append(loss_auto)

    return losses_auto

epochs = 1000
lr = 0.001
batch_size = 200
print('training starts.....')
losses_auto = fit(epochs, lr)

plt.figure(figsize=(10,5))
plt.title("autoencoder and aecoder Loss During Training")
plt.plot(losses_auto,label="AE")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

del losses_auto
del x_train
n = 10
with torch.no_grad():
    temp = encoder.forward(x_test[0:1000,:,:,:])
    x_hat = decoder.forward(temp)

x_test_in = x_test[0:1000,:,:,:].to('cpu')
x_hat_in = x_hat.to('cpu')


x_test_in = x_test_in.numpy()
x_hat_in = x_hat_in.numpy()



plt.figure(figsize=(20, 4))
for i in range(n):
    # display origoutal
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test_in[i, 0, :, :]-0.5 + 1j*(x_test_in[i, 1, :, :]-0.5))
    plot = np.max(np.max(x_testplo))-x_testplo.T
    # print(plot)
    # print(np.mean(plot))
    # plot = plot - np.mean(plot)
    # print(plot)
    plt.imshow(plot)
    # plt.imshow(x_testplo)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat_in[i, 0, :, :]-0.5 
                          + 1j*(x_hat_in[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    # plt.imshow(decoded_imgsplo)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.show()


x_test_real = np.reshape(x_test_in[:, 0, :, :], (len(x_test_in), -1))
x_test_imag = np.reshape(x_test_in[:, 1, :, :], (len(x_test_in), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat_in[:, 0, :, :], (len(x_hat_in), -1))
x_hat_imag = np.reshape(x_hat_in[:, 1, :, :], (len(x_hat_in), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

power = np.sum(abs(x_test_C)**2, axis=1)
mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)


print("NMSE is ", 10*math.log10(np.mean(mse/power)))


