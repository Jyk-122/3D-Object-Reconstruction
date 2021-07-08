import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
import torchvision.transforms as transforms
import PIL.Image as Image
import SimpleITK as itk
import imageio

import sys
sys.path.append('/home/vision/diska2/1JYK/HQ/models/VolT/')
from layers import *
from attentions import *


def binary(data, eps):
    data[data >= eps] = 1
    data[data < eps] = 0
    return data

def read_img(file_path, cuda_device=0):
    trans = transforms.Compose([transforms.ToTensor()])
    img = Image.open(file_path).convert('L')
    img = trans(img)
    img = img.cuda(cuda_device)
    return img.squeeze()


def read_mha(file_path, cuda_device=0):
    sitkimage = itk.ReadImage(file_path)
    volume = itk.GetArrayFromImage(sitkimage)
    volume = torch.Tensor(volume).cuda(cuda_device)
    return volume.squeeze()


def save_img(path, img):
    img = img.squeeze()
    img = img.data.cpu().numpy()
    imageio.imwrite(path, img)


def save_mha(path, mha):
    outputs = mha.squeeze()
    outputs = outputs.data.cpu().numpy()
    volume = itk.GetImageFromArray(outputs, isVector=False)
    itk.WriteImage(volume, path)

def save_model(model, path):
    torch.save(model, path)


def load_model(path, cuda_device):
    if torch.cuda.is_available():
        model = torch.load(path, map_location=lambda storage, loc: storage.cuda(cuda_device))
    else:
        model = torch.load(path)
    model.eval()
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path, cuda_device=0):
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda(cuda_device)
    return model, optimizer, epoch, loss


class block_2d(nn.Module):
    def __init__(self, dim=64, heads=8, head_dim=32, chunks=1, ff_mult=4, dropout=0.):
        super().__init__()
        self.attn = MH_DEAttn(dim=dim, heads=heads, head_dim=head_dim, dropout=dropout)
        self.ff = FF(chunks=chunks, dim=dim, ff_mult=ff_mult)
        # self.act = nn.ReLU()

    def forward(self, x, x0):
        x = self.attn(x, x0)
        x = self.ff(x)
        # x = self.act(x)
        return x


class block_3d(nn.Module):
    def __init__(self, dim=64, heads=8, head_dim=32, chunks=1, ff_mult=4, dropout=0.):
        super().__init__()
        self.attn1 = MultiHeadAttention(dim=dim, heads=heads, head_dim=head_dim, dropout=dropout)
        self.attn2 = MultiHeadAttention(dim=dim, heads=heads, head_dim=head_dim, dropout=dropout)
        self.ff = FF(chunks=chunks, dim=dim, ff_mult=ff_mult)
        # self.act = nn.ReLU()

    def forward(self, x, context):
        x = self.attn1(x)
        x = self.attn2(x, context)
        x = self.ff(x)
        # x = self.act(x)
        return x


class Encoder_2D(nn.Module):
    def __init__(self, dim=64, heads=8, head_dim=32, chunks=1, ff_mult=4, dropout=0.):
        super().__init__()
        self.vgg = VGG(cfg=[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.block1 = block_2d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block2 = block_2d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block3 = block_2d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block4 = block_2d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block5 = block_2d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block6 = block_2d(dim, heads, head_dim, chunks, ff_mult, dropout)
            
    def forward(self, x):
        x = self.vgg(x)
        x0 = x
        x = self.block1(x, x0)
        x = self.block2(x, x0)
        x = self.block3(x, x0)
        x = self.block4(x, x0)
        x = self.block5(x, x0)
        x = self.block6(x, x0)
        return x + x0


class Decoder_3D(nn.Module):
    def __init__(self, dim=64, heads=8, head_dim=64, chunks=1, ff_mult=4, dropout=0.):
        super().__init__()
        self.view_embeddings = nn.Parameter(torch.randn(1, 64, 512))
        self.pos_embeddings = nn.Parameter(torch.randn(64, 512))

        self.block1 = block_3d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block2 = block_3d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block3 = block_3d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block4 = block_3d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block5 = block_3d(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.block6 = block_3d(dim, heads, head_dim, chunks, ff_mult, dropout)

        # self.sig = nn.Sigmoid()
        
    def forward(self, context):
        bs = context.shape[0]
        x = self.view_embeddings.repeat([bs, 1, 1]) + self.pos_embeddings
        x = self.block1(x, context)
        x = self.block2(x, context)
        x = self.block3(x, context)
        x = self.block4(x, context)
        x = self.block5(x, context)
        x = self.block6(x, context)
        x = x.reshape(bs, 4, 4, 4, 8, 8, 8).permute(0, 1, 4, 2, 5, 3, 6)
        x = x.reshape(bs, 32, 32, 32)
        # return self.sig(x)
        return x


class VolT(nn.Module):
    def __init__(self, dim=512, heads=8, head_dim=32, chunks=1, ff_mult=4, dropout=0.):
        super().__init__()
        self.encoder = Encoder_2D(dim, heads, head_dim, chunks, ff_mult, dropout)
        self.decoder = Decoder_3D(dim, heads, head_dim, chunks, ff_mult, dropout)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    model = VolT().cuda()
    checkpoint = torch.load('/home/vision/diska2/1JYK/HQ/checkpoints/VolT/VolT_2021-04-19-14-07.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        val_img = read_img('/home/vision/diska2/1JYK/VQ/data/data_5000/image/04003.png').reshape(1, 1, 128, 128)
        val_vol = nn.AvgPool3d(4,4)(read_mha('/home/vision/diska2/1JYK/VQ/data/data_5000/volume/04002.mha').reshape(1, 1, 128, 128, 128))
        val_out = model(val_img)
        # val_out = binary(val_out, eps=0.5)
        # val_vol = binary(val_vol, eps=0.5)
        save_mha('/home/vision/diska2/1JYK/HQ/result/04003_std.mha', val_vol)
        save_mha('/home/vision/diska2/1JYK/HQ/result/04003_rec.mha', val_out)