import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchkeras
import tensorboardX
from tensorboardX import  SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


import numpy as np
from tqdm import tqdm
from sys import path
import time
import os
import PIL.Image as Image
import SimpleITK as itk
import imageio

path.append('/home/vision/diska2/1JYK/HQ/models/VolT/')
path.append('/home/vision/diska2/1JYK/HQ/')
from utils import data_process
import utils.scheduler as scheduler
import config
from VolT import VolT

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


def train(opt, checkpoint_path=None):
    opt.model_name = opt.model_name + '_' + time.strftime('%Y-%m-%d-%H-%M', time.localtime())

    batch_size = opt.batch_size
    full_dataset = data_process.Rec_Dataset(img_path='/home/vision/diska2/1JYK/VQ/data/data_5000/image/', vol_path='/home/vision/diska2/1JYK/VQ/data/data_5000/volume/', cuda_device=opt.device)
    # full_dataset = data_process.Img_Dataset(img_path='/home/vision/diska2/1JYK/VQ/data/data_5000/image/', cuda_device=opt.device)
    # full_dataset = data_process.Vol_Dataset(vol_path='/home/vision/diska2/1JYK/VQ/data/data_5000/volume/', cuda_device=opt.device)
    train_sampler, valid_sampler, test_sampler = data_process.data_split(full_dataset, split_ratio=[0.8, 0.1, 0.1])
    train_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, sampler=valid_sampler)
    # test_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, sampler=test_sampler)

    current_step = 0
    epoch_current = 0
    best_valid_loss = 1000

    model = VolT().cuda(opt.device)
    device = next(model.parameters()).device
    # for name, parameters in model.named_parameters():
    #     print(name, parameters.shape)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, eps=1e-8)

    # lr_sch = StepLR(optimizer, 20, gamma=0.5)
    # sch_warmup = scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=lr_sch)
    
    lr_sch = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    sch_warmup = scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=lr_sch)

    optimizer.step()
    sch_warmup.step()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model, optimizer, epoch_current, loss = load_checkpoint(model, optimizer, checkpoint_path)
        print("Pre-trained models loaded.")
    
    for epoch in range(epoch_current, opt.epochs):
        print(optimizer.param_groups[0]['lr'])
        sch_warmup.step()

    writer = SummaryWriter('runs/VolT/' + opt.model_name)
    config.save_config(opt, '/home/vision/diska2/1JYK/HQ/checkpoints/VolT/' + opt.model_name + '.json')

    for epoch in range(epoch_current, opt.epochs):
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        total_train_loss = 0
        
        for data in tqdm(train_loader):
            model.zero_grad()
            
            img, vol = data
            out = model(img)
            loss = F.mse_loss(vol, out)
            total_train_loss += loss.item()
            
            current_step += 1
            writer.add_scalar('train_step_loss', loss.item(), global_step=current_step)

            loss.backward()
            optimizer.step()

            # for name, p in model.named_parameters():
            #     print(name, torch.norm(p.grad, float('inf')))

            bi_out = binary(out, eps=0.5)
            acc = torch.sum(bi_out == vol).item() / (vol.numel())
            print(loss.item(), torch.sum(bi_out == vol).item() / (vol.numel()) * 100)
            # print(loss.item())
            writer.add_scalar('train_step_acc', acc, global_step=current_step)
        
        sch_warmup.step()
        print("Epoch[%d] train loss: %.5lf" %(epoch, total_train_loss / len(train_loader)))
        writer.add_scalar('train_epoch_loss', total_train_loss / len(train_loader), global_step=epoch)

        save_checkpoint(model, optimizer, epoch, loss.item(), '/home/vision/diska2/1JYK/HQ/checkpoints/VolT/' + opt.model_name + '.tar')

        with torch.no_grad():
            val_img = read_img('/home/vision/diska2/1JYK/VQ/data/data_5000/image/04001.png').reshape(1, 1, 128, 128)
            val_vol = nn.AvgPool3d(4,4)(read_mha('/home/vision/diska2/1JYK/VQ/data/data_5000/volume/04001.mha').reshape(1, 1, 128, 128, 128))
            val_out = model(val_img)
            val_out = binary(val_out, eps=0.5)
            val_vol = binary(val_vol, eps=0.5)
            save_mha('/home/vision/diska2/1JYK/HQ/result/04001_std.mha', val_vol)
            save_mha('/home/vision/diska2/1JYK/HQ/result/04001_rec.mha', val_out)

            total_valid_loss = 0
            for data in tqdm(valid_loader):
                img, vol = data
                out = model(img)
                loss = F.mse_loss(vol, out)
                total_valid_loss += loss.item()
                print(loss.item())
            writer.add_scalar('valid_epoch_loss', total_valid_loss / len(valid_loader), global_step=epoch)
            
            if total_valid_loss / len(valid_loader) < best_valid_loss:
                best_valid_loss = total_valid_loss / len(valid_loader)
                save_model(model, '/home/vision/diska2/1JYK/HQ/checkpoints/VolT/' + opt.model_name + '.pth')


if __name__ == '__main__':
    parser = config.gen_config()
    train(parser)
