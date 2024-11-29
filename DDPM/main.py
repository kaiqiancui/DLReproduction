#训练算法
import torch
import torch.nn as nn
from DL_Reproduction.DDPM.dataset import get_dataloader, get_img_shape
from DL_Reproduction.DDPM.ddpm import DDPM
import cv2
import numpy as np
import einops

batch_size = 512
n_epochs = 100

def train(ddpm: DDPM, net, device, ckpt_path):
    #n_steps 就是t
    #net是继承自torch.nn.Module 的神经网络
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss() #选择nn里面的MSE作为损失函数 #均方误差
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)
    
    for e in range(n_epochs):
        for x, _ in dataloader:
            pass