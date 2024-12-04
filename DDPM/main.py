#训练算法
import torch
import torch.nn as nn
from DLReproduction.DDPM.dataset import get_dataloader, get_img_shape
from DLReproduction.DDPM.ddpm import DDPM
from DLReproduction.DDPM.network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)
import cv2
import numpy as np
import einops

batch_size = 512
n_epochs = 100

def train(ddpm: DDPM, net, device, ckpt_path):
    #n_steps 就是t
    #net是继承自torch.nn.Module 的神经网络
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)#随机对x进行采样
    net = net.to(device)
    loss_fn = nn.MSELoss() #选择nn里面的MSE作为损失函数 #均方误差
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)
    
    for e in range(n_epochs):
        for x, _ in dataloader:
            current_bach_size = x.shape[0]
            x = x.to(device)
            #随机产生t
            t = torch.randint(0, n_steps, (current_bach_size, )).to(device)
            #随机产生x_t(随机噪声)
            eps = torch.rand_like(x).to(device)
            #前向传播过程，直接计算，将会输入进预测过程
            x_t = ddpm.sample_forward(x, t, eps)
            #将x_t输入进预测过程，计算我们要预测的误差
            eps_theta = net(x_t, t.reshape(current_bach_size, 1))
            #计算预测噪声和实际噪声的均方误差
            loss = loss_fn(eps_theta, eps)
            #梯度是累积的，每次反向传播之前都要清零
            #清除之前的梯度
            optimizer.zero_grad()
            #计算梯度
            loss.backword()
            optimizer.step()
    torch.save(net.state_dict(), ckpt_path)

def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]


if __name__ == '__main__':
    n_steps = 1000
    config_id = 4
    device = 'cuda'
    model_path = 'DLReproduction/DDPM/model_unet_res.pth'

    config = unet_res_cfg
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device=device, ckpt_path=model_path)
    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, 'work_dirs/diffusion.jpg', device=device)