#神经网络实现
import torch
import torch.nn as nn
from DLReproduction.DDPM.dataset import get_dataloader, get_img_shape
from DLReproduction.DDPM.ddpm import DDPM
import cv2
import numpy as np
import einops

#位置编码
class PositionalEncoding(nn.Module):
    #整个借鉴了transformer
    def __init__(self, max_seq_len:int, d_model:int):
        #max_seq_len : 时间步的最大长度
        #d_model: 编码的维度 必须是偶数
        super().__init__()
        assert d_model % 2 == 0
        
        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len -1, max_seq_len) #时间步序列
        j_seq = torch.linspace(0, d_model - 2, d_model //2) #维度序列
        pos, two_i = torch.meshgrid(i_seq, j_seq) #生成网格
        #偶数位置编码
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))      # sin部分
        #奇数位置编码，值在[-1, 1]
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))    # cos部分
        #组合编码
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)
        
        #创建嵌入层 ： 嵌入层本质是查找表
        self.embedding = nn.Embedding(max_seq_len, d_model)  # 创建嵌入层
        self.embedding.weight.data = pe                      # 设置权重
        self.embedding.requires_grad_(False)                 # 冻结参数
    
    def forword(self, t):
        return self.embedding(t)


class ResidualBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)  # 3x3卷积，步长1，填充1
        self.bn1 = nn.BatchNorm2d(out_c)  # 批量归一化
        self.actvation1 = nn.ReLU()  # ReLU激活函数
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.actvation2 = nn.ReLU()
        
        # 捷径连接（shortcut）
        if in_c != out_c:
            # 当输入输出通道数不同时，使用1x1卷积调整
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),  # 1x1卷积
                nn.BatchNorm2d(out_c)
            )
        else:
            # 通道数相同时，直接使用恒等映射
            self.shortcut = nn.Identity()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.actvation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(input)  # 添加残差连接
        x = self.actvation2(x)  # 最后再做激活
        return x


class ConvNet(nn.Module):
    def __init__(self,
                n_steps,  # 时间步数
                intermediate_channels=[10, 20, 40],  # 中间层通道数
                pe_dim=10,  # 位置编码维度
                insert_t_to_all_layers=False):  # 是否在所有层插入时间信息
        super().__init__()
        C, H, W = get_img_shape()  # 获取输入图像形状（1, 28, 28）
        
        # 位置编码层
        self.pe = PositionalEncoding(n_steps, pe_dim)
        
        # 存储时间编码的线性层
        self.pe_linears = nn.ModuleList()
        self.all_t = insert_t_to_all_layers
        
        # 如果不在所有层插入时间信息，只在第一层插入
        if not insert_t_to_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim, C))
        
        # 构建残差块序列
        self.residual_blocks = nn.ModuleList()
        prev_channel = C
        for channel in intermediate_channels:
            # 添加残差块
            self.residual_blocks.append(ResidualBlock(prev_channel, channel))
            
            # 根据设置决定是否添加时间编码层
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))
            else:
                self.pe_linears.append(None)
            prev_channel = channel
            
        # 输出层
        self.output_layer = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]  # 批次大小
        t = self.pe(t)  # 计算时间位置编码
        
        # 遍历所有残差块和对应的时间编码层
        for m_x, m_t in zip(self.residual_blocks, self.pe_linears):
            if m_t is not None:
                # 如果有时间编码层，将时间信息加入特征图
                pe = m_t(t).reshape(n, -1, 1, 1)
                x = x + pe
            x = m_x(x)  # 通过残差块
            
        x = self.output_layer(x)  # 最后通过输出层
        return x


class UnetBlock(nn.Module):
    #基本构件块
    def __init__(self, shape, in_c, out_c, residual= False):
        #shape 输入特征图的形状
        #in_channel and out_channnel
        # residual：是否使用残差链接
        self.ln = nn.LayerNorm(shape) #层归一化
        #3*3的卷积，stride = 1步长1,padding=1保持特征图的大小不变
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1) #第一个卷积
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        
        if residual:
            if in_c == out_c:
                # 输入输出通道数相同，直接连接
                self.residual_conv = nn.Identity()
            else:
                # 通道数不同，用1x1卷积调整，方便结果一致
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)
                
    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:
            out += self.residual_conv(x)
        
        out = self.activation(out)
        return out

class UNet(nn.Module):
    
    """
    这是一个典型的UNet结构,主要特点是:
    1. 使用下采样-上采样的对称结构
    2. 包含skip connection
    3. 加入了位置编码来处理时序信息
    4. 可选的残差连接
    """
    def __init__(self,
                 n_steps,
                 channels=[10, 20, 40, 80],  # 各层的通道数列表
                 pe_dim=10,                  # 位置编码的维度
                 residual=False) -> None:    # 是否使用残差连接
        super().__init__()
        
        # 获取输入图像的通道数(C)、高度(H)和宽度(W)
        C, H, W = get_img_shape()
        layers = len(channels)
        
        # 存储每一层的特征图尺寸,每下采样一次尺寸减半
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers - 1):
            cH //= 2
            cW //= 2
            Hs.append(cH)
            Ws.append(cW)

        # 初始化位置编码层
        self.pe = PositionalEncoding(n_steps, pe_dim)

        # 创建各个组件的ModuleList
        self.encoders = nn.ModuleList()      # 编码器层列表
        self.decoders = nn.ModuleList()      # 解码器层列表
        self.pe_linears_en = nn.ModuleList() # 编码器的位置编码线性层
        self.pe_linears_de = nn.ModuleList() # 解码器的位置编码线性层
        self.downs = nn.ModuleList()         # 下采样层列表
        self.ups = nn.ModuleList()           # 上采样层列表

        # 构建编码器部分
        prev_channel = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            # 添加位置编码的线性变换层,包含两个线性层和ReLU激活
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))
            
            # 添加编码器块,每个编码器包含两个UnetBlock
            # 第一个Block改变通道数,第二个Block保持通道数不变
            self.encoders.append(
                nn.Sequential(
                    UnetBlock((prev_channel, cH, cW),
                              prev_channel,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))
            
            # 添加下采样层(步长为2的卷积)
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        # 构建网络的中间层
        self.pe_mid = nn.Linear(pe_dim, prev_channel)  # 位置编码的线性变换
        channel = channels[-1]
        # 两个UnetBlock的序列
        self.mid = nn.Sequential(
            UnetBlock((prev_channel, Hs[-1], Ws[-1]),
                      prev_channel,
                      channel,
                      residual=residual),
            UnetBlock((channel, Hs[-1], Ws[-1]),
                      channel,
                      channel,
                      residual=residual),
        )

        # 构建解码器部分
        prev_channel = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            # 添加位置编码线性层
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            # 添加转置卷积上采样层
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))
            
            # 添加解码器块
            # 输入通道数是channel*2(因为有skip connection)
            self.decoders.append(
                nn.Sequential(
                    UnetBlock((channel * 2, cH, cW),
                              channel * 2,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))

            prev_channel = channel

        # 最后添加输出卷积层,将通道数变回原始图像的通道数
        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        # 前向传播接收:
        # x: 输入图像 
        # t: 时间步
        n = t.shape[0]
        t = self.pe(t)  # 对时间进行位置编码
        
        # 编码器前向传播
        encoder_outs = []
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders, self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)  # 计算位置编码并reshape
            x = encoder(x + pe)                      # 将位置编码加到特征上并通过编码器处理
            encoder_outs.append(x)                   # 保存特征用于skip connection
            x = down(x)                              # 下采样

        # 中间层处理
        pe = self.pe_mid(t).reshape(n, -1, 1, 1)
        x = self.mid(x + pe)

        # 解码器前向传播
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1, 1)  # 计算位置编码
            x = up(x)                               # 上采样特征

            # 对特征进行padding以匹配编码器的输出尺寸
            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = nn.functional.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))
            
            x = torch.cat((encoder_out, x), dim=1)  # 拼接编码器的特征(skip connection)
            x = decoder(x + pe)                     # 加入位置编码并通过解码器处理

        x = self.conv_out(x)  # 通过输出卷积层
        return x

# 小型ConvNet配置
convnet_small_cfg = {
    'type': 'ConvNet',                    # 网络类型
    'intermediate_channels': [10, 20],     # 中间层通道数
    'pe_dim': 128                         # 位置编码维度
}

# 中型ConvNet配置
convnet_medium_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],  # 更多的层和通道
    'pe_dim': 256,                        # 更大的位置编码维度
    'insert_t_to_all_layers': True        # 在所有层都插入时间信息
}

# 大型ConvNet配置
convnet_big_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160],  # 更大的通道数
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}

# 基础UNet配置
unet_1_cfg = {
    'type': 'UNet', 
    'channels': [10, 20, 40, 80],         # UNet各层的通道数
    'pe_dim': 128
}

# 带残差连接的UNet配置
unet_res_cfg = {
    'type': 'UNet',
    'channels': [10, 20, 40, 80],
    'pe_dim': 128,
    'residual': True                      # 启用残差连接
}

def build_network(config: dict, n_steps):
    """
    根据配置构建网络
    参数:
        config: 网络配置字典
        n_steps: 时间步数
    返回:
        构建好的网络实例
    """
    network_type = config.pop('type')      # 获取并移除type字段
    if network_type == 'ConvNet':
        network_cls = ConvNet
    elif network_type == 'UNet':
        network_cls = UNet

    network = network_cls(n_steps, **config)  # 创建网络实例
    return network
