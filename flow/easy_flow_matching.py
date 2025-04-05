# from 直观理解Flow Matching算法（带源码） - 张云聪的文章 - 知乎
# https://zhuanlan.zhihu.com/p/28731517852
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 超参数
dim = 2         # 数据维度（2D点）
num_samples = 1000
num_steps = 50  # ODE求解步数
lr = 1e-3
epochs = 5000


# 目标分布：正弦曲线上的点（x1坐标）
x1_samples = torch.rand(num_samples, 1) * 4 * torch.pi  # 0到4π
y1_samples = torch.sin(x1_samples)                      # y=sin(x)
target_data = torch.cat([x1_samples, y1_samples], dim=1)

# 噪声分布：高斯噪声（x0坐标）
noise_data = torch.randn(num_samples, dim) * 2

class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # x:[batch_size, dim = 2]
            # t:[batchsize, 1] 每个样本一个时间点
            # 从 2D点+时间维度 = 3维度 -> 64维度
            nn.Linear(dim + 1, 64),
            nn.ReLU(),
            # 最终输出的维度是2维的坐标
            nn.Linear(64, dim)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim =1))

model = VectorField()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

for epoch in range(epochs):
    # 相当于batch size = 1000
    idx = torch.randperm(num_samples)
    x0 = noise_data[idx]
    x1 = target_data[idx]
    # 生成1000个范围在0-1里的随机时间点，学习在任意时间点的去噪
    t = torch.rand(x0.size(0), 1)
    
    # 线性插值生成中间点
    xt = (1 - t) * x0 + t * x1
    
    # 模型预测的向量场
    vt_pred = model(xt, t)

    # 目标向量场
    vt_target = x1 - x0
    
    # 损失函数
    loss = torch.mean((vt_pred - vt_target)**2)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 到此为止我们已经学习到了从噪声中转换到目标分布的路径
# 现在可以随机一个噪声，然后找到变化路径

# 从噪声数据中取第一个样本
x = noise_data[0:1]

# 创建一个轨迹列表，初始化为x的值
trajectory = [x.detach().numpy()]

# 创建时间标签t=1（最终时间点）
tag = torch.tensor([1])  # 或 torch.ones(1, 1)

t = 0 # 初始时间点
delta_t = 1 / num_steps # 分为步骤来生成

with torch.no_grad():
    for i in range(num_steps):
        vt = model(x, torch.tensor([[t]], dtype=torch.float32))
        t += delta_t
        x = x + vt * delta_t  # x(t+Δt) = x(t) + v(t)Δt
        trajectory.append(x.detach().numpy())

trajectory = torch.tensor(trajectory).squeeze()


print(trajectory[-1] / (torch.pi / 10 * 4))

# 绘制向量场和生成轨迹
plt.figure(figsize=(10, 5))
plt.scatter(target_data[:,0], target_data[:,1], c='blue', label='Target (sin(x))')
plt.scatter(noise_data[:,0], noise_data[:,1], c='red', alpha=0.3, label='Noise')
plt.plot(trajectory[:,0], trajectory[:,1], 'g-', linewidth=2, label='Generated Path')
plt.legend()
plt.title("Flow Matching: From Noise to Target Distribution")
plt.show()