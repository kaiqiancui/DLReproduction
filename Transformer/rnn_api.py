import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 定义超参数
batch_size, num_steps = 32, 35
num_hiddens = 256
vocab_size = 10000  # 假设词汇表大小为10000
learning_rate = 0.01
num_epochs = 10

# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = nn.RNN(vocab_size, num_hiddens)
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.linear = nn.Linear(num_hiddens, vocab_size)

    def forward(self, inputs, state):
        # 将输入转换为one-hot编码
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        # 通过RNN层
        Y, state = self.rnn(X, state)
        # 通过全连接层
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        # 初始化隐状态
        return torch.zeros((1, batch_size, self.num_hiddens), device=device)

# 初始化模型
model = RNNModel(vocab_size, num_hiddens)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 生成随机训练数据
def generate_data(num_samples, num_steps, vocab_size):
    data = torch.randint(0, vocab_size, (num_samples, num_steps))
    labels = torch.roll(data, shifts=-1, dims=1)  # 标签是下一个时间步的输入
    return data, labels

# 生成训练数据
num_samples = 1000  # 假设有1000个样本
train_data, train_labels = generate_data(num_samples, num_steps, vocab_size)

# 创建DataLoader
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_data, batch_labels in train_loader:
        # 初始化隐状态
        state = model.begin_state(device='cpu', batch_size=batch_size)
        
        # 前向传播
        outputs, state = model(batch_data, state)
        
        # 计算损失
        loss = criterion(outputs, batch_labels.reshape(-1))
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 打印每个epoch的损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

print("训练完成！")