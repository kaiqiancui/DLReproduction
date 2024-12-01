import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


n_epochs = 3 #循环整个训练集的次数
batch_size_train = 64 
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                            ])),
    batch_size=batch_size_test, shuffle=True)
#测试数据
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_targets)
print(example_data.shape)

#构建网络
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() #调用父类的构造函数 保证正确的初始化
        #第一个卷积层：1通道输入，10通道输出，卷积核大小五乘五
        #灰度图像一般一开始只有一个通道
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #不同的输出通道可以检测不同的学习特征
        #第二个卷积层：输入通道对应第一个卷积层的输出通道
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #drop_out 在训练过程中随机丢弃一些神经元
        #相当于一个正则化技术 默认概率 = 0.5
        self.conv2_drop = nn.Dropout2d()
        #全联接层 输入特征32 输出特征50
        self.fc1 = nn.Linear(320, 50)
        #全联接层
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        #前向传播过程，x是输入数据
        # 卷积-池化-relu
        #2表示第二次的最大池化
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 第二次卷积，加个drop out
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #展平
        #假设输入张量形状为 (N, C, H, W)：
        # N: batch size（批次大小）
        # C: channels（通道数）
        # H: height（高度）
        # W: width（宽度）
        # 展平后变为 (N, C*H*W)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        #training=self.training确保只在训练时应用dropout
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

network = Net()
optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum=momentum)