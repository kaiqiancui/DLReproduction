import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor
mnist = torchvision.datasets.MNIST(root = './data/mnist', download=True)

def get_dataloader(batch_size: int):
    #Lambda 是 transform 里面的一个转换类， 把一个普通的python函数包装为转换器对象
    #batch_size=batch_size：指定每个 batch 包含的样本数量。
    #shuffle=True：在每个 epoch 开始时随机打乱数据顺序。
    transform = transform.Compose([ToTensor(), Lambda(lambda x:(x - 0.5)*2)])
    dataset = torchvision.datasets.MNIST(root = './data/mnist', transform=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle = True)
