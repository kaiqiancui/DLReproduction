import math
import torch
from torch import nn
from torch.nn import functional as F
import torch
from torch.utils.data import Dataset, DataLoader

class TinyShakespeareDataset(Dataset):
    # 处理莎士比亚数据集
    def __init__(self, file_path, seq_length):
        # seq_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        # 提取文中的唯一字符，排序
        self.chars = sorted(list(set(self.text)))
        # 创建一个字典，每个字符唯一的索引
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        # 创建字典 字符 -> 索引映射
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in self.text]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # 从data当中提取一个seq_length的目标序列 
        # y是x的下一个字符用来预测
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x), torch.tensor(y)

def load_data_machine(batch_size, num_steps):
    # 下载或加载 Tiny Shakespeare 数据集
    file_path = "tinyshakespeare.txt"  # 确保文件存在
    dataset = TinyShakespeareDataset(file_path, num_steps)
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab = dataset.char_to_idx
    return train_iter, vocab

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    # 正态分布初始化函数：0.01保证生成小随机数
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))  # 输入到隐藏的权重
    W_hh = normal((num_hiddens, num_hiddens))  # 隐藏到隐藏的权重
    b_h = torch.zeros(num_hiddens, device=device)  # 隐藏层偏置

    W_hq = normal((num_hiddens, num_outputs))  # 隐藏到输出的权重
    b_q = torch.zeros(num_outputs, device=device)  # 输出层偏置

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化 RNN 隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

# RNN 前向传播
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# RNN 模型类
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # one_hot 编码
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# 推理函数
def predict(prefix, num_preds, net, vocab, device):
    """prefix (str): 初始字符序列
       num_preds (int): 要预测的字符数量"""
    # 初始化网络状态
    state = net.begin_state(batch_size=1, device=device)

    # 将第一个字符转换为索引并存入输出列表
    outputs = [vocab[prefix[0]]]

    # 定义输入获取函数
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # 预热期：处理前缀剩余字符
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # 预测阶段：生成新字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    # 将索引序列转换为字符
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 梯度裁剪
def grad_clipping(net, theta):
    """
    net (nn.Module or dict): 神经网络模型或参数字典
    theta (float): 梯度裁剪阈值
    """
    # 获取需要梯度的参数
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    # 计算梯度的L2范数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    # 如果范数超过阈值，按比例缩小梯度
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

# 训练函数
def train_epoch(net, train_iter, loss, optimizer, device, use_random_iter):
    """训练网络一个迭代周期
       use_random_iter: 指示是否在每次迭代时随机初始化隐藏状态
    """
    state = None
    metric = [0.0, 0]  # 训练损失之和, 词元数量
    # 初始化state
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.size(0), device=device)
        else:
            # 如果不需要初始化，就使用前面的state
            # 使用detach_分离计算图，防止梯度传导
            if isinstance(state, torch.Tensor):
                state.detach_()
            elif isinstance(state, tuple):
                for s in state:
                    s.detach_()

        X, Y = X.to(device), Y.to(device)
        # 标签重塑为一维向量
        y = Y.T.reshape(-1)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        optimizer.zero_grad()
        l.backward()
        grad_clipping(net, 1)  # 梯度裁剪
        optimizer.step()

        # 计算困惑度
        metric[0] += l.item() * y.numel()
        metric[1] += y.numel()
    return math.exp(metric[0] / metric[1]), metric[1]

# 主程序
if __name__ == "__main__":
    # 读取数据
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_machine(batch_size, num_steps)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_hiddens = 256
    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)

    # 定义损失函数和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.params, lr=1.0)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(net, train_iter, loss, optimizer, device, use_random_iter=False)
        print(f"Epoch {epoch + 1}: Perplexity {ppl:.1f}, Speed {speed:.1f} tokens/sec")
