import torch 
class DDPM():
    #n-step = (paper)T
    def __init__(self,
                device,
                n_steps: int,#(T)
                min_beta: float = 0.0001,
                max_beta: float = 0.02
                ):
        #线性生成beta值,(T,1)
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        
        #计算论文中的a_bar, 是一个连乘运算
        #对于每一个时间，都有一个阿尔法bar
        #alpha_bars[i] = product of (α_1, α_2, ..., α_i)
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        #存到类里面方便使用    
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
    
    #前向过程 从图片生成噪声的过程
    def sample_forward(self, x, t, eps = None):
        #eps：一个可选参数，表示随机噪声。如果没有提供，方法会自动生成一个与 x 形状相同的随机噪声。
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        #通常 x 是一个四维张量，形状为 [batch_size, channels, height, width]
        #-1表示自动匹配， 1表示直接匹配
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self, img_shape, net, device, simple_var = True):
        #simple_var = True表示使用简单方差
        #还有一种更复杂的方差，可以用来改进，赋值false
        x = torch.randn(img_shape).to(device) #x_t, 采样开始
        net = net.to(device)
        for t in range(self.n_steps-1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x
        
    def sample_backward_step(self, x_t, t, net, simple_var = True):
        #n = batch_size
        n = x_t.shape[0]
        #每个批次都需要自己的时间步，我们需要把时间步复制n份，然后转化时间
        #to的这个设备是x_t的设备，这样可以保证所有的运算都在同一个设备
        #unsqueeze表示在指定位置添加一个维度
        ## x_t 的形状通常是 [batch_size, channels, height, width]
        # 例如对于一批图像可能是 [n, 3, 64, 64]
        # t_tensor 在 unsqueeze 之前的形状是 [batch_size]
        
        
    #x_{t-1} = mean + noise    
        t_tensor = torch.tensor([t]*n, dtype= torch.long).to(x_t.device).unsqueeze(1)
        #在网络上预测噪声
        eps = net(x_t, t_tensor)
        #对随机噪声乘标准差得到最终的噪声
    #这一步是计算noise, 
        if t == 0:
            noise = 0
        else:
            if simple_var:
                #简单方差 var = beta_t
                var = self.betas[t]
            else:
                #复杂方差
                var = var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)
    #计算mean
    #括号内的内容会被Python解释器视为一个完整的表达式，直到找到对应的闭合括号
        mean = (x_t - 
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps
                ) / torch.sqrt(self.alphas[t])
        x_t = mean + noise
        
        return x_t