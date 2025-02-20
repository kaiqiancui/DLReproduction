import torch
import torch.nn as nn
import torch.functional as F
class FeedForward(nn.Module):
    def __init__(self, dim_vector, dim_hidden, dropout=0.1):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(dim_vector, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_vector)
        )
        
    def forward(self, x):
        out = self.feedforward(x)
        return out
    

# 专家模型
class Expert(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# 路由部分
# 这里我们假设定义n_embed为32， num_experts=4, top_k=2

class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k # 选择前 k 个专家
        self.linear =nn.Linear(n_embed, num_experts)
    def forward(self, mh_output):
        
        logits = self.linear(mh_output)
        # 选top_k个专家
        # 返回值和索引
        top_k_logits, indics = logits.topk(self.top_k, dim = -1)
        # 照着logits创建一个inf矩阵
        zeros = torch.full_like(logits, float('-inf'))
        # 在indices指定的位置填入top_k_logits的值
        sparse_logits = zeros.scatter(-1, indics, top_k_logits)
        # 计算softmax，只有top_k位置有非0概率
        router_output = F.softmax(sparse_logits, dim=-1)
        
        
# 为了避免专家垄断，增添负载均衡
# # 1. 新增噪声生成层
# self.noise_linear = nn.Linear(n_embed, num_experts)

# # 2. 生成噪声logits
# noise_logits = self.noise_linear(mh_output)

# # 3. 添加缩放的高斯噪声
# noise = torch.randn_like(logits) * F.softplus(noise_logits)
# noisy_logits = logits + noise
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear =nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices
    

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        # 初始化路由器
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        # 创建专家列表
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        # 选多少个专家
        self.top_k = top_k

    def forward(self, x):
        # 使用路由器得到两个输出
        # 1. 输入进入router得到两个输出
        gating_output, indices = self.router(x)
        
        # 2.初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros_like(x)

        # 3.展平，即把每个batch拼接到一起，这里对输入x和router后的结果都进行了展平
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # 以每个专家为单位进行操作，即把当前专家处理的所有token都进行加权
        for i, expert in enumerate(self.experts):
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            expert_mask = (indices == i).any(dim=-1)
            # 5. 展平操作
            flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if flat_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                expert_input = flat_x[flat_mask]
                # 7. 将token输入expert得到输出
                expert_output = expert(expert_input)

                # 8. 计算当前专家对于有作用的token的权重分数
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                # 9. 将expert输出乘上权重分数
                weighted_output = expert_output * gating_scores

                # 10. 循环进行做种的结果叠加
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


class Block(nn.Module):
    """Mixture of Experts Transformer block: communication followed by computation (multi-head self attention + SparseMoE) """
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x
