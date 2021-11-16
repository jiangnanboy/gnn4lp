import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = Parameter(torch.FloatTensor(2 * output_dim, 1))
        self.reset_parameters() # 初始化
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.elu = nn.ELU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_normal_(self.a)

    def forward(self, input, adj):
        # 节点特征矩阵inputs: (N, input_dim), 邻接矩阵adj: (N, N)
        Wh = torch.mm(input, self.weight) # (N, output_dim) = (N, input_dim) * (input_dim, output_dim)

        # 图注意力系数矩阵，得到图中所有结点对之间的注意力系数
        Wh1 = torch.matmul(Wh, self.a[:self.output_dim, :]) # (N, 1) = (N, output_dim) * (out_dim, 1)
        Wh2 = torch.matmul(Wh, self.a[self.output_dim:, :]) # (N, 1) = (N, output_dim) * (out_dim, 1)
        e = Wh1 + Wh2.T # (N, N)
        e = self.leakyrelu(e)
        #注意力系数可能为0，这里需要进行筛选操作，便于后续mask
        zero_vec = -9e15 * torch.ones_like(e)
        # torch.where(condition, x, y)返回从x,y中选择元素所组成的tensor。如果满足条件，则返回x中元素。若不满足，返回y中元素。
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        # 结合注意力系数
        attention_wh = torch.matmul(attention, Wh) # (N, output_dim) = (N, N) * (N, output_dim)

        return self.elu(attention_wh)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        input = self.dropout(input)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

class NHGATModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, hidden_dim3, num_heads, dropout, alpha, vae_bool=True):
        super(NHGATModelVAE, self).__init__()
        self.vae_bool = vae_bool
        # 多头self_attention
        self.n_heads_attentions = nn.ModuleList(
            [GraphAttentionLayer(input_feat_dim, hidden_dim1, dropout=dropout, alpha=alpha) for _ in
             range(num_heads)])
        # 输出
        self.gc_att = GraphAttentionLayer(num_heads * hidden_dim1, hidden_dim1, dropout=dropout, alpha=alpha)

        self.gc1 = GraphConvolution(hidden_dim1, hidden_dim2, dropout)
        self.gc2 = GraphConvolution(hidden_dim2, hidden_dim3, dropout)
        self.gc3 = GraphConvolution(hidden_dim2, hidden_dim3, dropout)
        self.ip = InnerProductDecoder(dropout)
        self.relu = nn.ReLU()

    def encode(self, input, adj):
        nheads_att_out = torch.cat([attention(input, adj.to_dense()) for attention in self.n_heads_attentions], dim=-1)
        gc_att = self.gc_att(nheads_att_out, adj.to_dense())
        hidden1 = self.relu(self.gc1(gc_att, adj))
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.vae_bool:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)  # 乘std加mu
        else:
            return mu

    def forward(self, input, adj):
        mu, logvar = self.encode(input, adj) #两个GCN分别得到mean和std
        z = self.reparameterize(mu, logvar) #得到z
        return self.ip(z), mu, logvar

class InnerProductDecoder(nn.Module):
    '''
    内积用来做decoder，用来生成邻接矩阵
    '''
    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        z = self.dropout(z)
        adj = torch.mm(z, z.t())
        return adj
