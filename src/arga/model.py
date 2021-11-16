import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class GCNModelARGA(nn.Module):
    # The Adversarially Regularized Graph Auto-Encoder model
    # 对抗正则化图自编码，利用gae/vgae作为生成器；一个三层前馈网络作判别器
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, vae_bool=True):
        super(GCNModelARGA, self).__init__()
        self.vae_bool = vae_bool
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout)
        self.ip = InnerProductDecoder(dropout)
        self.relu = nn.ReLU()
        self.discriminator = Discriminator(hidden_dim2, hidden_dim1)

    def encode(self, input, adj):
        hidden1 = self.relu(self.gc1(input, adj))
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
        z_fake = self.reparameterize(mu, logvar) #得到z_fake
        z_real = torch.randn(z_fake.shape).to(DEVICE) # 得到高斯分布的z_real
        # 判别器判断真假
        dis_real = self.discriminator(z_real)
        dis_fake = self.discriminator(z_fake)
        return self.ip(z_fake), dis_real, dis_fake, mu, logvar # ip(z_fake)生成邻接矩阵

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

class Discriminator(nn.Module):
    # 判别器
    def __init__(self, hidden_dim2, hidden_dim1):
        super(Discriminator, self).__init__()

        self.fc = nn.Sequential(nn.Linear(hidden_dim2, hidden_dim1),
                                nn.ReLU(),
                                nn.Linear(hidden_dim1, hidden_dim2),
                                nn.ReLU(),
                                nn.Linear(hidden_dim2, 1))

    def forward(self, z):
        return self.fc(z)

