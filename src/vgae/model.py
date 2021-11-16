import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

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

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, vae_bool=True):
        super(GCNModelVAE, self).__init__()
        self.vae_bool = vae_bool
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout)
        self.ip = InnerProductDecoder(dropout)
        self.relu = nn.ReLU()

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
