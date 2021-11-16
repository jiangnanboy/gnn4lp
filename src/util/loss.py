import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgae_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    '''
    变分图自编码，损失函数包括两部分：
        1.生成图和原始图之间的距离度量
        2.节点表示向量分布和正态分布的KL散度
    '''
    # 负样本边的weight都为1，正样本边的weight都为pos_weight
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def gae_loss_function(preds, labels, norm, pos_weight):
    '''
    图自编码，损失函数是生成图和原始图之间的距离度量
    '''
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return cost

def varga_loss_function(preds, labels, dis_real, dis_fake, mu, logvar, n_nodes, norm, pos_weight):
    # 对抗变分图正则化图自编码损失：生成和判别的loss
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    dis_real_loss = F.binary_cross_entropy_with_logits(dis_real, torch.ones(dis_real.shape).to(DEVICE))
    dis_fake_loss = F.binary_cross_entropy_with_logits(dis_fake, torch.zeros(dis_fake.shape).to(DEVICE))
    return cost +KLD + dis_real_loss + dis_fake_loss

def arga_loss_function(preds, labels, dis_real, dis_fake, norm, pos_weight):
    # 对抗图正则化图自编码损失：生成和判别的loss
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    dis_real_loss = F.binary_cross_entropy_with_logits(dis_real, torch.ones(dis_real.shape).to(DEVICE))
    dis_fake_loss = F.binary_cross_entropy_with_logits(dis_fake, torch.zeros(dis_fake.shape).to(DEVICE))
    return cost + dis_real_loss + dis_fake_loss

