# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class Contrastive_Loss(nn.Module):

    def __init__(self, tau=1, project=False, in_dim_1=None, in_dim_2=None, out_dim=None) -> None:
        super().__init__()
        self.tau = tau
        self.project = project

        if project:
            if not in_dim_1:
                return ValueError
            self.x_projector = nn.Linear(in_dim_1, out_dim)
            self.y_projector = nn.Linear(in_dim_2, out_dim)


    def forward(self, X, Y):
        
        if self.project:
            X = self.x_projector(X)
            Y = self.y_projector(Y)

        loss = self.compute_cl(X, Y) + self.compute_cl(Y, X)

        return loss
    

    def compute_cl(self, X, Y):

        '''
        X: (bs, hidden_size), Y: (bs, hidden_size)
        tau: the temperature factor
        '''
        #sim_matrix = X.mm(Y.t())    # (bs, bs)
        sim_matrix = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2)
        pos = torch.exp(torch.diag(sim_matrix) / self.tau).unsqueeze(0)   # (1, bs)
        neg = torch.sum(torch.exp(sim_matrix / self.tau), dim=0) - pos     # (1, bs)
        #TODO: 这里的这个pos到底用不用减去
        loss = - torch.log(pos / neg)
        loss = loss.view(X.shape[0], -1)

        return loss
    


class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    


class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention



class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self,x,y,log_seqs):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output



class Attention(nn.Module):

    def __init__(self, hidden_size, method="dot"):

        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == "dot":
            pass
        elif self.method == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size,bias=False)


    def forward(self, query, key):
        """
        query: [bs, hidden_size]
        key: [bs, seq_len, hidden_size]
        weight: [bs, seq_len, 1]
        """

        if self.method == "dot":
            return self.dot_score(query, key)
        elif self.method == "general":
            return self.general_score(query, key)


    def dot_score(self, query, key):
        
        query = query.unsqueeze(2)  #[bs, hidden_size, 1]
        attn_energies = torch.bmm(key, query) # (bs, seq_len, hidden_size) * (bs, hidden_size, 1) --> (bs, seq_len, 1)
        attn_energies = attn_energies.squeeze(-1) # (bs, seq_len)

        return F.softmax(attn_energies, dim=-1).unsqueeze(-1)  # [batch_size, seq_len, 1]
    

    def general_score(self, query, key):

        query = self.Wa(query).unsqueeze(2) # (bs, hidden_size, 1)
        attn_energies = torch.bmm(key, query).squeeze(-1) 
        
        return F.softmax(attn_energies,dim=-1).unsqueeze(-1)
    

def reg_params(model):
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss


def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return torch.sum(F.softplus(neg_preds - pos_preds))



class SpAdjEdgeDrop(nn.Module):

    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__()

    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        newVals = vals[mask]# / keep_rate
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)




