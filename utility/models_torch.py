# -*- coding: utf-8 -*-
# @Time : 2022/4/9 8:44 下午
# @Author : crisis
# @FileName: models_torch.py
# @Function：
# @Usage:

import torch
import torch.nn as nn
"""
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
"""

# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''
#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature  # 即sqrt(d_k)
#         self.dropout = nn.Dropout(attn_dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, q, k, v):
#         """
#         :param q: [N, n_head, len_q, d_k]
#         :param k: [N, n_head, len_k, d_k]
#         :param v: [N, n_head, len_v, d_v]
#         :return: output [N, n_head, len_q, d_v]  attn [N, n_head, len_q, len_k]
#         """
#         # [N, n_head, len_q, d_k] * [N, n_head, d_k, len_k] -> [N, n_head, len_q, len_k]
#         attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
#         attn = self.dropout(self.softmax(attn))
#         # [N, n_head, len_q, len_k] * [N, n_head, len_v, d_v] -> [N, n_head, len_q, d_v]
#         output = torch.matmul(attn, v)
#
#         return output, attn


# class MultiHeadAttention(nn.Module):
#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()
#
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
#
#         self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
#         self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
#
#         self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
#
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)
#
#     def forward(self, q, k, v):
#         """
#         输入[N, edge_type_cnt, dim]，希望模拟edge_type之间的相关性，输出[N, edge_type_cnt, dim]
#         :param q: [N, len_q, d_model]
#         :param k: [N, len_k, d_model]
#         :param v: [N, len_v, d_model]
#         len_q=len_k=len_v, 即q=k=v=x，传入参数全是[N, edge_type_cnt, dim]
#         :return: [N, len_q, d_model]
#         """
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#         sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
#
#         residual = q  # [N, len_q, d_model]
#
#
#         # 1. 由输入计算q.k.v。理解为计算每个edge_type下的q，k，v
#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # x * Wq -> q
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)  # x * Wk -> k
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)  # x * Wv -> v
#
#         # 调整维度，将n_head移到前面，方便之后算scaled dot attention
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [N, n_head, len_q, d_k]
#
#         # 2. 计算(q*k)/sqrt(d_k) * v
#         q, attn = self.attention(q, k, v)  # q. [N, n_head, len_q, d_v], attn. [N, n_head, len_q, len_k]
#
#         # 调整维度，将n_head移到embedding size的维度上，方便之后进行转换
#         q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # [N, len_q, n_head * d_v]
#
#         # 3. 特征变换。经过一个全连接层得到最终embedding。
#         q = self.dropout(self.fc(q))  # [N, len_q, d_model]
#         q += residual
#
#         q = self.layer_norm(q)
#         return q, attn

# 重写tf版本

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature  # 即sqrt(d_k)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        :param q: [N, len_q, d_k]
        :param k: [N, len_k, d_k]
        :param v: [N, len_v, d_v]
        :return: output [N, len_q, d_v]  attn [N, len_q, len_k]
        """
        # [N, len_q, d_k] * [N, d_k, len_k] -> [N, len_q, len_k]
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # attn = self.dropout(self.softmax(attn))
        # [N, n_head, len_q, len_k] * [N, n_head, len_v, d_v] -> [N, n_head, len_q, d_v]
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_model, bias=True)
        self.w_ks = nn.Linear(d_model, d_model, bias=True)
        self.w_vs = nn.Linear(d_model, d_model, bias=True)

        # for debug 常数初始化
        # self.w_qs = nn.Linear(d_model, d_model, bias=False)
        # self.w_ks = nn.Linear(d_model, d_model, bias=False)
        # self.w_vs = nn.Linear(d_model, d_model, bias=False)
        # nn.init.constant_(self.w_qs.weight, 1)
        # nn.init.constant_(self.w_ks.weight, 1)
        # nn.init.constant_(self.w_vs.weight, 1)

        # self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)

    def forward(self, q, k, v):
        """
        输入[N, edge_type_cnt, dim]，希望模拟edge_type之间的相关性，输出[N, edge_type_cnt, dim]
        :param q: [N, len_q, d_model]
        :param k: [N, len_k, d_model]
        :param v: [N, len_v, d_model]
        len_q=len_k=len_v, 即q=k=v=x，传入参数全是[N, edge_type_cnt, dim]
        :return: [N, len_q, d_model]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        d_model = q.size(2)
        # 1. 由输入计算q.k.v。理解为计算每个edge_type下的q，k，v
        q = self.w_qs(q)  # x * Wq -> q
        k = self.w_ks(k)  # x * Wk -> k
        v = self.w_vs(v)  # x * Wv -> v
        # return q, k, v

        q_ = torch.cat(torch.chunk(q, n_head, dim=2), dim=0)  # (h*N, len_q, d_model/h)
        k_ = torch.cat(torch.chunk(k, n_head, dim=2), dim=0)  # (h*N, len_k, d_model/h)
        v_ = torch.cat(torch.chunk(v, n_head, dim=2), dim=0)  # (h*N, len_k, d_model/h)
        # return q_, k_, v_

        # Attention
        outputs, attn = self.attention(q_, k_, v_)  # [h*N, len_q, d_model/h]
        # return outputs, attn

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, n_head, dim=0), dim=2)  # (N, len_q, d_model)
        attn = torch.cat(torch.chunk(attn, n_head, dim=0), dim=1)  # [N, nh, len_q, len_k]

        # Residual connection
        outputs += q

        # return outputs
        # Normalize
        outputs = self.layer_norm(outputs)

        return outputs, attn

import random
import numpy as np
import os
if __name__ == '__main__':
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.

        torch.set_deterministic(True)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(2020)
    multi = MultiHeadAttention(n_head=1, d_model=4, d_k=4, d_v=4)
    scale = ScaledDotProductAttention(temperature=4 ** 0.5, attn_dropout=0.1)
    # input = torch.ones([5, 3, 4])
    input = torch.FloatTensor([
        [[1, 2, 3, 4], [2, 3, 4, 5]],
        [[2, 3, 4, 5], [3, 4, 5, 6]]
    ])
    # q, k, v = multi(input, input, input)
    # print(q)
    # print(k)
    # print(v)

    output = multi(input, input, input)
    print(output)
