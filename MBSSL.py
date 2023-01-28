# -*- coding: utf-8 -*-
# @Time : 2023/1/3 20:33
# @Author : crisis
# @FileName: MBSSL_unified.py
# @Function：用来看和mbssl_unified在beibei上为什么无法复现结果
# @Usage:
# CUDA_VISIBLE_DEVICES=0 python MBSSL_unified.py --dataset Taobao--wid [0.01,0.01,0.01] --coefficient [1.0/6,4.0/6,1.0/6] --decay 0.01 --ssl_temp 0.2 --ssl_reg_inter [0.5,0.5]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import os
import sys
import pandas as pd
import copy
import scipy.sparse as sp
from time import time
from utility.models_torch import *
from utility.helper import *
from utility.batch_test import *
import pickle
import multiprocessing
import torch.multiprocessing
import random
from utility.metabalance import MetaBalance, MetaBalance2, MetaBalance3
import fitlog


class Augmentor():
    def __init__(self, data_config, args):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.train_matrix = data_config['trn_mat']  # 目标行为交互矩阵 dok matrix形式
        self.training_user, self.training_item = self.get_train_interactions()  # 基于目标行为交互矩阵 获取交互边的两端点 user_list item_list
        self.ssl_ratio = args.ssl_ratio
        self.aug_type = args.aug_type

    def get_train_interactions(self):
        """
        返回user list和item list
        :return:
        """
        # dok_matrix = self.train_matrix.todok()  # 返回csr matrix的dok matrix形式，是字典形式的矩阵存储。key为(row,col)，value为对应位置的值
        users_list, items_list = [], []
        for (user, item), value in self.train_matrix.items():
            users_list.append(user)
            items_list.append(item)
        # (users_np, items_np) = self.train_matrix.nonzero()

        return users_list, items_list

    def augment_adj_mat(self, aug_type=0):
        np.seterr(divide='ignore')
        n_nodes = self.n_users + self.n_items
        if aug_type in [0, 1, 2] and self.ssl_ratio > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                drop_user_idx = np.random.choice(self.n_users, size=int(self.n_users * self.ssl_ratio),
                                                 replace=False)  # 选取随机丢弃的user节点 []
                drop_item_idx = np.random.choice(self.n_items, size=int(self.n_items * self.ssl_ratio),
                                                 replace=False)  # 选取随机丢弃的item节点 []
                indicator_user = np.ones(self.n_users, dtype=np.float32)
                indicator_item = np.ones(self.n_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)  # [n_user, n_user]
                diag_indicator_item = sp.diags(indicator_item)  # [n_item, n_item]
                R = sp.csr_matrix(
                    (np.ones_like(self.training_user, dtype=np.float32), (self.training_user, self.training_item)),
                    shape=(self.n_users, self.n_items))  # 原始的rating matrix. csr_matrix (n_user, n_item)
                R_prime = diag_indicator_user.dot(R).dot(
                    diag_indicator_item)  # 开始dropout [n_user, n_user] * [n_user, n_item] * [n_item, n_item]  -> [n_user, n_item]
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.n_users)),
                                        shape=(n_nodes, n_nodes))
            if aug_type in [1, 2]:
                keep_idx = np.random.choice(len(self.training_user),
                                            size=int(len(self.training_user) * (1 - self.ssl_ratio)),
                                            replace=False)  # 随机选择保存下来的边idx
                user_np = np.array(self.training_user)[keep_idx]
                item_np = np.array(self.training_item)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_users)), shape=(n_nodes, n_nodes))

        adj_mat = tmp_adj + tmp_adj.T  # 变成对称矩阵，无向图。

        # pre adjcency matrix  D^-0.5 * A * D^-0.5， D为度矩阵，A为邻接矩阵
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)  # 对称归一化的度矩阵
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        # print(adj_matrix.tocsr())
        return adj_matrix.tocsr()


class GHCF(nn.Module):
    name = 'GHCF'

    def __init__(self, max_item_list, data_config, args):
        super(GHCF, self).__init__()
        # ********************** input data *********************** #
        self.max_item_list = max_item_list
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.num_nodes = self.n_users + self.n_items
        self.pre_adjs = data_config['pre_adjs']
        self.pre_adjs_tensor = [self._convert_sp_mat_to_sp_tensor(adj).to(device) for adj in self.pre_adjs]
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        # ********************** hyper parameters *********************** #
        self.coefficient = torch.tensor(eval(args.coefficient)).view(1, -1).to(device)
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.mess_dropout = eval(args.mess_dropout)  # dropout ratio
        self.aug_type = args.aug_type
        self.nhead = args.nhead
        self.att_dim = args.att_dim
        # ********************** learnable parameters *********************** #
        self.all_weights = {}
        self.all_weights['user_embedding'] = Parameter(torch.FloatTensor(self.n_users, self.emb_dim))
        self.all_weights['item_embedding'] = Parameter(torch.FloatTensor(self.n_items, self.emb_dim))
        self.all_weights['relation_embedding'] = Parameter(torch.FloatTensor(self.n_relations, self.emb_dim))

        self.weight_size_list = [self.emb_dim] + self.weight_size  # [64,64,64,64] 用于定义权重W的shape

        for k in range(self.n_layers):
            self.all_weights['W_gc_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))
            self.all_weights['W_rel_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))

        # self.all_weights['trans_weights'] = Parameter(torch.FloatTensor(self.n_relations, self.emb_dim, self.emb_dim))
        self.all_weights['trans_weights_s1'] = Parameter(
            torch.FloatTensor(self.n_relations, self.emb_dim, self.att_dim))
        self.all_weights['trans_weights_s2'] = Parameter(torch.FloatTensor(self.n_relations, self.att_dim, 1))
        self.reset_parameters()
        self.all_weights = nn.ParameterDict(self.all_weights)
        # self.dropout = nn.Dropout(1 - self.mess_dropout[0])
        self.dropout = nn.Dropout(self.mess_dropout[0], inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        # self.s_att = MultiHeadAttention(n_head=self.nhead, d_model=self.emb_dim, d_k=self.emb_dim, d_v=self.emb_dim, dropout=self.mess_dropout[0])
        # self.s_att = MultiHeadAttention(n_head=1, d_model=self.emb_dim, d_k=self.emb_dim, d_v=self.emb_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.all_weights['user_embedding'])
        nn.init.xavier_uniform_(self.all_weights['item_embedding'])
        nn.init.xavier_uniform_(self.all_weights['relation_embedding'])
        # nn.init.xavier_uniform_(self.all_weights['trans_weights'])
        nn.init.xavier_uniform_(self.all_weights['trans_weights_s1'])
        nn.init.xavier_uniform_(self.all_weights['trans_weights_s2'])
        for k in range(self.n_layers):
            nn.init.xavier_uniform_(self.all_weights['W_gc_%d' % k])
            nn.init.xavier_uniform_(self.all_weights['W_rel_%d' % k])

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape
        return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape))

    # def forward(self, sub_mats, device):
    #     """
    #     :param sub_mats: dict
    #     aug_type=0/1 {'sub1':sparseTensor, 'sub2':sparseTensor}
    #     aug_type=2 {'sub11':sparseTensor, 'sub12':sparseTensor, 'sub21':sparseTensor, 'sub22':sparseTensor}
    #     :return:
    #     """
    #     self.sub_mat = {}
    #     # 1. 为每一层gcn都指定两个增强图，其中只有random walk每一层的图不一样，否则每一层的两个图都一样
    #     for k in range(1, self.n_layers + 1):
    #         # 1.1 Node/Edge dropout，每一层的图都一样。 保存每一层两张图的邻接矩阵到sub_mat中
    #         if self.aug_type in [0, 1]:
    #             self.sub_mat['sub_mat_1%d' % k] = sub_mats['sub1'].to(device)
    #             self.sub_mat['sub_mat_2%d' % k] = sub_mats['sub2'].to(device)
    #         # 1.2 random walk。保存每一层两张图的邻接矩阵到sub_mat中
    #         else:
    #             self.sub_mat['sub_mat_1%d' % k] = sub_mats['sub1%d' % k].to(device)
    #             self.sub_mat['sub_mat_2%d' % k] = sub_mats['sub2%d' % k].to(device)
    #
    #     all_rela_embs = {}  # {'pv':[r0, r1], 'buy':[r0, r1], 'train':[r0, r1]}
    #     for i in range(self.n_relations):
    #         beh = self.behs[i]
    #         rela_emb = self.all_weights['relation_embedding'][i]
    #         rela_emb = torch.reshape(rela_emb, (-1, self.emb_dim))
    #         all_rela_embs[beh] = [rela_emb]
    #
    #     ego_embeddings = torch.cat((self.all_weights['user_embedding'], self.all_weights['item_embedding']), dim=0).unsqueeze(1).repeat(1, self.n_relations, 1)  # [N, 3, dim]
    #     all_embeddings = ego_embeddings
    #
    #     total_mm_time = 0.
    #
    #     # 1. 先将所有层的rela embedding存下来
    #     for k in range(0, self.n_layers):
    #         for i in range(self.n_relations):
    #             rela_emb = torch.matmul(all_rela_embs[self.behs[i]][k],
    #                                     self.all_weights['W_rel_%d' % k])  # 当前层的rela embedding 经过变换得到
    #             all_rela_embs[self.behs[i]].append(rela_emb)
    #
    #     # 1. 原图
    #     for k in range(0, self.n_layers):
    #         embeddings_list = []  # 所有行为在当前层的节点 embedding
    #         for i in range(self.n_relations):
    #             st = time()
    #             embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings[:,  i, :])
    #             total_mm_time += time() - st
    #             rela_emb = all_rela_embs[self.behs[i]][k]  # 取当前层，即第k层的relation embedding
    #             embeddings_ = self.leaky_relu(
    #                 torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gc_%d' % k]))
    #             embeddings_list.append(embeddings_)
    #         embeddings_st = torch.stack(embeddings_list, dim=1)  # [N, 3, dim]
    #         ego_embeddings, attn = self.s_att(embeddings_st, embeddings_st, embeddings_st)  # [N, 3, dim]
    #         all_embeddings = all_embeddings + ego_embeddings
    #     all_embeddings /= self.n_layers + 1
    #     u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
    #     token_embedding = torch.zeros([1, self.n_relations, self.emb_dim], device=device)
    #     i_g_embeddings = torch.cat((i_g_embeddings, token_embedding), dim=0)
    #
    #     # 2. sub1
    #     ego_embeddings = torch.cat((self.all_weights['user_embedding'], self.all_weights['item_embedding']), dim=0).unsqueeze(1).repeat(1, self.n_relations, 1)  # [N, 3, dim]
    #     all_embeddings = ego_embeddings
    #     for k in range(0, self.n_layers):
    #         embeddings_list = []  # 所有行为在当前层的节点 embedding
    #         for i in range(self.n_relations):
    #             st = time()
    #             if i != self.n_relations - 1:
    #                 embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings[:,  i, :])
    #             else:
    #                 embeddings_ = torch.matmul(self.sub_mat['sub_mat_1%d' % (k + 1)],
    #                                            ego_embeddings[:,  i, :])
    #             total_mm_time += time() - st
    #             rela_emb = all_rela_embs[self.behs[i]][k]  # 取当前层，即第k层的relation embedding
    #             embeddings_ = self.leaky_relu(
    #                 torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gc_%d' % k]))
    #             embeddings_list.append(embeddings_)
    #         embeddings_st = torch.stack(embeddings_list, dim=1)  # [N, 3, dim]
    #         ego_embeddings, attn = self.s_att(embeddings_st, embeddings_st, embeddings_st)  # [N, 3, dim]
    #         all_embeddings = all_embeddings + ego_embeddings
    #     # print('total mm time: %.5fs' % (total_mm_time))  # 0.3s， 整个model 0.6s
    #     all_embeddings /= self.n_layers + 1
    #     u_g_embeddings_sub1, i_g_embeddings_sub1 = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
    #     i_g_embeddings_sub1 = torch.cat((i_g_embeddings_sub1, token_embedding), dim=0)
    #
    #     # 3. sub2
    #     ego_embeddings = torch.cat((self.all_weights['user_embedding'], self.all_weights['item_embedding']), dim=0).unsqueeze(1).repeat(1, self.n_relations, 1)  # [N, 3, dim]
    #     all_embeddings = ego_embeddings
    #     for k in range(0, self.n_layers):
    #         embeddings_list = []  # 所有行为在当前层的节点 embedding
    #         for i in range(self.n_relations):
    #             st = time()
    #             if i != self.n_relations - 1:
    #                 embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings[:,  i, :])
    #             else:
    #                 embeddings_ = torch.matmul(self.sub_mat['sub_mat_2%d' % (k + 1)],
    #                                            ego_embeddings[:,  i, :])
    #             total_mm_time += time() - st
    #             rela_emb = all_rela_embs[self.behs[i]][k]  # 取当前层，即第k层的relation embedding
    #             embeddings_ = self.leaky_relu(
    #                 torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gc_%d' % k]))
    #             embeddings_list.append(embeddings_)
    #         embeddings_st = torch.stack(embeddings_list, dim=1)  # [N, 3, dim]
    #         ego_embeddings, attn = self.s_att(embeddings_st, embeddings_st, embeddings_st)  # [N, 3, dim]
    #         all_embeddings = all_embeddings + ego_embeddings
    #     # print('total mm time: %.5fs' % (total_mm_time))  # 0.3s， 整个model 0.6s
    #     all_embeddings /= self.n_layers + 1
    #     u_g_embeddings_sub2, i_g_embeddings_sub2 = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
    #     i_g_embeddings_sub2 = torch.cat((i_g_embeddings_sub1, token_embedding), dim=0)
    #
    #     for i in range(self.n_relations):
    #         all_rela_embs[self.behs[i]] = torch.mean(torch.stack(all_rela_embs[self.behs[i]], 0), 0)
    #
    #     return u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2, all_rela_embs

    def forward(self, sub_mats, device):
        """
        :param sub_mats: dict
        aug_type=0/1 {'sub1':sparseTensor, 'sub2':sparseTensor}
        aug_type=2 {'sub11':sparseTensor, 'sub12':sparseTensor, 'sub21':sparseTensor, 'sub22':sparseTensor}
        :return:
        """

        self.sub_mat = {}
        # 1. 为每一层gcn都指定两个增强图，其中只有random walk每一层的图不一样，否则每一层的两个图都一样
        for k in range(1, self.n_layers + 1):
            # 1.1 Node/Edge dropout，每一层的图都一样。 保存每一层两张图的邻接矩阵到sub_mat中
            if self.aug_type in [0, 1]:
                self.sub_mat['sub_mat_1%d' % k] = sub_mats['sub1'].to(device)
                self.sub_mat['sub_mat_2%d' % k] = sub_mats['sub2'].to(device)
            # 1.2 random walk。保存每一层两张图的邻接矩阵到sub_mat中
            else:
                self.sub_mat['sub_mat_1%d' % k] = sub_mats['sub1%d' % k].to(device)
                self.sub_mat['sub_mat_2%d' % k] = sub_mats['sub2%d' % k].to(device)

        ego_embeddings = torch.cat((self.all_weights['user_embedding'], self.all_weights['item_embedding']),
                                   dim=0).unsqueeze(1).repeat(1, self.n_relations, 1)  # [N, 3, dim]
        ego_embeddings_sub1 = ego_embeddings
        ego_embeddings_sub2 = ego_embeddings

        all_embeddings = ego_embeddings
        all_embeddings_sub1 = ego_embeddings_sub1
        all_embeddings_sub2 = ego_embeddings_sub2

        all_rela_embs = {}  # {'pv':[r0, r1], 'buy':[r0, r1], 'train':[r0, r1]}
        for i in range(self.n_relations):
            beh = self.behs[i]
            rela_emb = self.all_weights['relation_embedding'][i]
            rela_emb = torch.reshape(rela_emb, (-1, self.emb_dim))
            all_rela_embs[beh] = [rela_emb]

        total_mm_time = 0.
        for k in range(0, self.n_layers):
            # 1. 原图
            embeddings_list = []  # 所有行为在当前层的节点 embedding
            for i in range(self.n_relations):
                st = time()
                embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings[:, i, :])
                total_mm_time += time() - st
                rela_emb = all_rela_embs[self.behs[i]][k]  # 取当前层，即第k层的relation embedding
                embeddings_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gc_%d' % k]))
                embeddings_list.append(embeddings_)
            embeddings_st = torch.stack(embeddings_list, dim=1)  # [N, 3, dim]
            """gatne self-att """
            embeddings_list = []  # 所有行为 attention后的emb
            attention_list = []
            for i in range(self.n_relations):
                attention = F.softmax(
                    torch.matmul(
                        torch.tanh(torch.matmul(embeddings_st, self.all_weights['trans_weights_s1'][i])),
                        self.all_weights['trans_weights_s2'][i]
                    ).squeeze(2),
                    dim=1
                ).unsqueeze(1)  # [N, 1, 3]
                attention_list.append(attention)
                embs_cur_rela = torch.matmul(attention, embeddings_st).squeeze(1)  # [N, 1, dim]
                # 忘掉了trans_w
                # embs_cur_rela = torch.matmul(torch.matmul(attention, embeddings_st),
                #                              self.all_weights['trans_weights'][i]).squeeze(1)  # [N, 1, dim]
                embeddings_list.append(embs_cur_rela)
            ego_embeddings = torch.stack(embeddings_list, dim=1)
            attn = torch.cat(attention_list, dim=1)  # [N, 3, 3]

            """ trm self-att """
            # ego_embeddings, attn = self.s_att(embeddings_st, embeddings_st, embeddings_st)  # [N, 3, dim]

            # dropout1
            # new_embeddings_st_dp_list = [self.dropout(ego_embeddings[:, i, :]) for i in
            #                              range(3)]  # [em1, em2, em3]
            # ego_embeddings = torch.stack(new_embeddings_st_dp_list, dim=1)  # [N, 3, dim]

            # dropout2
            ego_embeddings = self.dropout(ego_embeddings)
            all_embeddings = all_embeddings + ego_embeddings

            # 2. sub1
            embeddings_list_sub1 = []  # 所有行为在当前层的节点 embedding
            for i in range(self.n_relations):
                rela_emb = all_rela_embs[self.behs[i]][k]  # 取当前层，即第k层的relation embedding
                st = time()
                if i != self.n_relations - 1:
                    embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings_sub1[:, i, :])
                else:
                    embeddings_ = torch.matmul(self.sub_mat['sub_mat_1%d' % (k + 1)],
                                               ego_embeddings_sub1[:, i, :])
                total_mm_time += time() - st
                embeddings_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gc_%d' % k]))
                embeddings_list_sub1.append(embeddings_)

            embeddings_st = torch.stack(embeddings_list_sub1, dim=1)  # [N, 3, dim]

            """gatne self-att """
            embeddings_list_sub1 = []  # 所有行为 attention后的emb
            # attention_list_sub1 = []
            for i in range(self.n_relations):
                attention = F.softmax(
                    torch.matmul(
                        torch.tanh(torch.matmul(embeddings_st, self.all_weights['trans_weights_s1'][i])),
                        self.all_weights['trans_weights_s2'][i]
                    ).squeeze(2),
                    dim=1
                ).unsqueeze(1)  # [N, 1, 3]
                # attention_list_sub1.append(attention)
                embs_cur_rela = torch.matmul(attention, embeddings_st).squeeze(1)  # [N, 1, dim]
                # 忘掉了trans_w
                # embs_cur_rela = torch.matmul(torch.matmul(attention, embeddings_st),
                #                              self.all_weights['trans_weights'][i]).squeeze(1)  # [N, 1, dim]
                embeddings_list_sub1.append(embs_cur_rela)
            ego_embeddings_sub1 = torch.stack(embeddings_list_sub1, dim=1)
            # attn_sub1 = torch.cat(attention_list_sub1, dim=1)

            """ trm self-att """
            # ego_embeddings_sub1, attn_sub1 = self.s_att(embeddings_st, embeddings_st, embeddings_st)  # [N, 3, dim]
            # dropout1
            # new_embeddings_st_dp_list = [self.dropout(ego_embeddings_sub1[:, i, :]) for i in
            #                              range(3)]  # [em1, em2, em3]
            # ego_embeddings_sub1 = torch.stack(new_embeddings_st_dp_list, dim=1)  # [N, 3, dim]
            # dropout2
            ego_embeddings_sub1 = self.dropout(ego_embeddings_sub1)
            all_embeddings_sub1 = all_embeddings_sub1 + ego_embeddings_sub1

            # 3. sub2
            embeddings_list_sub2 = []  # 所有行为在当前层的节点 embedding
            for i in range(self.n_relations):
                rela_emb = all_rela_embs[self.behs[i]][k]  # 取当前层，即第k层的relation embedding
                st = time()
                if i != self.n_relations - 1:
                    embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings_sub2[:, i, :])
                else:
                    embeddings_ = torch.matmul(self.sub_mat['sub_mat_2%d' % (k + 1)],
                                               ego_embeddings_sub2[:, i, :])
                total_mm_time += time() - st
                embeddings_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gc_%d' % k]))
                embeddings_list_sub2.append(embeddings_)

            embeddings_st = torch.stack(embeddings_list_sub2, dim=1)  # [N, 3, dim]
            embeddings_list_sub2 = []  # 所有行为 attention后的emb
            # attention_list_sub2 = []
            for i in range(self.n_relations):
                attention = F.softmax(
                    torch.matmul(
                        torch.tanh(torch.matmul(embeddings_st, self.all_weights['trans_weights_s1'][i])),
                        self.all_weights['trans_weights_s2'][i]
                    ).squeeze(2),
                    dim=1
                ).unsqueeze(1)  # [N, 1, 3]
                # attention_list_sub2.append(attention)
                embs_cur_rela = torch.matmul(attention, embeddings_st).squeeze(1)  # [N, 1, dim]
                # 忘掉了trans_w
                # embs_cur_rela = torch.matmul(torch.matmul(attention, embeddings_st),
                #                              self.all_weights['trans_weights'][i]).squeeze(1)  # [N, 1, dim]
                embeddings_list_sub2.append(embs_cur_rela)
            ego_embeddings_sub2 = torch.stack(embeddings_list_sub2, dim=1)
            # attn_sub2 = torch.cat(attention_list_sub2, dim=1)

            """ trm self-att """
            # ego_embeddings_sub2, attn_sub2 = self.s_att(embeddings_st, embeddings_st, embeddings_st)  # [N, 3, dim]
            # dropout1
            # new_embeddings_st_dp_list = [self.dropout(ego_embeddings_sub2[:, i, :]) for i in
            #                              range(3)]  # [em1, em2, em3]
            # ego_embeddings_sub2 = torch.stack(new_embeddings_st_dp_list, dim=1)  # [N, 3, dim]
            # dropout2
            ego_embeddings_sub2 = self.dropout(ego_embeddings_sub2)
            all_embeddings_sub2 = all_embeddings_sub2 + ego_embeddings_sub2

            # 更新下一层的relation embedding
            for i in range(self.n_relations):
                rela_emb = torch.matmul(all_rela_embs[self.behs[i]][k],
                                        self.all_weights['W_rel_%d' % k])  # 当前层的rela embedding 经过变换得到
                all_rela_embs[self.behs[i]].append(rela_emb)

        # print('total mm time: %.5fs' % (total_mm_time))  # 0.3s， 整个model 0.6s
        all_embeddings /= self.n_layers + 1
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        token_embedding = torch.zeros([1, self.n_relations, self.emb_dim], device=device)
        i_g_embeddings = torch.cat((i_g_embeddings, token_embedding), dim=0)

        all_embeddings_sub1 /= self.n_layers + 1
        u_g_embeddings_sub1, i_g_embeddings_sub1 = torch.split(all_embeddings_sub1, [self.n_users, self.n_items], 0)
        i_g_embeddings_sub1 = torch.cat((i_g_embeddings_sub1, token_embedding), dim=0)

        all_embeddings_sub2 /= self.n_layers + 1
        u_g_embeddings_sub2, i_g_embeddings_sub2 = torch.split(all_embeddings_sub2, [self.n_users, self.n_items], 0)
        i_g_embeddings_sub2 = torch.cat((i_g_embeddings_sub2, token_embedding), dim=0)

        attn_user, attn_item = torch.split(attn, [self.n_users, self.n_items], 0)

        for i in range(self.n_relations):
            all_rela_embs[self.behs[i]] = torch.mean(torch.stack(all_rela_embs[self.behs[i]], 0), 0)

        return u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2, all_rela_embs, attn_user, attn_item


class RecLoss(nn.Module):
    def __init__(self, data_config, args):
        super(RecLoss, self).__init__()
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_dim = args.embed_size
        self.coefficient = eval(args.coefficient)
        self.wid = eval(args.wid)

    def forward(self, input_u, label_phs, ua_embeddings, ia_embeddings, rela_embeddings):
        """
        :param input_u:
        :param label_phs:
        :param ua_embeddings: [N, 3, dim]
        :param ia_embeddings: [N, 3, dim]
        :param rela_embeddings: [3, dim]
        :return:
        """
        uid = ua_embeddings[input_u]
        uid = torch.reshape(uid, (-1, self.n_relations, self.emb_dim))
        pos_r_list = []
        for i in range(self.n_relations):
            beh = self.behs[i]
            # 取出所有item的embedding，注意[B, max_item, dim]中item的id可能为num_item+1，但是由于ia_emebdding最后有个token embedding，所以无所谓
            pos_beh = ia_embeddings[:, i, :][label_phs[i]]  # [B, max_item, dim]
            # [B, max_item]中看每个用户的真实正样本（id为num_items的都为假的）, 得到[B, max_item]每个值是float32 0或1，代表该item是否为真实正样本
            pos_num_beh = torch.ne(label_phs[i], self.n_items).float()
            pos_beh = torch.einsum('ab,abc->abc', pos_num_beh,
                                   pos_beh)  # [B, max_item] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ac,abc->abc', uid[:, i, :],
                                 pos_beh)  # [B, dim] * [B, max_item, dim] -> [B, max_item, dim]
            pos_r = torch.einsum('ajk,lk->aj', pos_r, rela_embeddings[beh])
            pos_r_list.append(pos_r)

        loss = 0.
        # recommendation loss, 用各自embedding算各自行为上的loss
        for i in range(self.n_relations):
            beh = self.behs[i]
            temp = torch.einsum('ab,ac->bc', ia_embeddings[:, i, :], ia_embeddings[:, i, :]) \
                   * torch.einsum('ab,ac->bc', uid[:, i, :], uid[:, i, :])  # [B, dim]' * [B, dim] -> [dim, dim]
            tmp_loss = self.wid[i] * torch.sum(
                temp * torch.matmul(rela_embeddings[beh].T, rela_embeddings[beh]))
            tmp_loss += torch.sum((1.0 - self.wid[i]) * torch.square(pos_r_list[i]) - 2.0 * pos_r_list[i])

            loss += self.coefficient[i] * tmp_loss

        regularizer = torch.sum(torch.square(uid)) * 0.5 + torch.sum(torch.square(ia_embeddings)) * 0.5
        emb_loss = args.decay * regularizer

        return loss, emb_loss


# class SSLoss(nn.Module):
#     def __init__(self, data_config, args):
#         super(SSLoss, self).__init__()
#         self.ssl_temp = args.ssl_temp
#         self.ssl_reg = args.ssl_reg
#         self.ssl_mode = args.ssl_mode
#
#     def forward(self, input_u_list, input_i_list, ua_embeddings_sub1, ua_embeddings_sub2, ia_embeddings_sub1,
#                 ia_embeddings_sub2):
#         if self.ssl_mode in ['user_side', 'both_side']:
#             user_emb1 = ua_embeddings_sub1[input_u_list]
#             user_emb2 = ua_embeddings_sub2[input_u_list]  # [B, dim]
#
#             normalize_user_emb1 = F.normalize(user_emb1, dim=1)
#             normalize_user_emb2 = F.normalize(user_emb2, dim=1)
#             normalize_all_user_emb2 = F.normalize(ua_embeddings_sub2, dim=1)  # 分母为所有user  # [N, dim]
#             pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2),
#                                        dim=1)  # 同一个用户在两个视图上的embedding看作是正样本对  [B,]
#             ttl_score_user = torch.matmul(normalize_user_emb1,
#                                           normalize_all_user_emb2.T)  # 该用户在视图1的embedding和视图2上所有其他节点的embedding都看作是负样本对 [B, N]
#
#             pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
#             ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)  # [B, ]
#
#             ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))
#
#         if self.ssl_mode in ['item_side', 'both_side']:
#             item_emb1 = ia_embeddings_sub1[input_i_list]
#             item_emb2 = ia_embeddings_sub2[input_i_list]
#
#             normalize_item_emb1 = F.normalize(item_emb1, dim=1)
#             normalize_item_emb2 = F.normalize(item_emb2, dim=1)
#             normalize_all_item_emb2 = F.normalize(ia_embeddings_sub2, dim=1)
#             pos_score_item = torch.sum(torch.mul(normalize_item_emb1, normalize_item_emb2), dim=1)
#             ttl_score_item = torch.matmul(normalize_item_emb1, normalize_all_item_emb2.T)
#
#             pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
#             ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)
#
#             ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))
#
#         if self.ssl_mode == 'user_side':
#             ssl_loss = self.ssl_reg * ssl_loss_user
#         elif self.ssl_mode == 'item_side':
#             ssl_loss = self.ssl_reg * ssl_loss_item
#         else:
#             ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)
#
#         return ssl_loss


class SSLoss(nn.Module):
    def __init__(self, data_config, args):
        super(SSLoss, self).__init__()
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.ssl_mode = args.ssl_mode
        self.hard_neg = args.hard_neg_ssl1
        self.topk1 = 1000  # 目标是取top500-1000的总共500
        self.topk2 = 500

    def forward(self, input_u_list, input_i_list, ua_embeddings_sub1, ua_embeddings_sub2, ia_embeddings_sub1,
                ia_embeddings_sub2):
        if self.ssl_mode in ['user_side', 'both_side']:
            user_emb1 = ua_embeddings_sub1[input_u_list]
            user_emb2 = ua_embeddings_sub2[input_u_list]  # [B, dim]
            normalize_user_emb1 = F.normalize(user_emb1, dim=1)
            normalize_user_emb2 = F.normalize(user_emb2, dim=1)
            normalize_all_user_emb2 = F.normalize(ua_embeddings_sub2, dim=1)  # 分母为所有user  # [N, dim]
            pos_score_user = torch.sum(torch.mul(normalize_user_emb1, normalize_user_emb2),
                                       dim=1)  # 同一个用户在两个视图上的embedding看作是正样本对  [B,]
            pos_score_user = torch.exp(pos_score_user / self.ssl_temp)

            ttl_score_user = torch.matmul(normalize_user_emb1,
                                          normalize_all_user_emb2.T)  # 该用户在视图1的embedding和视图2上所有其他节点的embedding都看作是负样本对 [B, N]
            # 对于每个用户，选取top500-1000的用户作为负样本。
            # 实现时，即将[B,N]的矩阵每一行除了top500-1000以外的值置为0。
            if self.hard_neg:
                values1 = torch.topk(ttl_score_user, self.topk1, dim=1, largest=True)[0]
                values2 = torch.topk(ttl_score_user, self.topk1 - self.topk2, dim=1, largest=True)[0]
                indices_to_remove1 = ttl_score_user < values1[..., -1, None]
                indices_to_remove2 = ttl_score_user >= values2[..., 0, None]
                ttl_score_user[indices_to_remove1] = 0.
                ttl_score_user[indices_to_remove2] = 0.
            ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)  # [B, ]

            ssl_loss_user = -torch.sum(torch.log(pos_score_user / ttl_score_user))

        if self.ssl_mode in ['item_side', 'both_side']:
            """ 去掉token emb """
            # ia_embeddings_sub1 = ia_embeddings_sub1[:-1, :]
            # ia_embeddings_sub2 = ia_embeddings_sub2[:-1, :]
            item_emb1 = ia_embeddings_sub1[input_i_list]
            item_emb2 = ia_embeddings_sub2[input_i_list]

            normalize_item_emb1 = F.normalize(item_emb1, dim=1)
            normalize_item_emb2 = F.normalize(item_emb2, dim=1)
            normalize_all_item_emb2 = F.normalize(ia_embeddings_sub2, dim=1)
            pos_score_item = torch.sum(torch.mul(normalize_item_emb1, normalize_item_emb2), dim=1)
            ttl_score_item = torch.matmul(normalize_item_emb1, normalize_all_item_emb2.T)

            if self.hard_neg:
                values1 = torch.topk(ttl_score_item, self.topk1, dim=1, largest=True)[0]
                values2 = torch.topk(ttl_score_item, self.topk1 - self.topk2, dim=1, largest=True)[0]
                indices_to_remove1 = ttl_score_item < values1[..., -1, None]
                indices_to_remove2 = ttl_score_item >= values2[..., 0, None]
                ttl_score_item[indices_to_remove1] = 0.
                ttl_score_item[indices_to_remove2] = 0.

            pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
            ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)

            ssl_loss_item = -torch.sum(torch.log(pos_score_item / ttl_score_item))

        if self.ssl_mode == 'user_side':
            ssl_loss = self.ssl_reg * ssl_loss_user
        elif self.ssl_mode == 'item_side':
            ssl_loss = self.ssl_reg * ssl_loss_item
        else:
            ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)

        return ssl_loss


# class SSLoss2(nn.Module):
#     def __init__(self, data_config, args):
#         super(SSLoss2, self).__init__()
#         self.ssl_temp = args.ssl_temp
#         self.ssl_reg_inter = eval(args.ssl_reg_inter)  # [0.5, 0.5]
#         self.ssl_mode_inter = args.ssl_inter_mode
#         # self.aux_beh_idx = eval(args.aux_beh_idx)  # [0, 1]
#
#     def forward(self, input_u_list, input_i_list, ua_embeddings, ia_embeddings, aux_beh):
#         ssl2_loss = 0.
#         if self.ssl_mode_inter in ['user_side', 'both_side']:
#             emb_tgt = ua_embeddings[input_u_list, -1, :]
#             normalize_emb_tgt = F.normalize(emb_tgt, dim=1)
#             emb_aux = ua_embeddings[input_u_list, aux_beh, :]
#             normalize_emb_aux = F.normalize(emb_aux, dim=1)  # [B, dim]
#             normalize_all_emb_aux = F.normalize(ua_embeddings[:, aux_beh, :], dim=1)  # [N, dim]
#             pos_score = torch.sum(torch.mul(normalize_emb_tgt, normalize_emb_aux),
#                                   dim=1)
#             ttl_score = torch.matmul(normalize_emb_tgt, normalize_all_emb_aux.T)
#
#             pos_score = torch.exp(pos_score / self.ssl_temp)
#             ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
#
#             ssl2_loss += -torch.sum(torch.log(pos_score / ttl_score)) * self.ssl_reg_inter[aux_beh]
#
#         if self.ssl_mode_inter in ['item_side', 'both_side']:
#             emb_tgt = ia_embeddings[input_i_list, -1, :]
#             normalize_emb_tgt = F.normalize(emb_tgt, dim=1)
#             emb_aux = ia_embeddings[input_i_list, aux_beh, :]
#             normalize_emb_aux = F.normalize(emb_aux, dim=1)  # [B, dim]
#             normalize_all_emb_aux = F.normalize(ia_embeddings[:, aux_beh, :], dim=1)  # [N, dim]
#             pos_score = torch.sum(torch.mul(normalize_emb_tgt, normalize_emb_aux),
#                                   dim=1)
#             ttl_score = torch.matmul(normalize_emb_tgt, normalize_all_emb_aux.T)
#
#             pos_score = torch.exp(pos_score / self.ssl_temp)
#             ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
#
#             ssl2_loss += -torch.sum(torch.log(pos_score / ttl_score)) * self.ssl_reg_inter[aux_beh]
#
#         return ssl2_loss


class SSLoss2(nn.Module):
    def __init__(self, data_config, args):
        super(SSLoss2, self).__init__()
        self.config = data_config
        self.ssl_temp = args.ssl_temp
        self.ssl_reg_inter = eval(args.ssl_reg_inter)  # [0.5, 0.5]
        self.ssl_mode_inter = args.ssl_inter_mode
        self.hard_neg = args.hard_neg_ssl2  # v1/v2/v3
        self.select_neg = args.select_neg_ssl2
        self.topk1_user = args.topk1_user  # 目标是取top500-1000的总共500
        self.topk2_user = args.topk2_user
        self.topk1_item = args.topk1_item
        self.topk2_item = args.topk2_item
        self.pmi_th = args.pmi_th
        # self.aux_beh_idx = eval(args.aux_beh_idx)  # [0, 1]
        self.user_indices_remove, self.item_indices_remove = None, None

    def forward(self, input_u_list, input_i_list, ua_embeddings, ia_embeddings, aux_beh, user_batch_indices=None,
                item_batch_indices=None):
        ssl2_loss = 0.

        if self.ssl_mode_inter in ['user_side', 'both_side']:
            emb_tgt = ua_embeddings[input_u_list, -1, :]  # [B, d]
            normalize_emb_tgt = F.normalize(emb_tgt, dim=1)
            emb_aux = ua_embeddings[input_u_list, aux_beh, :]  # [B, d]
            normalize_emb_aux = F.normalize(emb_aux, dim=1)  # [B, dim]
            normalize_all_emb_aux = F.normalize(ua_embeddings[:, aux_beh, :], dim=1)  # [N, dim]
            pos_score = torch.sum(torch.mul(normalize_emb_tgt, normalize_emb_aux),
                                  dim=1)
            ttl_score = torch.matmul(normalize_emb_tgt, normalize_all_emb_aux.T)  # [B, N]
            # 对于每个用户，选取top500-1000的用户作为负样本。
            # 实现时，即将[B,N]的矩阵每一行除了top500-1000以外的值置为0。
            """
            v1: 基于emb相似度，用辅助行为和目标行为下的emb算相似度
            v2: 基于emb相似度，只用目标行为下的emb算相似度
            v3: 基于PMI相似度，只用目标行为下的图算相似度
            选负样本时的策略：基于相似度
            1. 选择topk的作为负样本池子
            2. topk1-topk2的作为负样本池子
            3. 1的逆，除了topk的作为负样本池子 
            """
            if self.hard_neg == 'v1':
                values1 = torch.topk(ttl_score, min(self.topk1_user, self.config['n_users']), dim=1, largest=True)[
                    0]  # [B, topk1]
                indices_to_remove1 = ttl_score < values1[..., -1, None]
                if self.select_neg == 3:
                    indices_to_remove1 = ttl_score >= values1[..., -1, None]
                elif self.select_neg == 2:
                    values2 = torch.topk(ttl_score, self.topk1_user - self.topk2_user, dim=1, largest=True)[0]
                    indices_to_remove2 = ttl_score >= values2[..., 0, None]

                ttl_score[indices_to_remove1] = 0.
                if self.select_neg == 2:
                    ttl_score[indices_to_remove2] = 0.
            elif self.hard_neg == 'v2':
                normalize_all_emb_tar = F.normalize(ua_embeddings[:, -1, :], dim=1)  # [N, dim]
                ttl_score_tar = torch.matmul(normalize_emb_tgt, normalize_all_emb_tar.T)  # [B, N]

                values1 = torch.topk(ttl_score_tar, min(self.topk1_user, self.config['n_users']), dim=1, largest=True)[
                    0]
                indices_to_remove1 = ttl_score_tar < values1[..., -1, None]
                if self.select_neg == 3:
                    indices_to_remove1 = ttl_score_tar >= values1[..., -1, None]
                if self.select_neg == 2:
                    values2 = torch.topk(ttl_score_tar, self.topk1_user - self.topk2_user, dim=1, largest=True)[0]
                    indices_to_remove2 = ttl_score_tar >= values2[..., 0, None]

                ttl_score[indices_to_remove1] = 0.
                if self.select_neg == 2:
                    ttl_score[indices_to_remove2] = 0.

            elif self.hard_neg == 'v3':
                # idx = torch.tensor(range(ttl_score.size(0))).view(-1, 1)
                # ttl_score[idx, user_batch_indices] = 0.
                ttl_score[user_batch_indices] = 0.

            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)

            ssl2_loss += -torch.sum(torch.log(pos_score / ttl_score)) * self.ssl_reg_inter[aux_beh]

        if self.ssl_mode_inter in ['item_side', 'both_side']:
            # ia_embeddings = ia_embeddings[:-1, :, :]  # 去除token embedding
            emb_tgt = ia_embeddings[input_i_list, -1, :]
            normalize_emb_tgt = F.normalize(emb_tgt, dim=1)
            emb_aux = ia_embeddings[input_i_list, aux_beh, :]
            normalize_emb_aux = F.normalize(emb_aux, dim=1)  # [B, dim]
            normalize_all_emb_aux = F.normalize(ia_embeddings[:, aux_beh, :], dim=1)  # [N, dim]
            pos_score = torch.sum(torch.mul(normalize_emb_tgt, normalize_emb_aux),
                                  dim=1)
            ttl_score = torch.matmul(normalize_emb_tgt, normalize_all_emb_aux.T)
            if self.hard_neg == 'v1':
                values1 = torch.topk(ttl_score, min(self.topk1_item, self.config['n_items']), dim=1, largest=True)[0]
                indices_to_remove1 = ttl_score < values1[..., -1, None]
                if self.select_neg == 3:
                    indices_to_remove1 = ttl_score >= values1[..., -1, None]
                elif self.select_neg == 2:
                    values2 = torch.topk(ttl_score, self.topk1_item - self.topk2_item, dim=1, largest=True)[0]
                    indices_to_remove2 = ttl_score >= values2[..., 0, None]

                ttl_score[indices_to_remove1] = 0.
                if self.select_neg == 2:
                    ttl_score[indices_to_remove2] = 0.
            elif self.hard_neg == 'v2':
                normalize_all_emb_tar = F.normalize(ia_embeddings[:, -1, :], dim=1)  # [N, dim]
                ttl_score_tar = torch.matmul(normalize_emb_tgt, normalize_all_emb_tar.T)  # [B, N]

                values1 = torch.topk(ttl_score_tar, min(self.topk1_item, self.config['n_items']), dim=1, largest=True)[
                    0]
                indices_to_remove1 = ttl_score_tar < values1[..., -1, None]
                if self.select_neg == 3:
                    indices_to_remove1 = ttl_score_tar >= values1[..., -1, None]
                elif self.select_neg == 2:
                    values2 = torch.topk(ttl_score_tar, self.topk1_item - self.topk2_item, dim=1, largest=True)[0]
                    indices_to_remove2 = ttl_score_tar >= values2[..., 0, None]

                ttl_score[indices_to_remove1] = 0.
                if self.select_neg == 2:
                    ttl_score[indices_to_remove2] = 0.

            elif self.hard_neg == 'v3':
                # idx = torch.tensor(range(ttl_score.size(0))).view(-1, 1)
                # ttl_score[idx, item_batch_indices] = 0.
                ttl_score[item_batch_indices] = 0.

            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)

            ssl2_loss += -torch.sum(torch.log(pos_score / ttl_score)) * self.ssl_reg_inter[aux_beh]

        return ssl2_loss


def get_lables(temp_set, k=0.9999):
    """
    将temp_set的交互item做过滤和补全，把所有用户的item个数padding到统一长度
    :param temp_set:
    :param k: 长度的比例
    :return: padding后的temp_set以及max_item
    """
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k) - 1]

    print(max_item)
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]  # 截断
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)  # 补全，补的值是n_item,实际上是异常值，因为取不到这个id
    return max_item, temp_set


def get_train_instances1(max_item_list, beh_label_list):
    """
    生成样本对，每个user对应于固定个数的item，个数由行为类型决定。
    比如每个user在view下对应于max_view个交互item
    :param max_item_list: 每个行为下的最大item [ , , ]
    :param beh_label_list: 每个行为的交互字典 [dict1, dict2, dict2]
    :return: user_train: 所有输入训练的用户 ndarray[N, 1]
    beh_item_list: 每个用户在每个行为下的固定个数的交互item list[ndarray[N, max_item1], ndarray[N, max_item2], ndarray[N, max_item3]]
    """
    user_train = []
    beh_item_list = [list() for i in range(n_behs)]  #

    for i in beh_label_list[-1].keys():
        user_train.append(i)
        beh_item_list[-1].append(beh_label_list[-1][i])
        for j in range(n_behs - 1):
            if not i in beh_label_list[j].keys():
                beh_item_list[j].append([n_items] * max_item_list[j])
            else:
                beh_item_list[j].append(beh_label_list[j][i])

    user_train = np.array(user_train)
    beh_item_list = [np.array(beh_item) for beh_item in beh_item_list]
    user_train = user_train[:, np.newaxis]
    return user_train, beh_item_list


def get_train_pairs(user_train_batch, beh_item_tgt_batch):
    """
    对输入的batch users 和 batch items生成训练样本对,
    :param user_train_batch: ndarray[B, 1]
    :param beh_item_tgt_batch: target behavior ndarray[B, max_item_buy]
    :return: user_list ndarray[N, ]  item_list ndarray[N, ]  N为样本对的数量
    """
    input_u_list, input_i_list = [], []
    for i in range(len(user_train_batch)):
        pos_items = beh_item_tgt_batch[i][np.where(beh_item_tgt_batch[i] != n_items)]  # ndarray [x,]
        uid = user_train_batch[i][0]
        input_u_list += [uid] * len(pos_items)
        input_i_list += pos_items.tolist()

    return np.array(input_u_list).reshape([-1]), np.array(input_i_list).reshape([-1])


def test_torch(ua_embeddings, ia_embeddings, rela_embedding, users_to_test, batch_test_flag=False):
    def get_score_np(ua_embeddings, ia_embeddings, rela_embedding, users, items):
        ug_embeddings = ua_embeddings[users]  # []
        pos_ig_embeddings = ia_embeddings[items]
        dot = np.multiply(pos_ig_embeddings, rela_embedding)  # [I, dim] * [1, dim]-> [I, dim]
        batch_ratings = np.matmul(ug_embeddings, dot.T)  # [U, dim] * [dim, I] -> [U, I]
        return batch_ratings

    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    test_users = users_to_test
    n_test_users = len(test_users)

    # pool = torch.multiprocessing.Pool(cores)
    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        # 直接对所有item进行打分
        item_batch = range(ITEM_NUM)
        rate_batch = get_score_np(ua_embeddings, ia_embeddings, rela_embedding, user_batch, item_batch)

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users
    assert count == n_test_users

    pool.close()
    return result


def preprocess_sim_v3(args, config):
    t1 = time()
    select_neg = args.select_neg_ssl2
    topk1_user, topk2_user = args.topk1_user, args.topk2_user
    topk1_item, topk2_item = args.topk1_item, args.topk2_item
    pmi_th = args.pmi_th

    # user_indices_remove_list, item_indices_remove_list = [], []  # 每个行为下都有各自的
    input_u_sim = torch.tensor(config['user_sim'])
    if select_neg == 3:
        user_topk_values1 = torch.topk(input_u_sim, min(topk1_user, config['n_users']), dim=1, largest=True)[
            0][..., -1, None]  # 每行的第topk个值 [N, ]
        user_indices_false = user_topk_values1 == 0.
        user_indices_remove = input_u_sim > user_topk_values1[..., -1, None]  # [N, N] True代表需要remove

        print('user topk 为0值的个数：', sum(user_topk_values1[..., -1, None] == 0.))
        user_indices_remove2 = torch.topk(input_u_sim, min(topk1_user, config['n_users']), dim=1, largest=True)[1]
        # [N, topk]  注意这里是严格保存topk个，和原来的实现方式根据值去选择有点不同——所以可能会造成结果有点不一致

    input_i_sim = torch.tensor(config['item_sim'])
    if select_neg == 3:
        item_topk_values1 = torch.topk(input_i_sim, min(topk1_item, config['n_items']), dim=1, largest=True)[
            0]  # [N, topk]
        item_indices_remove = input_i_sim > item_topk_values1[..., -1, None]  # [N, N] True代表需要remove
        print('item topk 为0值的个数：', sum(item_topk_values1[..., -1, None] == 0.))
        item_indices_token = torch.tensor([False] * item_indices_remove.shape[0], dtype=torch.bool).reshape(-1, 1)
        item_indices_remove = torch.cat([item_indices_remove, item_indices_token], dim=1)

    print('preprocess sim v3:', time() - t1)
    return user_indices_remove, item_indices_remove

# def preprocess_sim_v3(args, config):
#     t1 = time()
#     select_neg = args.select_neg_ssl2
#     topk1_user, topk2_user = args.topk1_user, args.topk2_user
#     topk1_item, topk2_item = args.topk1_item, args.topk2_item
#     pmi_th = args.pmi_th
#
#     # user_indices_remove_list, item_indices_remove_list = [], []  # 每个行为下都有各自的
#     input_u_sim = torch.tensor(config['user_sim'])
#     if select_neg == 3:
#         user_indices_remove = torch.topk(input_u_sim, min(topk1_user, config['n_users']), dim=1, largest=True)[
#             1]  # [N, topk]  注意这里是严格保存topk个，和原来的实现方式根据值去选择有点不同——所以可能会造成结果有点不一致
#
#     input_i_sim = torch.tensor(config['item_sim'])
#     if select_neg == 3:
#         item_indices_remove = torch.topk(input_i_sim, min(topk1_item, config['n_items']), dim=1, largest=True)[
#             1]  # [N, topk]
#     del input_u_sim, input_i_sim
#     print('preprocess sim v3:', time() - t1)
#     return user_indices_remove, item_indices_remove


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.set_deterministic(True)
    # # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # torch.multiprocessing.set_start_method('spawn')
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ******************* fitlog ****************** #
    fitlog_dir = os.path.join('logs', 'MBSSL_unified_debug2', args.dataset)
    os.makedirs(fitlog_dir, exist_ok=True)
    fitlog.set_log_dir(fitlog_dir)  # 设定日志存储的目录

    var_paras = ['batch_size', 'layer_size', 'embed_size'] + ['ssl_reg', 'ssl_ratio', 'ssl_temp', 'ssl_reg_inter'] + [
        'wid', 'decay', 'coefficient', 'mess_dropout'] + ['meta_s', 'meta_r'] + \
                ['hard_neg_ssl2', 'select_neg_ssl2', 'topk1_user', 'topk2_user', 'topk1_item', 'topk2_item', 'pmi_th',
                 'sim_measure', 'sim_mode', 'sim_single', 'sim_weights']
    args_dict = dict({para: vars(args)[para] for para in var_paras})
    fitlog.add_hyper(args_dict)  # 通过这种方式记录ArgumentParser的参数
    set_seed(2020)

    config = dict()
    config['device'] = device
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['behs'] = data_generator.behs
    config['trn_mat'] = data_generator.trnMats[-1]  # 目标行为交互矩阵

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """

    pre_adj_list = data_generator.get_adj_mat()
    config['pre_adjs'] = pre_adj_list
    print('use the pre adjcency matrix')
    n_users, n_items = data_generator.n_users, data_generator.n_items
    behs = data_generator.behs
    n_behs = data_generator.beh_num

    """ 处理相似度 """
    # MBSSL_unified
    sim_weight = eval(args.sim_weights)
    user_sim_mat, item_sim_mat = data_generator.get_similarity_score(beh_idx=0, sim_measure=args.sim_measure)
    user_sim_mat_unified, item_sim_mat_unified = sim_weight[0] * user_sim_mat, sim_weight[0] * item_sim_mat
    # del user_sim_mat, item_sim_mat
    for beh in range(1, len(config['behs'])):
        user_sim_mat, item_sim_mat = data_generator.get_similarity_score(beh_idx=beh, sim_measure=args.sim_measure)
        user_sim_mat_unified += sim_weight[beh] * user_sim_mat
        item_sim_mat_unified += sim_weight[beh] * item_sim_mat
        # del user_sim_mat, item_sim_mat
    config['user_sim'] = user_sim_mat_unified.todense()
    config['item_sim'] = item_sim_mat_unified.todense()

    # MBSSL_unified 之前
    # user_sim_mat_list, item_sim_mat_list = [], []
    # for beh in range(len(config['behs'])):
    #     user_sim_mat, item_sim_mat = data_generator.get_similarity_score_old(beh_idx=beh, sim_measure=args.sim_measure)
    #     user_sim_mat_list.append(user_sim_mat)
    #     item_sim_mat_list.append(item_sim_mat)
    # user_sim, item_sim = unified_sim(user_sim_mat_list, item_sim_mat_list, args, config)
    # config['user_sim'] = user_sim
    # config['item_sim'] = item_sim

    """
    preprocess sim mat for v3
    """
    if args.hard_neg_ssl2 == 'v3':
        user_indices, item_indices = preprocess_sim_v3(args, config)

    trnDicts = copy.deepcopy(data_generator.trnDicts)
    max_item_list = []  # 每个行为矩阵的最大item编号
    beh_label_list = []  # 每个行为的交互信息字典(key:uid, value:[item1, item2])，padding item长度到统一长度
    for i in range(n_behs):
        max_item, beh_label = get_lables(trnDicts[i])
        max_item_list.append(max_item)
        beh_label_list.append(beh_label)

    t0 = time()

    model = GHCF(max_item_list, data_config=config, args=args).to(device)
    augmentor = Augmentor(data_config=config, args=args)
    recloss = RecLoss(data_config=config, args=args).to(device)
    ssloss = SSLoss(data_config=config, args=args).to(device)
    ssloss2 = SSLoss2(data_config=config, args=args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # metabalance = MetaBalance(model.parameters())
    metabalance = MetaBalance3(model.parameters(), relax_factor=args.meta_r, beta=args.meta_b)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_gamma)
    cur_best_pre_0 = 0.
    print('without pretraining.')

    run_time = 1

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False

    user_train1, beh_item_list = get_train_instances1(max_item_list, beh_label_list)

    nonshared_idx = -1  # 非共享的参数idx

    for epoch in range(args.epoch):
        model.train()

        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        beh_item_list = [beh_item[shuffle_indices] for beh_item in beh_item_list]

        t1 = time()
        loss, rec_loss, emb_loss, ssl_loss, ssl2_loss = 0., 0., 0., 0., 0.

        n_batch = int(len(user_train1) / args.batch_size)

        # augment the graph
        aug_time = time()
        sub_mat = {}
        if args.aug_type in [0, 1]:
            sub_mat['sub1'] = model._convert_sp_mat_to_sp_tensor(augmentor.augment_adj_mat(aug_type=args.aug_type))
            sub_mat['sub2'] = model._convert_sp_mat_to_sp_tensor(augmentor.augment_adj_mat(aug_type=args.aug_type))
        else:
            for k in range(1, model.n_layers + 1):
                sub_mat['sub1%d' % k] = model._convert_sp_mat_to_sp_tensor(
                    augmentor.augment_adj_mat(aug_type=args.aug_type))
                sub_mat['sub2%d' % k] = model._convert_sp_mat_to_sp_tensor(
                    augmentor.augment_adj_mat(aug_type=args.aug_type))
        # print('aug time: %.1fs' % (time() - aug_time))

        iter_time = time()

        for idx in range(n_batch):
            optimizer.zero_grad()

            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            beh_batch = [beh_item[start_index:end_index] for beh_item in
                         beh_item_list]  # [[B, max_item1], [B, max_item2], [B, max_item3]]

            u_batch_list, i_batch_list = get_train_pairs(user_train_batch=u_batch,
                                                         beh_item_tgt_batch=beh_batch[-1])  # ndarray[N, ]  ndarray[N, ]

            # load into cuda
            u_batch = torch.from_numpy(u_batch).to(device)
            beh_batch = [torch.from_numpy(beh_item).to(device) for beh_item in beh_batch]
            u_batch_indices, i_batch_indices = None, None
            if args.hard_neg_ssl2 == 'v3':
                u_batch_indices = user_indices[u_batch_list].to(device)  # [B, N]
                i_batch_indices = item_indices[i_batch_list].to(device)  # [B, N]
            u_batch_list = torch.from_numpy(u_batch_list).to(device)
            i_batch_list = torch.from_numpy(i_batch_list).to(device)

            model_time = time()
            ua_embeddings, ia_embeddings, ua_embeddings_sub1, ia_embeddings_sub1, ua_embeddings_sub2, ia_embeddings_sub2, rela_embeddings, \
            attn_user, attn_item = model(sub_mat, device)
            # print('model time: %.1fs' % (time() - model_time))
            # rec_loss_time = time()
            batch_rec_loss, batch_emb_loss = recloss(u_batch, beh_batch, ua_embeddings, ia_embeddings, rela_embeddings)
            # print('rec loss time: %.1fs' % (time() - rec_loss_time))
            # ssl_loss_time = time()
            batch_ssl_loss = ssloss(u_batch_list, i_batch_list, ua_embeddings_sub1[:, -1, :],
                                    ua_embeddings_sub2[:, -1, :], ia_embeddings_sub1[:, -1, :],
                                    ia_embeddings_sub2[:, -1, :])
            # print('ssl loss time: %.1fs' % (time() - ssl_loss_time))
            batch_ssl2_loss_list = []
            for aux_beh in eval(args.aux_beh_idx):
                """  1. 用各自行为下的相似度去选负样本 """
                # aux_beh_ssl2_loss = ssloss2(u_batch_list, i_batch_list, ua_embeddings, ia_embeddings, aux_beh, u_batch_indices[aux_beh], i_batch_indices[aux_beh])
                """  2. 用特定统一的相似度去选负样本 """
                aux_beh_ssl2_loss = ssloss2(u_batch_list, i_batch_list, ua_embeddings, ia_embeddings, aux_beh,
                                            u_batch_indices, i_batch_indices)
                batch_ssl2_loss_list.append(aux_beh_ssl2_loss)
            batch_ssl2_loss = sum(batch_ssl2_loss_list)
            batch_loss = batch_rec_loss + batch_emb_loss + batch_ssl_loss + batch_ssl2_loss
            # for debug, watch the gradient of each loss
            # batch_emb_loss.backward(retain_graph=True)
            # model.zero_grad()
            # batch_rec_loss.backward(retain_graph=True)
            # model.zero_grad()
            # batch_ssl_loss.backward(retain_graph=True)
            # model.zero_grad()
            # batch_ssl2_loss.backward(retain_graph=True)
            # model.zero_grad()

            # 1. 找到非共享参数
            if nonshared_idx == -1:
                batch_ssl_loss.backward(retain_graph=True)
                for p_idx, p_name in enumerate(model.all_weights):
                    p = model.all_weights[p_name]
                    if p.grad is None or p.grad.equal(torch.zeros_like(p.grad)):
                        nonshared_idx = p_idx
                        break
                model.zero_grad()

            # ************************************* meta balance ************************************** #
            metabalance.step([batch_rec_loss, batch_ssl_loss] + batch_ssl2_loss_list, nonshared_idx, args.meta_s)
            batch_emb_loss.backward()  # 累加梯度
            # batch_emb_loss.backward()
            optimizer.step()

            loss += batch_loss.item() / n_batch
            rec_loss += batch_rec_loss.item() / n_batch
            emb_loss += batch_emb_loss.item() / n_batch
            ssl_loss += batch_ssl_loss.item() / n_batch
            ssl2_loss += batch_ssl2_loss.item() / n_batch

        # print('iter time: %.1fs' % (time() - iter_time))
        if args.lr_decay: scheduler.step()
        torch.cuda.empty_cache()

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % args.test_epoch != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, rec_loss, emb_loss, ssl_loss, ssl2_loss)
                print(perf_str)
            continue

        t2 = time()
        model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, _, _, _, _, rela_embeddings, attn_user, attn_item = model(sub_mat, device)
            users_to_test = list(data_generator.test_set.keys())
            ret = test_torch(ua_embeddings[:, -1, :].detach().cpu().numpy(),
                             ia_embeddings[:, -1, :].detach().cpu().numpy(),
                             rela_embeddings[behs[-1]].detach().cpu().numpy(), users_to_test)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]:, recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (
                           epoch, t2 - t1, t3 - t2, ret['recall'][0],
                           ret['recall'][1],
                           ret['precision'][0], ret['precision'][1], ret['hit_ratio'][0], ret['hit_ratio'][1],
                           ret['ndcg'][0], ret['ndcg'][1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop, flag = early_stopping_new(ret['recall'][0], cur_best_pre_0,
                                                                              stopping_step, expected_order='acc',
                                                                              flag_step=10)
        if flag:
            fitlog.add_best_metric({"test": {"recall@%d" % Ks[i]: ret['recall'][i] for i in range(len(Ks))}})
            fitlog.add_best_metric({"test": {"ndcg@%d" % Ks[i]: ret['ndcg'][i] for i in range(len(Ks))}})
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        if flag and args.save_att:
            arg_str = '_'.join(
                ['v' + str(args.ssl_type), args.ssl_mode, str(args.ssl_ratio), str(args.ssl_reg),
                 str(args.ssl_temp)])
            data_path = os.path.join('Att_Scores', data_generator.dataset_name, arg_str)
            os.makedirs(data_path, exist_ok=True)
            with open(data_path + '/user_att_SSL_v7_gatne_meta_debug.txt', 'w') as f:
                save_user_idx = list(range(100))
                for i in range(100):
                    f.write(str(attn_user[save_user_idx[i]]) + '\n\n')

            with open(data_path + '/item_att_SSL_v7_gatne_meta_debug.txt', 'w') as f:
                save_item_idx = list(range(100))
                for i in range(100):
                    f.write(str(attn_item[save_item_idx[i]]) + '\n\n')

            print('saving att finished. \n')

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if args.save_emb:
            if eval(args.sim_weights[0]) == 0.:
                sim_name = "tar"
            else: sim_name = "avg"
            arg_str = '_'.join(
                ['temp-' + str(args.ssl_temp), 'r-' + str(args.meta_r), 'sw-' + sim_name])
            data_path = os.path.join('Embeddings_torch', data_generator.dataset_name, 'mbssl_unified_debug2', arg_str)
            os.makedirs(data_path, exist_ok=True)
            if args.save_emb_target:
                user_file_name = '/userEmbeddingsTar'
                item_file_name = '/itemEmbeddingsTar'
                ua_embeddings = ua_embeddings[:, -1, :].detach().cpu().numpy().reshape([-1, args.embed_size])
                ia_embeddings = ia_embeddings[:, -1, :].detach().cpu().numpy().reshape([-1, args.embed_size])
            else:
                user_file_name = '/userEmbeddingsAll'
                item_file_name = '/itemEmbeddingsAll'
                ua_embeddings = ua_embeddings.detach().cpu().numpy()
                ia_embeddings = ia_embeddings.detach().cpu().numpy()
            if args.save_bst:
                if flag:
                    print('Find a better model.')
                    with open(data_path + user_file_name + '_bst.pkl', 'wb') as f:
                        pickle.dump(ua_embeddings, f)
                    with open(data_path + item_file_name + '_bst.pkl', 'wb') as f:
                        pickle.dump(ia_embeddings, f)
                    with open(data_path + '/relaEmbeddingsTar_bst.pkl', 'wb') as f:
                        pickle.dump(rela_embeddings[behs[-1]], f)
                    print('Saving embeddings finished.')
            else:
                with open(data_path + user_file_name + '.pkl', 'wb') as f:
                    pickle.dump(ua_embeddings, f)
                with open(data_path + item_file_name + '.pkl', 'wb') as f:
                    pickle.dump(ia_embeddings, f)
                with open(data_path + '/relaEmbeddingsTar.pkl', 'wb') as f:
                    pickle.dump(rela_embeddings[behs[-1]], f)
                print('Saving embeddings finished.')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    # final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
    #              (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
    #               '\t'.join(['%.5f' % r for r in pres[idx]]),
    #               '\t'.join(['%.5f' % r for r in hit[idx]]),
    #               '\t'.join(['%.5f' % r for r in ndcgs[idx]]))

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.4f' % r for r in recs[idx]]),
                  '\t'.join(['%.4f' % r for r in ndcgs[idx]]))
    print(final_perf)
fitlog.finish()