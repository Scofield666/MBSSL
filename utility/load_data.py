'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import math

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import pickle
import os
from itertools import combinations
import tqdm
# class Data(object):
#     def __init__(self, path, batch_size):
#         self.path = path
#         self.batch_size = batch_size
#
#         train_file = path + '/train.txt'
#         test_file = path + '/test.txt'
#
#         pv_file = path +'/pv.txt'
#         cart_file = path +'/cart.txt'
#
#         #get number of users and items
#         self.n_users, self.n_items = 0, 0  # train.txt+test.txt中的用户数目和item数目
#         self.n_train, self.n_test = 0, 0  # 代表train.txt和test.txt中的交互信息数
#         self.neg_pools = {}
#
#         self.exist_users = []  # train.txt中的用户，即目标行为中的用户
#
#         with open(train_file) as f:
#             for l in f.readlines():
#                 if len(l) > 0:
#                     l = l.strip('\n').split(' ')
#                     items = [int(i) for i in l[1:]]
#                     uid = int(l[0])
#                     self.exist_users.append(uid)
#                     self.n_items = max(self.n_items, max(items))
#                     self.n_users = max(self.n_users, uid)
#                     self.n_train += len(items)
#
#         with open(test_file) as f:
#             for l in f.readlines():
#                 if len(l) > 0:
#                     l = l.strip('\n')
#                     try:
#                         items = [int(i) for i in l.split(' ')[1:]]
#                     except Exception:
#                         continue
#                     self.n_items = max(self.n_items, max(items))
#                     self.n_test += len(items)
#         self.n_items += 1
#         self.n_users += 1
#
#         self.print_statistics()
#         # dok_matrix: dictionary of keys. 适用于递增的矩阵，之后需要填充该矩阵
#         # 1. 交互信息邻接矩阵形式，为每个行为都初始化一个稀疏矩阵，之后慢慢填充
#         self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # target beh
#         self.R_pv = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # pv heh
#         self.R_cart = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # cart beh
#         # 2. 交互信息邻接链表形式，每个行为的交互信息都是一个字典。key为uid, value为item list[]
#         self.train_items, self.test_set = {}, {}
#         self.pv_set, self.cart_set = {},{}
#         with open(train_file) as f_train:
#             with open(test_file) as f_test:
#                 for l in f_train.readlines():
#                     if len(l) == 0: break
#                     l = l.strip('\n')
#                     items = [int(i) for i in l.split(' ')]
#                     uid, train_items = items[0], items[1:]
#
#                     for i in train_items:
#                         self.R[uid, i] = 1.  # 填充矩阵
#
#                     self.train_items[uid] = train_items
#
#                 for l in f_test.readlines():
#                     if len(l) == 0: break
#                     l = l.strip('\n')
#                     try:
#                         items = [int(i) for i in l.split(' ')]
#                     except Exception:
#                         continue
#
#                     uid, test_items = items[0], items[1:]
#                     self.test_set[uid] = test_items
#         with open(pv_file) as f_pv:
#             for l in f_pv.readlines():
#                 if len(l) == 0: break
#                 l = l.strip('\n')
#                 items = [int(i) for i in l.split(' ')]
#                 uid, pv_items = items[0], items[1:]
#
#                 for i in pv_items:
#                     self.R_pv[uid, i] = 1.
#                 self.pv_set[uid]=pv_items
#
#         with open(cart_file) as f_cart:
#             for l in f_cart.readlines():
#                 if len(l) == 0: break
#                 l = l.strip('\n')
#                 items = [int(i) for i in l.split(' ')]
#                 uid, cart_items = items[0], items[1:]
#
#                 for i in cart_items:
#                     self.R_cart[uid, i] = 1.
#                 self.cart_set[uid]=cart_items
#
#
#     def get_adj_mat(self):
#         try:
#             t1 = time()
#             adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
#             norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
#             mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
#
#             adj_mat_pv = sp.load_npz(self.path + '/s_adj_mat_pv.npz')
#             norm_adj_mat_pv = sp.load_npz(self.path + '/s_norm_adj_mat_pv.npz')
#             mean_adj_mat_pv = sp.load_npz(self.path + '/s_mean_adj_mat_pv.npz')
#
#             adj_mat_cart = sp.load_npz(self.path + '/s_adj_mat_cart.npz')
#             norm_adj_mat_cart = sp.load_npz(self.path + '/s_norm_adj_mat_cart.npz')
#             mean_adj_mat_cart = sp.load_npz(self.path + '/s_mean_adj_mat_cart.npz')
#
#             print('already load adj matrix', adj_mat.shape, time() - t1)
#
#         except Exception:
#             adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat(self.R)
#             adj_mat_pv, norm_adj_mat_pv, mean_adj_mat_pv = self.create_adj_mat(self.R_pv)
#             adj_mat_cart, norm_adj_mat_cart, mean_adj_mat_cart = self.create_adj_mat(self.R_cart)
#             sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
#             sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
#             sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
#
#             sp.save_npz(self.path + '/s_adj_mat_pv.npz', adj_mat_pv)
#             sp.save_npz(self.path + '/s_norm_adj_mat_pv.npz', norm_adj_mat_pv)
#             sp.save_npz(self.path + '/s_mean_adj_mat_pv.npz', mean_adj_mat_pv)
#
#             sp.save_npz(self.path + '/s_adj_mat_cart.npz', adj_mat_cart)
#             sp.save_npz(self.path + '/s_norm_adj_mat_cart.npz', norm_adj_mat_cart)
#             sp.save_npz(self.path + '/s_mean_adj_mat_cart.npz', mean_adj_mat_cart)
#
#         try:
#             pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
#             pre_adj_mat_pv = sp.load_npz(self.path + '/s_pre_adj_mat_pv.npz')
#             pre_adj_mat_cart = sp.load_npz(self.path + '/s_pre_adj_mat_cart.npz')
#         except Exception:
#
#             rowsum = np.array(adj_mat.sum(1))
#             d_inv = np.power(rowsum, -0.5).flatten()
#             d_inv[np.isinf(d_inv)] = 0.
#             d_mat_inv = sp.diags(d_inv)
#
#             norm_adj = d_mat_inv.dot(adj_mat)
#             norm_adj = norm_adj.dot(d_mat_inv)
#             print('generate pre adjacency matrix.')
#             pre_adj_mat = norm_adj.tocsr()
#             sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
#
#             rowsum = np.array(adj_mat_pv.sum(1))
#             d_inv = np.power(rowsum, -0.5).flatten()
#             d_inv[np.isinf(d_inv)] = 0.
#             d_mat_inv = sp.diags(d_inv)
#
#             norm_adj = d_mat_inv.dot(adj_mat_pv)
#             norm_adj = norm_adj.dot(d_mat_inv)
#             print('generate pre view adjacency matrix.')
#             pre_adj_mat_pv = norm_adj.tocsr()
#             sp.save_npz(self.path + '/s_pre_adj_mat_pv.npz', norm_adj)
#
#             rowsum = np.array(adj_mat_cart.sum(1))
#             d_inv = np.power(rowsum, -0.5).flatten()
#             d_inv[np.isinf(d_inv)] = 0.
#             d_mat_inv = sp.diags(d_inv)
#
#             norm_adj = d_mat_inv.dot(adj_mat_cart)
#             norm_adj = norm_adj.dot(d_mat_inv)
#             print('generate pre view adjacency matrix.')
#             pre_adj_mat_cart = norm_adj.tocsr()
#             sp.save_npz(self.path + '/s_pre_adj_mat_cart.npz', norm_adj)
#
#
#         return pre_adj_mat,pre_adj_mat_pv,pre_adj_mat_cart
#
#     def create_adj_mat(self,which_R):
#         """
#         为当前行为的交互信息矩阵[n_user,n_item]，生成全图的邻接矩阵[n_user+n_item, n_user+n_item]
#         :param which_R: 该行为的交互信息矩阵。dok_matrix
#         :return:
#         """
#         # 1. 构造全图邻接稀疏矩阵 [n_user+n_item, n_user+n_item]
#         t1 = time()
#         adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)  # 全图邻接矩阵
#         # 将dok_matrix转成lil_matrix再统一给矩阵赋值
#         adj_mat = adj_mat.tolil()
#         R = which_R.tolil()
#         adj_mat[:self.n_users, self.n_users:] = R  # 矩阵右上角
#         adj_mat[self.n_users:, :self.n_users] = R.T  # 矩阵左下角
#         adj_mat = adj_mat.todok()
#         print('already create adjacency matrix', adj_mat.shape, time() - t1)
#
#         # 2. 构造GCN的归一化邻接矩阵，这里使用简单归一化：即D^-1(A+I)
#         t2 = time()
#
#         def normalized_adj_single(adj):
#             rowsum = np.array(adj.sum(1))
#
#             d_inv = np.power(rowsum, -1).flatten()
#             d_inv[np.isinf(d_inv)] = 0.
#             d_mat_inv = sp.diags(d_inv)  # 度矩阵
#
#             norm_adj = d_mat_inv.dot(adj)  # 归一化：左乘度矩阵
#             # norm_adj = adj.dot(d_mat_inv)
#             print('generate single-normalized adjacency matrix.')
#             return norm_adj.tocoo()
#
#         def check_adj_if_equal(adj):
#             dense_A = np.array(adj.todense())
#             degree = np.sum(dense_A, axis=1, keepdims=False)
#
#             temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
#             print('check normalized adjacency matrix whether equal to this laplacian matrix.')
#             return temp
#
#         norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
#         mean_adj_mat = normalized_adj_single(adj_mat)
#
#         print('already normalize adjacency matrix', time() - t2)
#         return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
#
#     def negative_pool(self):
#         t1 = time()
#         for u in self.train_items.keys():
#             neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
#             pools = [rd.choice(neg_items) for _ in range(100)]
#             self.neg_pools[u] = pools
#         print('refresh negative pools', time() - t1)
#
#     def sample(self):
#         if self.batch_size <= self.n_users:
#             users = rd.sample(self.exist_users, self.batch_size)
#         else:
#             users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
#
#
#         def sample_pos_items_for_u(u, num):
#             pos_items = self.train_items[u]
#             n_pos_items = len(pos_items)
#             pos_batch = []
#             while True:
#                 if len(pos_batch) == num: break
#                 pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
#                 pos_i_id = pos_items[pos_id]
#
#                 if pos_i_id not in pos_batch:
#                     pos_batch.append(pos_i_id)
#             return pos_batch
#
#         def sample_neg_items_for_u(u, num):
#             neg_items = []
#             while True:
#                 if len(neg_items) == num: break
#                 neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
#                 if neg_id not in self.train_items[u] and neg_id not in neg_items:
#                     neg_items.append(neg_id)
#             return neg_items
#
#         def sample_neg_items_for_u_from_pools(u, num):
#             neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
#             return rd.sample(neg_items, num)
#
#         pos_items, neg_items = [], []
#         for u in users:
#             pos_items += sample_pos_items_for_u(u, 1)
#             neg_items += sample_neg_items_for_u(u, 1)
#
#         return users, pos_items, neg_items
#
#     def sample_test(self):
#         if self.batch_size <= self.n_users:
#             users = rd.sample(self.test_set.keys(), self.batch_size)
#         else:
#             users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
#
#         def sample_pos_items_for_u(u, num):
#             pos_items = self.test_set[u]
#             n_pos_items = len(pos_items)
#             pos_batch = []
#             while True:
#                 if len(pos_batch) == num: break
#                 pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
#                 pos_i_id = pos_items[pos_id]
#
#                 if pos_i_id not in pos_batch:
#                     pos_batch.append(pos_i_id)
#             return pos_batch
#
#         def sample_neg_items_for_u(u, num):
#             neg_items = []
#             while True:
#                 if len(neg_items) == num: break
#                 neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
#                 if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
#                     neg_items.append(neg_id)
#             return neg_items
#
#         def sample_neg_items_for_u_from_pools(u, num):
#             neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
#             return rd.sample(neg_items, num)
#
#         pos_items, neg_items = [], []
#         for u in users:
#             pos_items += sample_pos_items_for_u(u, 1)
#             neg_items += sample_neg_items_for_u(u, 1)
#
#         return users, pos_items, neg_items
#
#
#
#
#
#
#     def get_num_users_items(self):
#         return self.n_users, self.n_items
#
#     def print_statistics(self):
#         print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
#         print('n_interactions=%d' % (self.n_train + self.n_test))
#         print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
#
#
#     def get_sparsity_split(self):
#         try:
#             split_uids, split_state = [], []
#             lines = open(self.path + '/sparsity.split', 'r').readlines()
#
#             for idx, line in enumerate(lines):
#                 if idx % 2 == 0:
#                     split_state.append(line.strip())
#                     print(line.strip())
#                 else:
#                     split_uids.append([int(uid) for uid in line.strip().split(' ')])
#             print('get sparsity split.')
#
#         except Exception:
#             split_uids, split_state = self.create_sparsity_split()
#             f = open(self.path + '/sparsity.split', 'w')
#             for idx in range(len(split_state)):
#                 f.write(split_state[idx] + '\n')
#                 f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
#             print('create sparsity split.')
#
#         return split_uids, split_state
#
#     def create_sparsity_split(self):
#         all_users_to_test = list(self.test_set.keys())
#         user_n_iid = dict()
#
#         # generate a dictionary to store (key=n_iids, value=a list of uid).
#         for uid in all_users_to_test:
#             train_iids = self.train_items[uid]
#             test_iids = self.test_set[uid]
#
#             n_iids = len(train_iids)
#
#             if n_iids not in user_n_iid.keys():
#                 user_n_iid[n_iids] = [uid]
#             else:
#                 user_n_iid[n_iids].append(uid)
#
#
#         split_uids = list()  # n类用户列表，[[1类用户列表]，[2类], ...]
#
#         # split the whole user set into four subset.
#         temp = []
#         count = 1
#         fold = 4
#         n_count = (self.n_train)
#         n_rates = 0
#
#         split_state = []  # 对split_uid的描述信息，代表每类的用户个数 [str1, str2, str3, ...]
#         temp0=[]
#         temp1=[]
#         temp2=[]
#         temp3=[]
#         temp4=[]
#
#         #print user_n_iid
#
#         for idx, n_iids in enumerate(sorted(user_n_iid)):
#             if n_iids <9:
#                 temp0+=user_n_iid[n_iids]
#             elif n_iids <13:
#                 temp1+=user_n_iid[n_iids]
#             elif n_iids <17:
#                 temp2+=user_n_iid[n_iids]
#             elif n_iids <20:
#                 temp3+=user_n_iid[n_iids]
#             else:
#                 temp4+=user_n_iid[n_iids]
#
#         split_uids.append(temp0)
#         split_uids.append(temp1)
#         split_uids.append(temp2)
#         split_uids.append(temp3)
#         split_uids.append(temp4)
#         split_state.append("#users=[%d]"%(len(temp0)))
#         split_state.append("#users=[%d]"%(len(temp1)))
#         split_state.append("#users=[%d]"%(len(temp2)))
#         split_state.append("#users=[%d]"%(len(temp3)))
#         split_state.append("#users=[%d]"%(len(temp4)))
#
#
#         return split_uids, split_state
#
#
#
#     def create_sparsity_split2(self):
#         all_users_to_test = list(self.test_set.keys())
#         user_n_iid = dict()
#
#         # generate a dictionary to store (key=n_iids, value=a list of uid).
#         for uid in all_users_to_test:
#             train_iids = self.train_items[uid]
#             test_iids = self.test_set[uid]
#
#             n_iids = len(train_iids) + len(test_iids)
#
#             if n_iids not in user_n_iid.keys():
#                 user_n_iid[n_iids] = [uid]
#             else:
#                 user_n_iid[n_iids].append(uid)
#         split_uids = list()
#
#         # split the whole user set into four subset.
#         temp = []
#         count = 1
#         fold = 4
#         n_count = (self.n_train + self.n_test)
#         n_rates = 0
#
#         split_state = []
#         for idx, n_iids in enumerate(sorted(user_n_iid)):
#             temp += user_n_iid[n_iids]
#             n_rates += n_iids * len(user_n_iid[n_iids])
#             n_count -= n_iids * len(user_n_iid[n_iids])
#
#             if n_rates >= count * 0.25 * (self.n_train + self.n_test):
#                 split_uids.append(temp)
#
#                 state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
#                 split_state.append(state)
#                 print(state)
#
#                 temp = []
#                 n_rates = 0
#                 fold -= 1
#
#             if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
#                 split_uids.append(temp)
#
#                 state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
#                 split_state.append(state)
#                 print(state)
#
#
#
#         return split_uids, split_state
from functools import partial
import multiprocessing
import torch

# def init(user_occur, item_occur, user_sim, item_sim):
#     global user_occur_mat, item_occur_mat
#     global user_sim_mat, item_sim_mat
#     user_occur_mat = user_occur
#     item_occur_mat = item_occur
#     user_sim_mat = user_sim
#     item_sim_mat = item_sim

def init(mat_name, mat):
    """
    根据变量名动态创建全局变量
    :param mat_name: 矩阵名，比如"user_occur_mat"
    :param mat: 矩阵变量
    :return:
    """
    globals()[mat_name + '_mat'] = mat
    # 等价于下面两行
    # global user_occur_mat
    # user_occur_mat = mat


class MProcessor(object):
    def __init__(self, trn_dict, n_nodes, mat_name):
        self.n_nodes = n_nodes
        self.trn_dict = trn_dict
        self.mat_name = mat_name  # 'user_occur'
        self.global_mat_name = self.mat_name + '_mat'  # str 'user_occur_mat'

    def func(self, row_idx, ARRAY_SHAPE):
        pass

    def run(self):
        pass


class Occur_MProcessor(MProcessor):
    def func(self, row_idx,  ARRAY_SHAPE):
        for col_idx in range(row_idx + 1, ARRAY_SHAPE[1]):
            res = list(set(self.trn_dict[row_idx]) & set(self.trn_dict[col_idx])).__len__()
            eval(self.global_mat_name)[row_idx * ARRAY_SHAPE[1] + col_idx] = res  # 基于变量名来引用变量，调用eval函数

    def run(self):
        print('{} MProcessor start'.format(self.global_mat_name))
        t1 = time()
        array = np.zeros((self.n_nodes, self.n_nodes))
        array_shared = multiprocessing.RawArray('d', array.ravel())
        func_partial = partial(self.func, ARRAY_SHAPE=array.shape)
        with multiprocessing.Pool(processes=8, initializer=init, initargs=(self.mat_name, array_shared)) as p:
            p.map(func_partial, range(array.shape[0]))
        new_array = np.frombuffer(array_shared, np.double).reshape(array.shape)
        new_array += new_array.T
        print('{} MProcessor finish {}'.format(self.global_mat_name, time() - t1))
        return new_array


class Sim_MProcessor(MProcessor):
    def __init__(self, trn_dict, n_nodes, mat_name, user_occur_mat, item_occur_mat, which, sim_measure):
        super(Sim_MProcessor, self).__init__(trn_dict, n_nodes, mat_name)
        self.user_occur_mat = user_occur_mat  # 在父类基础上，扩展新变量
        self.item_occur_mat = item_occur_mat
        self.which = which  # "user" or "item"，代表当前计算的是user还是item相似度
        self.sim_measure = sim_measure

    def func(self, row_idx, ARRAY_SHAPE):
        for col_idx in range(row_idx + 1, ARRAY_SHAPE[1]):
            res = getattr(self, self.sim_measure)(row_idx, col_idx)
            # if self.sim_measure == 'pmi': res = self.pmi(row_idx, col_idx)
            # elif self.sim_measure == 'swing': res = self.swing(row_idx, col_idx)
            eval(self.global_mat_name)[row_idx * ARRAY_SHAPE[1] + col_idx] = res  # 基于变量名来引用变量，调用eval函数

    def pmi(self, u, v):
        occur_mat = self.user_occur_mat if self.which == 'user' else self.item_occur_mat
        if u not in self.trn_dict or len(self.trn_dict[u]) == 0: return 0.
        if v not in self.trn_dict or len(self.trn_dict[v]) == 0: return 0.
        p_u = len(self.trn_dict[u]) * 1. / self.n_nodes
        p_v = len(self.trn_dict[v]) * 1. / self.n_nodes
        p_u_v = occur_mat[u][v] * 1. / self.n_nodes
        if p_u_v == 0.: return 0.
        return np.log(p_u_v / (p_u * p_v))

    def swing(self, u, v, alpha=0.5):
        occur_mat = self.item_occur_mat if self.which == 'user' else self.user_occur_mat
        cur_pairs = list(combinations(set(self.trn_dict[u]) & set(self.trn_dict[v]), 2))
        result = 0.0
        for (i, j) in cur_pairs:
            result += 1. / (alpha + occur_mat[i][j])
        return result

    def run(self):
        print('{} MProcessor start'.format(self.global_mat_name))
        t1 = time()
        array = np.zeros((self.n_nodes, self.n_nodes))
        array_shared = multiprocessing.RawArray('d', array.ravel())
        func_partial = partial(self.func, ARRAY_SHAPE=array.shape)
        with multiprocessing.Pool(processes=32, initializer=init, initargs=(self.mat_name, array_shared)) as p:
            p.map(func_partial, range(array.shape[0]))
            # p.map(func_partial, range(100))
        new_array = np.frombuffer(array_shared, np.double).reshape(array.shape)
        new_array += new_array.T
        print('{} MProcessor finish {}'.format(self.global_mat_name, time() - t1))
        return new_array


class DataHandler(object):
    def __init__(self, dataset, batch_size):
        self.dataset_name = dataset
        self.batch_size = batch_size
        if self.dataset_name.find('Taobao') != -1 or self.dataset_name.find('Beibei') != -1:
            behs = ['pv', 'cart', 'train']
        elif self.dataset_name.find('yelp') != -1:
            if self.dataset_name.find('notip') != -1:
                behs = ['neg', 'neutral', 'train']
            else:
                behs = ['tip', 'neg', 'neutral', 'train']
        elif self.dataset_name.find('ML10M') != -1:
            behs = ['neg', 'neutral', 'train']

        elif self.dataset_name.find('kwai') != -1:
            behs = ['click', 'like', 'comment', 'train']

        elif self.dataset_name.find('Tmall') != -1:
            behs = ['pv', 'fav', 'cart', 'train']

        elif self.dataset_name.find('IJCAI') != -1:
            behs = ['click', 'fav', 'cart', 'train']

        self.predir = '../../../data/' + self.dataset_name
        self.behs = behs
        self.beh_num = len(behs)

        self.trnMats = None  # [dok_matrix1, dok2, ..]  存放k个行为的交互信息，邻接矩阵形式
        self.tstMats = None  # dok_matrix
        self.trnDicts = None  # [dict{}, dict{}], 存放k个行为的交互信息，邻接链表形式 key为uid, value为item list[]
        self.trnDicts_item = None
        self.tstDicts = None  # dict()

        # get number of users and items
        self.n_users, self.n_items = 0, 0  # train.txt+test.txt中的用户数目和item数目
        self.n_train, self.n_test = 0, 0  # 代表train.txt和test.txt中的交互信息数
        self.neg_pools = {}

        self.exist_users = []  # train.txt中的用户，即目标行为中的用户

        self.LoadData()

        # For similartiy
        self.saveSimMatPath = 'Sim_Mats/' + self.dataset_name
        os.makedirs(self.saveSimMatPath, exist_ok=True)
        # self.get_similarity_score()

    def LoadData(self):
        # 1. 先加载train.txt 和 test.txt，填充基本信息，如n_user n_item n_train n_test
        for i in range(len(self.behs) - 1):
            beh = self.behs[i]
            file_name = self.predir + '/' + beh + '.txt'
            with open(file_name) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        self.n_items = max(self.n_items, max(items))
                        self.n_users = max(self.n_users, uid)

        train_file = self.predir + '/train.txt'
        test_file = self.predir + '/test.txt'
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1


        # 2. 再初始化和填充交互矩阵，
        # 2.1 邻接矩阵形式，用dok matrix保存到trnmats tstmats中
        # 2.2 邻接链表形式，用字典保存到trndicts tstdicts中
        # dok_matrix: dictionary of keys. 适用于递增的矩阵，之后需要填充该矩阵
        self.trnMats = [sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) for i in range(len(self.behs))]  # 存放k个行为的交互信息，邻接矩阵形式
        self.tstMats = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.trnDicts = [dict() for i in range(len(self.behs))]  # 存放k个行为的交互信息，邻接链表形式 key为uid, value为item list[]
        self.trnDicts_item = [dict() for i in range(len(self.behs))]
        self.tstDicts = dict()
        self.interNum = [0 for i in range(len(self.behs))]  # 存放每种行为的交互数量
        # 先填充其他行为
        for i in range(len(self.behs) - 1):
            beh = self.behs[i]
            beh_filename = self.predir + '/' + beh + '.txt'
            with open(beh_filename) as f:
                for l in f.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, items_list = items[0], items[1:]
                    self.interNum[i] += len(items_list)
                    for item in items_list:
                        self.trnMats[i][uid, item] = 1.
                    self.trnDicts[i][uid] = items_list
        # 再填充目标行为
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]
                self.interNum[-1] += len(train_items)
                for i in train_items:
                    self.trnMats[-1][uid, i] = 1.  # 目标行为是trnmats中最后一个矩阵

                self.trnDicts[-1][uid] = train_items
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')]
                except Exception:
                    continue
                uid, test_items = items[0], items[1:]

                for i in test_items:
                    self.tstMats[uid, i] = 1.  # tstMat就是单独一个矩阵
                self.tstDicts[uid] = test_items
        self.print_statistics()
        ## 填充一些之后要用到的东西，懒得改名字
        self.train_items = self.trnDicts[-1]
        self.test_set = self.tstDicts
        self.path = self.predir

    def get_adj_mat(self):
        self.saveAdjMatPath = 'Adj_Mats/' + self.dataset_name
        os.makedirs(self.saveAdjMatPath, exist_ok=True)

        adj_mat_list = []  # 所有行为的adj mat
        norm_adj_mat_list = []  # 所有行为的norm adj mat
        mean_adj_mat_list = []  # 所有行为的mean adj mat
        pre_adj_mat_list = []  # 所有行为的pre adj mat
        try:
            t1 = time()
            for i in range(len(self.behs)):
                beh = self.behs[i]
                adj_mat = sp.load_npz(self.saveAdjMatPath + '/s_adj_mat_' + beh + '.npz')
                norm_adj_mat = sp.load_npz(self.saveAdjMatPath + '/s_norm_adj_mat_' + beh + '.npz')
                mean_adj_mat = sp.load_npz(self.saveAdjMatPath + '/s_mean_adj_mat_' + beh + '.npz')
                adj_mat_list.append(adj_mat)
                norm_adj_mat_list.append(norm_adj_mat)
                mean_adj_mat_list.append(mean_adj_mat)
            # # 加载目标行为
            # adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            # norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            # mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            # adj_mat_list.append(adj_mat)
            # norm_adj_mat_list.append(norm_adj_mat)
            # mean_adj_mat_list.append(mean_adj_mat)

            print('already load adj matrix', time() - t1)

        except Exception:
            for i in range(len(self.behs)):
                beh = self.behs[i]
                adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat(self.trnMats[i])
                adj_mat_list.append(adj_mat)
                norm_adj_mat_list.append(norm_adj_mat)
                mean_adj_mat_list.append(mean_adj_mat)
                sp.save_npz(self.saveAdjMatPath + '/s_adj_mat_' + beh + '.npz', adj_mat)
                sp.save_npz(self.saveAdjMatPath + '/s_norm_adj_mat_' + beh + '.npz', norm_adj_mat)
                sp.save_npz(self.saveAdjMatPath + '/s_mean_adj_mat_' + beh + '.npz', mean_adj_mat)

        try:
            for i in range(len(self.behs)):
                beh = self.behs[i]
                pre_adj_mat = sp.load_npz(self.saveAdjMatPath + '/s_pre_adj_mat_' + beh + '.npz')
                pre_adj_mat_list.append(pre_adj_mat)
            print('already load pre adj matrix')
        except Exception:
            for i in range(len(self.behs)):
                beh = self.behs[i]
                adj_mat = adj_mat_list[i]
                rowsum = np.array(adj_mat.sum(1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat_inv = sp.diags(d_inv)

                norm_adj = d_mat_inv.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat_inv)
                print('generate pre adjacency matrix.')
                pre_adj_mat = norm_adj.tocsr()  # D^-0.5 (A) D^0.5
                pre_adj_mat_list.append(pre_adj_mat)
                sp.save_npz(self.saveAdjMatPath + '/s_pre_adj_mat_' + beh + '.npz', norm_adj)

        return pre_adj_mat_list

    def create_adj_mat(self, which_R):
        """
        为当前行为的交互信息矩阵[n_user,n_item]，生成全图的邻接矩阵[n_user+n_item, n_user+n_item]
        :param which_R: 该行为的交互信息矩阵。dok_matrix
        :return:
        """
        # 1. 构造全图邻接稀疏矩阵 [n_user+n_item, n_user+n_item]
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)  # 全图邻接矩阵
        # 将dok_matrix转成lil_matrix再统一给矩阵赋值
        adj_mat = adj_mat.tolil()
        R = which_R.tolil()
        adj_mat[:self.n_users, self.n_users:] = R  # 矩阵右上角
        adj_mat[self.n_users:, :self.n_users] = R.T  # 矩阵左下角
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        # 2. 构造GCN的归一化邻接矩阵，
        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)  # 度矩阵

            norm_adj = d_mat_inv.dot(adj)  # 归一化：左乘度矩阵
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))  # 简单归一化：即D^-1(A+I)
        mean_adj_mat = normalized_adj_single(adj_mat)  # D^-1 A

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def get_similarity_score(self, beh_idx, sim_measure):
        # user_sim_file_name = "_".join(["user_sim_mat", str(beh_idx), sim_measure, ".npy"])
        # item_sim_file_name = "_".join(["item_sim_mat", str(beh_idx), sim_measure, ".npy"])

        user_sim_file_name = "_".join(["user_sim_sp", str(beh_idx), sim_measure, ".npz"])
        item_sim_file_name = "_".join(["item_sim_sp", str(beh_idx), sim_measure, ".npz"])
        try:
            t1 = time()
            # user_sim_mat = np.load(os.path.join(self.saveSimMatPath, user_sim_file_name))
            # item_sim_mat = np.load(os.path.join(self.saveSimMatPath, item_sim_file_name))

            user_sim_sp = sp.load_npz(os.path.join(self.saveSimMatPath, user_sim_file_name))
            item_sim_sp = sp.load_npz(os.path.join(self.saveSimMatPath, item_sim_file_name))
            print('user sim mat nonzero samples in beh{}: {}'.format(beh_idx, len(user_sim_sp.nonzero()[0])))
            print('item sim mat nonzero samples in beh{}: {}'.format(beh_idx, len(item_sim_sp.nonzero()[0])))
            print('already load sim matrix for beh_{}: {}'.format(beh_idx, time() - t1))
        except Exception:
            user_sim_mat, item_sim_mat = self.create_sim_mat(beh_idx, sim_measure)
            user_sim_sp = sp.csr_matrix(user_sim_mat)
            item_sim_sp = sp.csr_matrix(item_sim_mat)
            sp.save_npz(os.path.join(self.saveSimMatPath, user_sim_file_name), user_sim_sp)
            sp.save_npz(os.path.join(self.saveSimMatPath, item_sim_file_name), item_sim_sp)
            # np.save(os.path.join(self.saveSimMatPath, user_sim_file_name), user_sim_mat)
            # np.save(os.path.join(self.saveSimMatPath, item_sim_file_name), item_sim_mat)
        return user_sim_sp, item_sim_sp

    def get_similarity_score_old(self, beh_idx, sim_measure):
        user_sim_file_name = "_".join(["user_sim_mat", str(beh_idx), sim_measure, ".npy"])
        item_sim_file_name = "_".join(["item_sim_mat", str(beh_idx), sim_measure, ".npy"])
        try:
            t1 = time()
            user_sim_mat = np.load(os.path.join(self.saveSimMatPath, user_sim_file_name))
            item_sim_mat = np.load(os.path.join(self.saveSimMatPath, item_sim_file_name))
            print('user sim mat nonzero samples in beh{}: {}'.format(beh_idx, len(user_sim_mat.nonzero()[0])))
            print('item sim mat nonzero samples in beh{}: {}'.format(beh_idx, len(item_sim_mat.nonzero()[0])))
            print('already load sim matrix for beh_{}: {}'.format(beh_idx, time() - t1))
        except Exception:
            user_sim_mat, item_sim_mat = self.create_sim_mat(beh_idx, sim_measure)
            np.save(os.path.join(self.saveSimMatPath, user_sim_file_name), user_sim_mat)
            np.save(os.path.join(self.saveSimMatPath, item_sim_file_name), item_sim_mat)
        return user_sim_mat, item_sim_mat

    def get_occur_mat(self, beh_idx):
        user_occur_file_name = "_".join(["user_occur_mat", str(beh_idx), ".npy"])
        item_occur_file_name = "_".join(["item_occur_mat", str(beh_idx), ".npy"])
        try:
            t1 = time()
            user_occur_mat = np.load(os.path.join(self.saveSimMatPath, user_occur_file_name))
            item_occur_mat = np.load(os.path.join(self.saveSimMatPath, item_occur_file_name))
            print('user occur mat nonzero samples in beh{}: {}'.format(beh_idx, len(user_occur_mat.nonzero()[0])))
            print('item occur mat nonzero samples in beh{}: {}'.format(beh_idx, len(item_occur_mat.nonzero()[0])))
            print('already load occur matrix for beh{}: {}'.format(beh_idx, time() - t1))
        except Exception:
            """ 单进程 """
            print('Start calculating occur.')
            t1 = time()
            trn_mat = self.trnMats[beh_idx]
            trn_dict_item = {i: np.nonzero(row)[1].tolist() for i, row in enumerate(trn_mat.T)}
            trn_dict_user = {u: np.nonzero(row)[1].tolist() for u, row in enumerate(trn_mat)}
            user_occur_mat = np.zeros((self.n_users, self.n_users))
            # 倒排索引
            for item, users in tqdm.tqdm(trn_dict_item.items(), desc='user_occur'):
                for u in users:
                    for v in users:
                        if u == v: continue
                        # user_occur[u, v] += 1
                        user_occur_mat[u][v] += 1

            item_occur_mat = np.zeros((self.n_items, self.n_items))
            # 倒排索引
            for user, items in tqdm.tqdm(trn_dict_user.items(), desc='item_occur'):
                for i in items:
                    for j in items:
                        if i == j: continue
                        item_occur_mat[i][j] += 1

            # for (u, v) in user_pairs:
            #     result = list(combinations(set(trn_dict_user[u]) & set(trn_dict_user[v]), 2)).__len__()
            #     user_occur[u][v] = result
            #     user_occur[v][u] = result
            #
            # item_pairs = list(combinations(trn_dict_item.keys(), 2))
            # item_occur = np.zeros((self.n_items, self.n_items))
            # for (i, j) in item_pairs:
            #     result = list(combinations(set(trn_dict_item[i]) & set(trn_dict_item[j]), 2)).__len__()
            #     item_occur[i][j] = result
            #     item_occur[j][i] = result
            """ 多进程 """
            # user_occur_processor = Occur_MProcessor(trn_dict_user, self.n_users, 'user_occur')
            # user_occur_mat = user_occur_processor.run()
            #
            # item_occur_processor = Occur_MProcessor(trn_dict_item, self.n_items, 'item_occur')
            # item_occur_mat = item_occur_processor.run()

            print('already create occur matrix for beh_{}: {}'.format(beh_idx, time() - t1))
            np.save(os.path.join(self.saveSimMatPath, user_occur_file_name), user_occur_mat)
            np.save(os.path.join(self.saveSimMatPath, item_occur_file_name), item_occur_mat)

        return user_occur_mat, item_occur_mat

    def create_sim_mat(self, beh_idx, sim_measure):
        """
        为【当前行为】根据【当前相似度方式】计算user/item相似度
        :param beh_idx: 如0
        :param sim_measure: 如swing
        :return:
        """
        t1 = time()
        trn_mat = self.trnMats[beh_idx]
        trn_dict_item = {i: np.nonzero(row)[1].tolist() for i, row in enumerate(trn_mat.T)}
        trn_dict_user = {u: np.nonzero(row)[1].tolist() for u, row in enumerate(trn_mat)}

        user_sim_mat = np.zeros((self.n_users, self.n_users))
        item_sim_mat = np.zeros((self.n_items, self.n_items))
        # user_sim_mat = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)
        # item_sim_mat = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)
        """ 1. 获取预处理好的occur mat，减少计算量 """
        user_occur_mat, item_occur_mat = self.get_occur_mat(beh_idx)
        # user_pairs = list(combinations(trn_dict_user.keys(), 2))

        """ 2. 计算sim mat """

        """ 多进程 """
        # user_sim_processor = Sim_MProcessor(trn_dict_user, self.n_users, 'user_sim', user_occur_mat, item_occur_mat, 'user', sim_measure)
        # user_sim_mat = user_sim_processor.run()
        #
        # item_sim_processor = Sim_MProcessor(trn_dict_item, self.n_items, 'item_sim', user_occur_mat, item_occur_mat, 'item',
        #                                     sim_measure)
        # item_sim_mat = item_sim_processor.run()

        """ 单进程 """
        user_sim_file_name = "_".join(["user_sim_mat", str(beh_idx), sim_measure, ".npy"])
        item_sim_file_name = "_".join(["item_sim_mat", str(beh_idx), sim_measure, ".npy"])

        if sim_measure == 'pmi':
            for u in range(self.n_users):
                if u not in trn_dict_user or len(trn_dict_user[u]) == 0: continue
                p_u = len(trn_dict_user[u]) * 1. / self.n_users
                for v in range(u + 1, self.n_users):
                    if v not in trn_dict_user or len(trn_dict_user[v]) == 0: continue
                    p_v = len(trn_dict_user[v]) * 1. / self.n_users
                    # p_u_v = len(set(trn_dict_user[u]) & set(trn_dict_user[v])) * 1. / self.n_users
                    p_u_v = user_occur_mat[u][v] * 1. / self.n_users
                    if p_u_v == 0.: continue
                    user_sim_mat[u][v] = np.log(p_u_v / (p_u * p_v))
            user_sim_mat += user_sim_mat.T - np.diag(user_sim_mat.diagonal())
            # print(np.transpose(user_sim_mat) == user_sim_mat)

            for i in range(self.n_items):
                if i not in trn_dict_item or len(trn_dict_item[i]) == 0: continue
                p_i = len(trn_dict_item[i]) * 1. / self.n_items
                for j in range(i + 1, self.n_items):
                    if j not in trn_dict_item or len(trn_dict_item[j]) == 0: continue
                    p_j = len(trn_dict_item[j]) * 1. / self.n_items
                    # p_i_j = len(set(trn_dict_item[i]) & set(trn_dict_item[j])) * 1. / self.n_items
                    p_i_j = item_occur_mat[i][j] * 1. / self.n_items
                    if p_i_j == 0.: continue
                    item_sim_mat[i][j] = np.log(p_i_j / (p_i * p_j))
            item_sim_mat += item_sim_mat.T - np.diag(item_sim_mat.diagonal())

        elif sim_measure == 'swing':
            alpha = 0.5

            for u in tqdm.tqdm(range(self.n_users), desc='sim on current users'):
            # for u in tqdm.tqdm(range(10), desc='sim on current users'):
                for v in range(self.n_users):
                    if u == v: continue
                    item_cur_pairs = list(combinations(set(trn_dict_user[u]) & set(trn_dict_user[v]), 2))
                    result = 0.0
                    for (i, j) in item_cur_pairs:
                        result += 1. / (alpha + item_occur_mat[i][j])
                    user_sim_mat[u][v] = result
            # print('User sim mat: ', user_sim_mat == user_sim_mat.T)  # for debug  user_sim_file_name), user_sim_mat)

            for i in tqdm.tqdm(range(self.n_items), desc='sim on current items'):
                for j in range(self.n_items):
                    if i == j: continue
                    user_cur_pairs = list(combinations(set(trn_dict_item[i]) & set(trn_dict_item[j]), 2))
                    result = 0.0
                    for (u, v) in user_cur_pairs:
                        result += 1. / (alpha + user_occur_mat[u][v])
                    item_sim_mat[i][j] = result

            # user_pairs = list(combinations(trn_dict_user.keys()[:100], 2))
            # item_pairs = list(combinations(trn_dict_item.keys()[:100], 2))
            #
            # print("user pairs length：{}".format(len(user_pairs)))
            # for (u, v) in user_pairs:
            #     item_cur_pairs = list(combinations(set(trn_dict_user[u]) & set(trn_dict_user[v]), 2))
            #     result = 0.0
            #     for (i, j) in item_cur_pairs:
            #         result += 1. / (alpha + item_occur_mat[i][j])
            #     user_sim_mat[u][v] = result
            # user_sim_mat += user_sim_mat.T - np.diag(user_sim_mat.diagonal())
            #
            #
            # print("item pairs length：{}".format(len(item_pairs)))
            # for (i, j) in item_cur_pairs:
            #     user_cur_pairs = list(combinations(set(trn_dict_item[i]) & set(trn_dict_item[j]), 2))
            #     result = 0.0
            #     for (u, v) in user_cur_pairs:
            #         result += 1. / (alpha + user_occur_mat[u][v])
            #     item_sim_mat[i][j] = result
            # item_sim_mat += item_sim_mat.T - np.diag(item_sim_mat.diagonal())

        print('{} finish create sim mat: {}'.format(sim_measure, time() - t1))
        return user_sim_mat, item_sim_mat

    def get_sim_topk(self, beh_idx, sim_measure, topk_user, topk_item):
        user_topk_sim_file = "_".join(["user_sim_topk", str(beh_idx), sim_measure, str(topk_user), ".pth"])
        item_topk_sim_file = "_".join(["user_sim_topk", str(beh_idx), sim_measure, str(topk_item), ".pth"])
        try:
            t1 = time()
            user_topk_indices = torch.load(os.path.join(self.saveSimMatPath, user_topk_sim_file))
            item_topk_indices = torch.load(os.path.join(self.saveSimMatPath, item_topk_sim_file))
            print('already load topk sim for beh_{}: {}'.format(beh_idx, time() - t1))
        except Exception:
            t1 = time()
            user_sim_mat, item_sim_mat = self.get_similarity_score(beh_idx, sim_measure)
            user_sim_mat = torch.tensor(user_sim_mat)
            item_sim_mat = torch.tensor(item_sim_mat)
            user_topk_indices = torch.topk(user_sim_mat, min(topk_user, self.n_users), dim=1, largest=True)[1]  # [N, topk]
            item_topk_indices = torch.topk(item_sim_mat, min(topk_item, self.n_items), dim=1, largest=True)[1]  # [N, topk]
            torch.save(user_topk_indices, user_topk_sim_file)
            torch.save(item_topk_indices, item_topk_sim_file)
            print('finish create topk sim for beh_{}: {}'.format(beh_idx, time() - t1))
        return user_topk_indices, item_topk_indices

    def trans_sim_sp(self, beh_idx, sim_measure):
        """
        把user_sim_mat_0_swing_.npy转化成稀疏矩阵存储
        :param beh_idx:
        :param sim_measure:
        :return:
        """
        user_sim_file_name = "_".join(["user_sim_mat", str(beh_idx), sim_measure, ".npy"])
        item_sim_file_name = "_".join(["item_sim_mat", str(beh_idx), sim_measure, ".npy"])
        try:
            t1 = time()
            user_sim_mat = np.load(os.path.join(self.saveSimMatPath, user_sim_file_name))
            item_sim_mat = np.load(os.path.join(self.saveSimMatPath, item_sim_file_name))
            print('already load sim matrix for beh_{}: {}'.format(beh_idx, time() - t1))

            user_sim_sp = sp.csr_matrix(user_sim_mat)
            item_sim_sp = sp.csr_matrix(item_sim_mat)

            sp.save_npz(os.path.join(self.saveSimMatPath, "_".join(["user_sim_sp", str(beh_idx), sim_measure, ".npz"])), user_sim_sp)
            sp.save_npz(
                os.path.join(self.saveSimMatPath, "_".join(["item_sim_sp", str(beh_idx), sim_measure, ".npz"])),
                item_sim_sp)
            print('already transform sim matrix to sp for beh_{}: {}'.format(beh_idx, time() - t1))
            # return user_sim_sp, item_sim_sp
        except Exception:
            print('Please create sim mat first.')

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u] + self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        num_users, num_items = self.n_users, self.n_items
        num_ratings = sum(self.interNum)
        density = 1.0*num_ratings/(num_users*num_items)
        sparsity = 1 - density
        data_info = ["Dataset name: %s" % self.dataset_name,
                     "The number of users: %d" % num_users,
                     "The number of items: %d" % num_items,
                     "The behavior ratings: {}".format(self.interNum),
                     "The number of ratings: %d" % num_ratings,
                     "Average actions of users: %.2f" % (1.0*num_ratings/num_users),
                     "Average actions of items: %.2f" % (1.0*num_ratings/num_items),
                     "The density of the dataset: %.6f" % (density),
                     "The sparsity of the dataset: %.6f%%" % (sparsity * 100)]
        data_info = "\n".join(data_info)
        print(data_info)

        # print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        # print('n_interactions=%d' % (self.n_train + self.n_test))
        # print('n_train=%d, n_test=%d, sparsity=%.5f' % (
        # self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        """
        根据交互数量将所有user分成若干cluster，即流行度分析。
        :return: split_state:统计信息[]
        split_uids [[temp0], [temp1], ..]  temp0: user list[u1, u2..]
        """
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)

        split_uids = list()  # [[temp0], [temp1], ..]  temp0: user list[u1, u2..]

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train)
        n_rates = 0

        split_state = []
        temp0 = []
        temp1 = []
        temp2 = []
        temp3 = []
        temp4 = []

        # print user_n_iid

        for idx, n_iids in enumerate(sorted(user_n_iid)):
            if n_iids < 9:
                temp0 += user_n_iid[n_iids]
            elif n_iids < 13:
                temp1 += user_n_iid[n_iids]
            elif n_iids < 17:
                temp2 += user_n_iid[n_iids]
            elif n_iids < 20:
                temp3 += user_n_iid[n_iids]
            else:
                temp4 += user_n_iid[n_iids]

        split_uids.append(temp0)
        split_uids.append(temp1)
        split_uids.append(temp2)
        split_uids.append(temp3)
        split_uids.append(temp4)
        split_state.append("#users=[%d]" % (len(temp0)))
        split_state.append("#users=[%d]" % (len(temp1)))
        split_state.append("#users=[%d]" % (len(temp2)))
        split_state.append("#users=[%d]" % (len(temp3)))
        split_state.append("#users=[%d]" % (len(temp4)))

        return split_uids, split_state

    def create_sparsity_split2(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state


# class SBDataHandler(object):
#     def __init__(self, path, batch_size):
#         self.path = path
#         self.batch_size = batch_size
#
#         train_file = path + '/train_all.txt'
#         test_file = path + '/test.txt'
#
#         pv_file = path + '/pv.txt'
#         cart_file = path + '/cart.txt'
#
#         # get number of users and items
#         self.n_users, self.n_items = 0, 0  # train.txt+test.txt中的用户数目和item数目
#         self.n_train, self.n_test = 0, 0  # 代表train.txt和test.txt中的交互信息数
#         self.neg_pools = {}
#
#         self.exist_users = []  # train.txt中的用户，即目标行为中的用户
#
#         with open(train_file) as f:
#             for l in f.readlines():
#                 if len(l) > 0:
#                     l = l.strip('\n').split(' ')
#                     items = [int(i) for i in l[1:]]
#                     uid = int(l[0])
#                     self.exist_users.append(uid)
#                     self.n_items = max(self.n_items, max(items))
#                     self.n_users = max(self.n_users, uid)
#                     self.n_train += len(items)
#
#         with open(test_file) as f:
#             for l in f.readlines():
#                 if len(l) > 0:
#                     l = l.strip('\n')
#                     try:
#                         items = [int(i) for i in l.split(' ')[1:]]
#                     except Exception:
#                         continue
#                     self.n_items = max(self.n_items, max(items))
#                     self.n_test += len(items)
#         self.n_items += 1
#         self.n_users += 1
#
#         self.print_statistics()
#         # dok_matrix: dictionary of keys. 适用于递增的矩阵，之后需要填充该矩阵
#         # 1. 交互信息邻接矩阵形式，为每个行为都初始化一个稀疏矩阵，之后慢慢填充
#         self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # target beh
#         self.R_pv = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # pv heh
#         self.R_cart = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # cart beh
#         # 2. 交互信息邻接链表形式，每个行为的交互信息都是一个字典。key为uid, value为item list[]
#         self.train_items, self.test_set = {}, {}
#         self.pv_set, self.cart_set = {}, {}
#         with open(train_file) as f_train:
#             with open(test_file) as f_test:
#                 for l in f_train.readlines():
#                     if len(l) == 0: break
#                     l = l.strip('\n')
#                     items = [int(i) for i in l.split(' ')]
#                     uid, train_items = items[0], items[1:]
#
#                     for i in train_items:
#                         self.R[uid, i] = 1.  # 填充矩阵
#
#                     self.train_items[uid] = train_items
#
#                 for l in f_test.readlines():
#                     if len(l) == 0: break
#                     l = l.strip('\n')
#                     try:
#                         items = [int(i) for i in l.split(' ')]
#                     except Exception:
#                         continue
#
#                     uid, test_items = items[0], items[1:]
#                     self.test_set[uid] = test_items
#         with open(pv_file) as f_pv:
#             for l in f_pv.readlines():
#                 if len(l) == 0: break
#                 l = l.strip('\n')
#                 items = [int(i) for i in l.split(' ')]
#                 uid, pv_items = items[0], items[1:]
#
#                 for i in pv_items:
#                     self.R_pv[uid, i] = 1.
#                 self.pv_set[uid] = pv_items
#
#         with open(cart_file) as f_cart:
#             for l in f_cart.readlines():
#                 if len(l) == 0: break
#                 l = l.strip('\n')
#                 items = [int(i) for i in l.split(' ')]
#                 uid, cart_items = items[0], items[1:]
#
#                 for i in cart_items:
#                     self.R_cart[uid, i] = 1.
#                 self.cart_set[uid] = cart_items
#
#     def get_adj_mat(self):
#         try:
#             t1 = time()
#             adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
#             norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
#             mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
#
#             adj_mat_pv = sp.load_npz(self.path + '/s_adj_mat_pv.npz')
#             norm_adj_mat_pv = sp.load_npz(self.path + '/s_norm_adj_mat_pv.npz')
#             mean_adj_mat_pv = sp.load_npz(self.path + '/s_mean_adj_mat_pv.npz')
#
#             adj_mat_cart = sp.load_npz(self.path + '/s_adj_mat_cart.npz')
#             norm_adj_mat_cart = sp.load_npz(self.path + '/s_norm_adj_mat_cart.npz')
#             mean_adj_mat_cart = sp.load_npz(self.path + '/s_mean_adj_mat_cart.npz')
#
#             print('already load adj matrix', adj_mat.shape, time() - t1)
#
#         except Exception:
#             adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat(self.R)
#             adj_mat_pv, norm_adj_mat_pv, mean_adj_mat_pv = self.create_adj_mat(self.R_pv)
#             adj_mat_cart, norm_adj_mat_cart, mean_adj_mat_cart = self.create_adj_mat(self.R_cart)
#             sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
#             sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
#             sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
#
#             sp.save_npz(self.path + '/s_adj_mat_pv.npz', adj_mat_pv)
#             sp.save_npz(self.path + '/s_norm_adj_mat_pv.npz', norm_adj_mat_pv)
#             sp.save_npz(self.path + '/s_mean_adj_mat_pv.npz', mean_adj_mat_pv)
#
#             sp.save_npz(self.path + '/s_adj_mat_cart.npz', adj_mat_cart)
#             sp.save_npz(self.path + '/s_norm_adj_mat_cart.npz', norm_adj_mat_cart)
#             sp.save_npz(self.path + '/s_mean_adj_mat_cart.npz', mean_adj_mat_cart)
#
#         try:
#             pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
#             pre_adj_mat_pv = sp.load_npz(self.path + '/s_pre_adj_mat_pv.npz')
#             pre_adj_mat_cart = sp.load_npz(self.path + '/s_pre_adj_mat_cart.npz')
#         except Exception:
#
#             rowsum = np.array(adj_mat.sum(1))
#             d_inv = np.power(rowsum, -0.5).flatten()
#             d_inv[np.isinf(d_inv)] = 0.
#             d_mat_inv = sp.diags(d_inv)
#
#             norm_adj = d_mat_inv.dot(adj_mat)
#             norm_adj = norm_adj.dot(d_mat_inv)
#             print('generate pre adjacency matrix.')
#             pre_adj_mat = norm_adj.tocsr()
#             sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
#
#             rowsum = np.array(adj_mat_pv.sum(1))
#             d_inv = np.power(rowsum, -0.5).flatten()
#             d_inv[np.isinf(d_inv)] = 0.
#             d_mat_inv = sp.diags(d_inv)
#
#             norm_adj = d_mat_inv.dot(adj_mat_pv)
#             norm_adj = norm_adj.dot(d_mat_inv)
#             print('generate pre view adjacency matrix.')
#             pre_adj_mat_pv = norm_adj.tocsr()
#             sp.save_npz(self.path + '/s_pre_adj_mat_pv.npz', norm_adj)
#
#             rowsum = np.array(adj_mat_cart.sum(1))
#             d_inv = np.power(rowsum, -0.5).flatten()
#             d_inv[np.isinf(d_inv)] = 0.
#             d_mat_inv = sp.diags(d_inv)
#
#             norm_adj = d_mat_inv.dot(adj_mat_cart)
#             norm_adj = norm_adj.dot(d_mat_inv)
#             print('generate pre view adjacency matrix.')
#             pre_adj_mat_cart = norm_adj.tocsr()
#             sp.save_npz(self.path + '/s_pre_adj_mat_cart.npz', norm_adj)
#
#         return pre_adj_mat, pre_adj_mat_pv, pre_adj_mat_cart
#
#     def create_adj_mat(self, which_R):
#         """
#         为当前行为的交互信息矩阵[n_user,n_item]，生成全图的邻接矩阵[n_user+n_item, n_user+n_item]
#         :param which_R: 该行为的交互信息矩阵。dok_matrix
#         :return:
#         """
#         # 1. 构造全图邻接稀疏矩阵 [n_user+n_item, n_user+n_item]
#         t1 = time()
#         adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)  # 全图邻接矩阵
#         # 将dok_matrix转成lil_matrix再统一给矩阵赋值
#         adj_mat = adj_mat.tolil()
#         R = which_R.tolil()
#         adj_mat[:self.n_users, self.n_users:] = R  # 矩阵右上角
#         adj_mat[self.n_users:, :self.n_users] = R.T  # 矩阵左下角
#         adj_mat = adj_mat.todok()
#         print('already create adjacency matrix', adj_mat.shape, time() - t1)
#
#         # 2. 构造GCN的归一化邻接矩阵，这里使用简单归一化：即D^-1(A+I)
#         t2 = time()
#
#         def normalized_adj_single(adj):
#             rowsum = np.array(adj.sum(1))
#
#             d_inv = np.power(rowsum, -1).flatten()
#             d_inv[np.isinf(d_inv)] = 0.
#             d_mat_inv = sp.diags(d_inv)  # 度矩阵
#
#             norm_adj = d_mat_inv.dot(adj)  # 归一化：左乘度矩阵
#             # norm_adj = adj.dot(d_mat_inv)
#             print('generate single-normalized adjacency matrix.')
#             return norm_adj.tocoo()
#
#         def check_adj_if_equal(adj):
#             dense_A = np.array(adj.todense())
#             degree = np.sum(dense_A, axis=1, keepdims=False)
#
#             temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
#             print('check normalized adjacency matrix whether equal to this laplacian matrix.')
#             return temp
#
#         norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
#         mean_adj_mat = normalized_adj_single(adj_mat)
#
#         print('already normalize adjacency matrix', time() - t2)
#         return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
#
#     def negative_pool(self):
#         t1 = time()
#         for u in self.train_items.keys():
#             neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
#             pools = [rd.choice(neg_items) for _ in range(100)]
#             self.neg_pools[u] = pools
#         print('refresh negative pools', time() - t1)
#
#     def sample(self):
#         if self.batch_size <= self.n_users:
#             users = rd.sample(self.exist_users, self.batch_size)
#         else:
#             users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
#
#         def sample_pos_items_for_u(u, num):
#             pos_items = self.train_items[u]
#             n_pos_items = len(pos_items)
#             pos_batch = []
#             while True:
#                 if len(pos_batch) == num: break
#                 pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
#                 pos_i_id = pos_items[pos_id]
#
#                 if pos_i_id not in pos_batch:
#                     pos_batch.append(pos_i_id)
#             return pos_batch
#
#         def sample_neg_items_for_u(u, num):
#             neg_items = []
#             while True:
#                 if len(neg_items) == num: break
#                 neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
#                 if neg_id not in self.train_items[u] and neg_id not in neg_items:
#                     neg_items.append(neg_id)
#             return neg_items
#
#         def sample_neg_items_for_u_from_pools(u, num):
#             neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
#             return rd.sample(neg_items, num)
#
#         pos_items, neg_items = [], []
#         for u in users:
#             pos_items += sample_pos_items_for_u(u, 1)
#             neg_items += sample_neg_items_for_u(u, 1)
#
#         return users, pos_items, neg_items
#
#     def sample_test(self):
#         if self.batch_size <= self.n_users:
#             users = rd.sample(self.test_set.keys(), self.batch_size)
#         else:
#             users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
#
#         def sample_pos_items_for_u(u, num):
#             pos_items = self.test_set[u]
#             n_pos_items = len(pos_items)
#             pos_batch = []
#             while True:
#                 if len(pos_batch) == num: break
#                 pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
#                 pos_i_id = pos_items[pos_id]
#
#                 if pos_i_id not in pos_batch:
#                     pos_batch.append(pos_i_id)
#             return pos_batch
#
#         def sample_neg_items_for_u(u, num):
#             neg_items = []
#             while True:
#                 if len(neg_items) == num: break
#                 neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
#                 if neg_id not in (self.test_set[u] + self.train_items[u]) and neg_id not in neg_items:
#                     neg_items.append(neg_id)
#             return neg_items
#
#         def sample_neg_items_for_u_from_pools(u, num):
#             neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
#             return rd.sample(neg_items, num)
#
#         pos_items, neg_items = [], []
#         for u in users:
#             pos_items += sample_pos_items_for_u(u, 1)
#             neg_items += sample_neg_items_for_u(u, 1)
#
#         return users, pos_items, neg_items
#
#     def get_num_users_items(self):
#         return self.n_users, self.n_items
#
#     def print_statistics(self):
#         print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
#         print('n_interactions=%d' % (self.n_train + self.n_test))
#         print('n_train=%d, n_test=%d, sparsity=%.5f' % (
#         self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))
#
#     def get_sparsity_split(self):
#         try:
#             split_uids, split_state = [], []
#             lines = open(self.path + '/sparsity.split', 'r').readlines()
#
#             for idx, line in enumerate(lines):
#                 if idx % 2 == 0:
#                     split_state.append(line.strip())
#                     print(line.strip())
#                 else:
#                     split_uids.append([int(uid) for uid in line.strip().split(' ')])
#             print('get sparsity split.')
#
#         except Exception:
#             split_uids, split_state = self.create_sparsity_split()
#             f = open(self.path + '/sparsity.split', 'w')
#             for idx in range(len(split_state)):
#                 f.write(split_state[idx] + '\n')
#                 f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
#             print('create sparsity split.')
#
#         return split_uids, split_state
#
#     def create_sparsity_split(self):
#         all_users_to_test = list(self.test_set.keys())
#         user_n_iid = dict()
#
#         # generate a dictionary to store (key=n_iids, value=a list of uid).
#         for uid in all_users_to_test:
#             train_iids = self.train_items[uid]
#             test_iids = self.test_set[uid]
#
#             n_iids = len(train_iids)
#
#             if n_iids not in user_n_iid.keys():
#                 user_n_iid[n_iids] = [uid]
#             else:
#                 user_n_iid[n_iids].append(uid)
#         split_uids = list()
#
#         # split the whole user set into four subset.
#         temp = []
#         count = 1
#         fold = 4
#         n_count = (self.n_train)
#         n_rates = 0
#
#         split_state = []
#         temp0 = []
#         temp1 = []
#         temp2 = []
#         temp3 = []
#         temp4 = []
#
#         # print user_n_iid
#
#         for idx, n_iids in enumerate(sorted(user_n_iid)):
#             if n_iids < 9:
#                 temp0 += user_n_iid[n_iids]
#             elif n_iids < 13:
#                 temp1 += user_n_iid[n_iids]
#             elif n_iids < 17:
#                 temp2 += user_n_iid[n_iids]
#             elif n_iids < 20:
#                 temp3 += user_n_iid[n_iids]
#             else:
#                 temp4 += user_n_iid[n_iids]
#
#         split_uids.append(temp0)
#         split_uids.append(temp1)
#         split_uids.append(temp2)
#         split_uids.append(temp3)
#         split_uids.append(temp4)
#         split_state.append("#users=[%d]" % (len(temp0)))
#         split_state.append("#users=[%d]" % (len(temp1)))
#         split_state.append("#users=[%d]" % (len(temp2)))
#         split_state.append("#users=[%d]" % (len(temp3)))
#         split_state.append("#users=[%d]" % (len(temp4)))
#
#         return split_uids, split_state
#
#     def create_sparsity_split2(self):
#         all_users_to_test = list(self.test_set.keys())
#         user_n_iid = dict()
#
#         # generate a dictionary to store (key=n_iids, value=a list of uid).
#         for uid in all_users_to_test:
#             train_iids = self.train_items[uid]
#             test_iids = self.test_set[uid]
#
#             n_iids = len(train_iids) + len(test_iids)
#
#             if n_iids not in user_n_iid.keys():
#                 user_n_iid[n_iids] = [uid]
#             else:
#                 user_n_iid[n_iids].append(uid)
#         split_uids = list()
#
#         # split the whole user set into four subset.
#         temp = []
#         count = 1
#         fold = 4
#         n_count = (self.n_train + self.n_test)
#         n_rates = 0
#
#         split_state = []
#         for idx, n_iids in enumerate(sorted(user_n_iid)):
#             temp += user_n_iid[n_iids]
#             n_rates += n_iids * len(user_n_iid[n_iids])
#             n_count -= n_iids * len(user_n_iid[n_iids])
#
#             if n_rates >= count * 0.25 * (self.n_train + self.n_test):
#                 split_uids.append(temp)
#
#                 state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
#                 split_state.append(state)
#                 print(state)
#
#                 temp = []
#                 n_rates = 0
#                 fold -= 1
#
#             if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
#                 split_uids.append(temp)
#
#                 state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
#                 split_state.append(state)
#                 print(state)
#
#         return split_uids, split_state


class DataChecker(object):
    def __init__(self, dataset):
        self.dataset_name = dataset
        self.root_dir = os.path.join(os.environ['HOME'], 'xjc/SGLMB/data')
        self.predir = os.path.join(self.root_dir, self.dataset_name)

        if self.dataset_name.find('Beibei') != -1:
            self.behs = ['pv', 'cart', 'train', 'test']
        elif self.dataset_name.find('Taobao') != -1:
            self.behs = ['pv', 'cart', 'train', 'test']
        elif self.dataset_name.find('yelp') != -1:
            if self.dataset_name.find('notip') != -1:
                self.behs = ['neg', 'neutral', 'train', 'test']
            else:
                self.behs = ['tip', 'neg', 'neutral', 'train', 'test']
        elif self.dataset_name.find('ML10M') != -1:
            self.behs = ['neg', 'neutral', 'train', 'test']
        elif self.dataset_name.find('kwai') != -1:
            self.behs = ['click', 'like', 'comment', 'train', 'test']

        self.num_users, self.num_items = 0, 0
        self.num_ratings = 0

        # self.showSingleBeh()
        # self.showDataset()

    def showSingleBeh(self):
        # 1. 首先加载所有.txt得到user num和 item num，包括train和test.txt
        for i in range(len(self.behs)):
            beh = self.behs[i]
            file_name = self.predir + '/' + beh + '.txt'
            with open(file_name) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        self.num_items = max(self.num_items, max(items))
                        self.num_users = max(self.num_users, uid)

        self.num_items += 1
        self.num_users += 1

        # all_user_list = []
        # all_item_list = []
        # with open(self.predir + '/train.txt') as f:
        #     for l in f.readlines():
        #         if len(l) > 0:
        #             l = l.strip('\n').split(' ')
        #             items = [int(i) for i in l[1:]]
        #             uid = int(l[0])
        #             for item in items:
        #                 all_user_list.append(uid)
        #                 all_item_list.append(item)
        #
        # with open(self.predir + '/test.txt') as f:
        #     for l in f.readlines():
        #         if len(l) > 0:
        #             l = l.strip('\n').split(' ')
        #             items = [int(i) for i in l[1:]]
        #             uid = int(l[0])
        #             for item in items:
        #                 all_user_list.append(uid)
        #                 all_item_list.append(item)
        #
        # self.num_users = max(all_user_list) + 1
        # self.num_items = max(all_item_list) + 1
        print(f'Dataset {self.dataset_name}, User num: {self.num_users}, Item num: {self.num_items}')

        # 2. 再加载其他行为打印统计数据，包括train.txt
        user_set_behs, item_set_behs = [], []
        for i in range(len(self.behs) - 1):
            beh = self.behs[i]
            file_name = self.predir + '/' + beh + '.txt'
            cur_user_list, cur_item_list = [], []
            cur_interactions = 0
            with open(file_name) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        cur_interactions += len(items)
                        for item in items:
                            cur_user_list.append(uid)
                            cur_item_list.append(item)
            self.num_ratings += cur_interactions
            cur_user_set = set(cur_user_list)
            cur_item_set = set(cur_item_list)
            user_set_behs.append(cur_user_set)
            item_set_behs.append(cur_item_set)
            user_num = len(cur_user_set)
            item_num = len(cur_item_set)
            left_users = set(range(self.num_users)) - cur_user_set  # 丢失的user
            left_items = set(range(self.num_items)) - cur_item_set
            print(f'{beh}.txt, User num: {user_num}, Item num: {item_num}, Interactions: {cur_interactions}, '
                  f'Left users num: {len(list(left_users))}, Left items num: {len(list(left_items))}')

        return user_set_behs, item_set_behs


    def showDataset(self):
        num_users, num_items = self.num_users, self.num_items
        num_ratings = self.num_ratings
        density = 1.0*num_ratings/(num_users*num_items)
        sparsity = 1 - density
        data_info = ["Dataset name: %s" % self.dataset_name,
                     "The number of users: %d" % self.num_users,
                     "The number of items: %d" % self.num_items,
                     "The number of ratings: %d" % self.num_ratings,
                     "Average actions of users: %.2f" % (1.0*num_ratings/num_users),
                     "Average actions of items: %.2f" % (1.0*num_ratings/num_items),
                     "The density of the dataset: %.6f" % (density),
                     "The sparsity of the dataset: %.6f%%" % (sparsity * 100)]
        data_info = "\n".join(data_info)
        print(data_info)


class DataSimChecker(DataHandler):
    def __init__(self, args, dataset, batch_size):
        super().__init__(dataset, batch_size)
        self.sim_weight = eval(args.sim_weights)
        self.sim_measure = args.sim_measure
        self.topk1_user = args.topk1_user
        self.topk1_item = args.topk1_item

    def unified_sim(self):
        user_sim_mat, item_sim_mat = self.get_similarity_score(beh_idx=0, sim_measure=self.sim_measure)
        user_sim_mat_unified, item_sim_mat_unified = self.sim_weight[0] * user_sim_mat, self.sim_weight[0] * item_sim_mat
        # del user_sim_mat, item_sim_mat
        for beh in range(1, len(self.behs)):
            user_sim_mat, item_sim_mat = self.get_similarity_score(beh_idx=beh, sim_measure=self.sim_measure)
            user_sim_mat_unified += self.sim_weight[beh] * user_sim_mat
            item_sim_mat_unified += self.sim_weight[beh] * item_sim_mat
            # del user_sim_mat, item_sim_mat
        return torch.tensor(user_sim_mat_unified.todense()), torch.tensor(item_sim_mat_unified.todense())

    def preprocess_sim_v3(self):
        t1 = time()
        input_u_sim, input_i_sim = self.unified_sim()

        user_topk_values1 = torch.topk(input_u_sim, min(self.topk1_user, self.n_users), dim=1, largest=True)[
            0][..., -1, None]  # 每行的第topk个值 [N, ]
        user_indices_remove = input_u_sim > user_topk_values1[..., -1, None]  # [N, N] True代表需要remove
        print('user topk 为0值的个数：', sum(user_topk_values1[..., -1, None] == 0.))
        # [N, topk]  注意这里是严格保存topk个，和原来的实现方式根据值去选择有点不同——所以可能会造成结果有点不一致

        item_topk_values1 = torch.topk(input_i_sim, min(self.topk1_item, self.n_items), dim=1, largest=True)[
            0]  # [N, topk]
        item_indices_remove = input_i_sim > item_topk_values1[..., -1, None]  # [N, N] True代表需要remove
        print('item topk 为0值的个数：', sum(item_topk_values1[..., -1, None] == 0.))
        item_indices_token = torch.tensor([False] * item_indices_remove.shape[0], dtype=torch.bool).reshape(-1, 1)
        item_indices_remove = torch.cat([item_indices_remove, item_indices_token], dim=1)

        print('preprocess sim v3:', time() - t1)
        return user_indices_remove, item_indices_remove


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Load Data to create sim mat.")
    parser.add_argument('--dataset', nargs='?', default='Beibei',
                        help='Choose a dataset from {Beibei,Taobao}')
    parser.add_argument('--sim_measure', type=str, default='swing')  # 相似度方法 pmi / swing
    parser.add_argument('--sim_weights', nargs='?', default='[1./3,1./3,1./3]')  # 每个行为的相似度权重

    parser.add_argument('--topk1_user', type=int, default=10)  #
    parser.add_argument('--topk1_item', type=int, default=10)  #

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    args = parser.parse_args()

    # python load_data.py --dataset Beibei --sim_weights [0.,0.,1.] --topk1_user 10 --topk1_item 10
    simchecker = DataSimChecker(args, args.dataset, args.batch_size)
    simchecker.preprocess_sim_v3()

    # for i in range(len(data_generator.behs)):
    #     # data_generator.get_occur_mat(i)
    #     data_generator.get_similarity_score(i, 'swing')
        # data_generator.get_similarity_score_old(i, "pmi")
        # data_generator.trans_sim_sp(i)