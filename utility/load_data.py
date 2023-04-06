'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import os


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

        self.predir = 'dataset/' + self.dataset_name
        self.behs = behs
        self.beh_num = len(behs)

        self.trnMats = None
        self.tstMats = None
        self.trnDicts = None
        self.trnDicts_item = None
        self.tstDicts = None

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        self.LoadData()

        self.saveSimMatPath = 'Sim_Mats/' + self.dataset_name
        os.makedirs(self.saveSimMatPath, exist_ok=True)

    def LoadData(self):
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

        self.trnMats = [sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) for i in
                        range(len(self.behs))]
        self.tstMats = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.trnDicts = [dict() for i in range(len(self.behs))]
        self.trnDicts_item = [dict() for i in range(len(self.behs))]
        self.tstDicts = dict()
        self.interNum = [0 for i in range(len(self.behs))]
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
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]
                self.interNum[-1] += len(train_items)
                for i in train_items:
                    self.trnMats[-1][uid, i] = 1.

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
                    self.tstMats[uid, i] = 1.
                self.tstDicts[uid] = test_items
        self.print_statistics()
        self.train_items = self.trnDicts[-1]
        self.test_set = self.tstDicts
        self.path = self.predir

    def get_adj_mat(self):
        self.saveAdjMatPath = 'Adj_Mats/' + self.dataset_name
        os.makedirs(self.saveAdjMatPath, exist_ok=True)

        adj_mat_list = []
        norm_adj_mat_list = []
        mean_adj_mat_list = []
        pre_adj_mat_list = []
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
                pre_adj_mat = norm_adj.tocsr()
                pre_adj_mat_list.append(pre_adj_mat)
                sp.save_npz(self.saveAdjMatPath + '/s_pre_adj_mat_' + beh + '.npz', norm_adj)

        return pre_adj_mat_list

    def create_adj_mat(self, which_R):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)  # 全图邻接矩阵
        adj_mat = adj_mat.tolil()
        R = which_R.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def get_unified_sim(self, sim_measure):
        user_unified_sim_file_name = "_".join(["user_unified_sim_mat", sim_measure, ".npz"])
        item_unified_sim_file_name = "_".join(["item_unified_sim_mat", sim_measure, ".npz"])
        try:
            t1 = time()
            user_unified_sim_mat = sp.load_npz(os.path.join(self.saveSimMatPath, user_unified_sim_file_name))
            item_unified_sim_mat = sp.load_npz(os.path.join(self.saveSimMatPath, item_unified_sim_file_name))
            print('already load unified sim'.format(time() - t1))
        except Exception:
            print('No Unified Sim File!')
        return user_unified_sim_mat, item_unified_sim_mat

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
        density = 1.0 * num_ratings / (num_users * num_items)
        sparsity = 1 - density
        data_info = ["Dataset name: %s" % self.dataset_name,
                     "The number of users: %d" % num_users,
                     "The number of items: %d" % num_items,
                     "The behavior ratings: {}".format(self.interNum),
                     "The number of ratings: %d" % num_ratings,
                     "Average actions of users: %.2f" % (1.0 * num_ratings / num_users),
                     "Average actions of items: %.2f" % (1.0 * num_ratings / num_items),
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
