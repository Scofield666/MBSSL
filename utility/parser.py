'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run GHCF.")

    # ******************************   optimizer paras      ***************************** #
    parser.add_argument('--lr', type=float, default=0.001,   #common parameter
                        help='Learning rate.')
    parser.add_argument('--lr_decay', action="store_true")  # default false --lr_decay
    parser.add_argument('--lr_decay_step', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=0.8)
    parser.add_argument('--meta_r', type=float, default=0.7)
    parser.add_argument('--meta_b', type=float, default=0.9)

    parser.add_argument('--test_epoch', type=int, default=5,
                        help='test epoch steps.')
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='Beibei',
                        help='Choose a dataset from {Beibei,Taobao}')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--is_norm', type=int, default=1,
                    help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,   #common parameter
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64,64]', #common parameter
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')

    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Gpu id')

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[10, 50]',
                        help='K for Top-K list')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')


    parser.add_argument('--aux_beh_idx', nargs='?', default='[0,1]')

    # ******************************   ssl inner loss  paras      ***************************** #
    parser.add_argument('--aug_type', type=int, default=0)
    parser.add_argument('--ssl_ratio', type=float, default=0.5)
    parser.add_argument('--ssl_temp', type=float, default=0.5)
    parser.add_argument('--ssl_reg', type=float, default=1)
    parser.add_argument('--ssl_mode', type=str, default='both_side')

    # ******************************   ssl inter loss  paras      ***************************** #
    parser.add_argument('--ssl_reg_inter', nargs='?', default='[1,1]')
    parser.add_argument('--ssl_inter_mode', type=str, default='both_side')

    # ******************************  ssl2 similarity paras      ***************************** #
    parser.add_argument('--sim_measure', type=str, default='swing')
    parser.add_argument('--topk1_user', type=int, default=10)  #
    parser.add_argument('--topk1_item', type=int, default=10)  #

    # ******************************   model hyper paras      ***************************** #
    parser.add_argument('--n_fold', type=int, default=50)
    parser.add_argument('--att_dim', type=int, default=16, help='self att dim')  # d_a
    parser.add_argument('--wid', nargs='?', default='[0.1,0.1,0.1]',
                        help='negative weight, [0.1,0.1,0.1] for beibei, [0.01,0.01,0.01] for taobao')

    parser.add_argument('--decay', type=float, default=10,
                        help='Regularization, 10 for beibei, 0.01 for taobao')  # regularization decay

    parser.add_argument('--mf_decay', type=float, default=1, help='mf loss decay')  # mf loss decay

    parser.add_argument('--coefficient', nargs='?', default='[0.0/6, 5.0/6, 1.0/6]',
                        help='Regularization, [0.0/6, 5.0/6, 1.0/6] for beibei, [1.0/6, 4.0/6, 1.0/6] for taobao')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.3]',
                        help='Keep probability w.r.t. message dropout')

    parser.add_argument('--dropout_ratio', type=float, default=0.5)

    return parser.parse_args()
