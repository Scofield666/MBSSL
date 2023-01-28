'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run GHCF.")

    parser.add_argument('--sparsity_res', action="store_true")  # default false
    parser.add_argument('--save_emb_target', action="store_false", help='save embeddings under target behavior.')
    parser.add_argument('--save_emb', action="store_true", help='save embeddings.')  # default false
    parser.add_argument('--save_bst', action="store_true", help='save best embeddings.')  # default false, --save_bst指定为true. 需要在--save_emb基础上再加, 加了表示保存最好的，不加表示保存最后一个时刻的。
    parser.add_argument('--save_att', action="store_true", help='save att scores.')
    parser.add_argument('--ssl_type', type=int, default=1)
    parser.add_argument('--aug_type', type=int, default=0)
    parser.add_argument('--ssl_ratio', type=float, default=0.5)
    parser.add_argument('--ssl_temp', type=float, default=0.5)
    parser.add_argument('--ssl_reg', type=float, default=1)
    parser.add_argument('--ssl_mode', type=str, default='both_side')
    parser.add_argument('--ssl_gcn', type=str, default='gcn')

    # ******************************   optimizer paras      ***************************** #
    parser.add_argument('--lr', type=float, default=0.001,   #common parameter
                        help='Learning rate.')
    parser.add_argument('--lr_decay', action="store_true")  # default false --lr_decay
    parser.add_argument('--lr_decay_step', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=0.8)
    parser.add_argument('--meta_s', type=int, default=8)  # strategy 1 or 2, 2为metabalance，1为辅助任务梯度小则不放大
    parser.add_argument('--meta_r', type=float, default=0.7)  # metabalance relax factor
    parser.add_argument('--meta_b', type=float, default=0.9)  # metabalance beta

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
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
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
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',   # not used in GHCF
                        help='Regularizations.')

    parser.add_argument('--model_type', nargs='?', default='ghcf',
                        help='Specify the name of model (lightgcn,ghcf).')
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='lightgcn',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')

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



    #GHCF parameters
    # ******************************   ab paras      ***************************** #
    parser.add_argument('--aux_beh_idx', nargs='?', default='[0,1]')  # V7, 与目标行为对比的辅助行为下标

    # ******************************   dt loss node paras      ***************************** #
    parser.add_argument('--dt_node_mode', type=str, default='both_side')
    parser.add_argument('--dt_node_reg', nargs='?', default='[0.1,0.1]')  # dt node loss reg, v8 拉开节点在不同行为emb距离
    # ******************************   dt loss relation paras      ***************************** #
    parser.add_argument('--dt_loss', action="store_true")  # default false, --dt_loss 激活dt loss
    parser.add_argument('--dt_decay', type=float, default=1.0)

    # ******************************   ssl inter loss  paras      ***************************** #
    parser.add_argument('--ssl_reg_inter', nargs='?', default='[1,1]')  # ssl_inter_loss reg, default 全为1不做任何设置
    parser.add_argument('--ssl_inter_mode', type=str, default='both_side')  # ssl_inter mode

    parser.add_argument('--ssl1_reg', type=float, default=1.0, help='ssl1 loss reg')  #
    parser.add_argument('--ssl2_reg', type=float, default=1.0, help='ssl2 loss reg')  # 所有reg都为1，用metabalance自动调参

    # ******************************  ssl2 similarity paras      ***************************** #
    parser.add_argument('--sim_measure', type=str, default='swing')  # 相似度方法 pmi / swing
    parser.add_argument('--hard_neg_ssl1', action="store_true")  # default false
    parser.add_argument('--hard_neg_ssl2', type=str, default='v3')  # v0: 不使用 v1：用各自行为下的emb去算相似度，v2:用目标行为下的emb
    parser.add_argument('--select_neg_ssl2', type=int, default=3)  # v1：选择topk v2：选择topk1-topk2
    parser.add_argument('--topk1_user', type=int, default=10)  #
    parser.add_argument('--topk2_user', type=int, default=500)  #
    parser.add_argument('--topk1_item', type=int, default=10)  #
    parser.add_argument('--topk2_item', type=int, default=500)  #
    parser.add_argument('--pmi_th', type=float, default=5.0)  # pmi相似度阈值，大于该值的认为相似
    parser.add_argument('--sim_weights', nargs='?', default='[1./3,1./3,1./3]')  # 每个行为的相似度权重
    parser.add_argument('--sim_mode', type=str, default='unified')  # unified：使用统一的相似度 single：使用各自行为下的相似度
    parser.add_argument('--sim_single', type=str, default='aux') # 当sim_mode=single有效. aux: 只用当前辅助行为 awt:用当前辅助行为和目标行为
    # ******************************   model hyper paras      ***************************** #
    parser.add_argument('--n_fold', type=int, default=50)
    parser.add_argument('--att_dim', type=int, default=16, help='gatne self att dim')  # d_a
    parser.add_argument('--wid', nargs='?', default='[0.1,0.1,0.1]',
                        help='negative weight, [0.1,0.1,0.1] for beibei, [0.01,0.01,0.01] for taobao')

    parser.add_argument('--decay', type=float, default=10,
                        help='Regularization, 10 for beibei, 0.01 for taobao')  # regularization decay

    parser.add_argument('--mf_decay', type=float, default=1, help='mf loss decay')  # mf loss decay

    parser.add_argument('--coefficient', nargs='?', default='[0.0/6, 5.0/6, 1.0/6]',
                        help='Regularization, [0.0/6, 5.0/6, 1.0/6] for beibei, [1.0/6, 4.0/6, 1.0/6] for taobao')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.3]',
                        help='Keep probability w.r.t. message dropout, 0.2 for beibei and taobao')

    parser.add_argument('--direc_v', type=int, default=1)   # 展示梯度方向的不同方法
    # EHCF parameters
    parser.add_argument('--dropout_ratio', type=float, default=0.5) # ehcf

    return parser.parse_args()
