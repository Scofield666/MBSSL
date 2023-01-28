# -*- coding: utf-8 -*-
# @Time : 2022/1/18 6:27 下午
# @Author : crisis
# @FileName: models.py
# @Function：
# @Usage:

import tensorflow.compat.v1 as tf  # change to tfv1

tf.disable_v2_behavior()


def ln(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def ScaledDotProductAttention(q, k, v, dropout=0.1, training=True, scope="scaled_dot_product_attention"):
    """
    :param q: [N, len_q, d_k]
    :param k: [N, len_k, d_k]
    :param v: [N, len_k, d_v]
    :return: output [N, len_q, d_v]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = k.get_shape().as_list()[-1]
        # [N, len_q, d_k] * [N, d_k, len_k] -> [N, len_q, len_k]
        attn = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        attn /= d_k ** 0.5
        attn = tf.nn.softmax(attn)
        attn = tf.layers.dropout(attn, rate=dropout, training=training)

        # [N, len_q, len_k] * [N, len_k, d_v] -> [N, len_q, d_v]
        output = tf.matmul(attn, v)
    return output

def MultiHeadAttention(q, k, v, n_head=1, scope="multihead_attention"):
    """
    :param q: [N, len_q, d_model]
    :param k: [N, len_k, d_model]
    :param v: [N, len_k, d_model]
    :param n_head: 1
    len_q=len_k=len_v, 即q=k=v=x，传入参数全是[N, edge_type_cnt, dim]
    :return:  [N, len_q, d_model]
    """
    d_model = q.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 1. 由输入计算q.k.v。理解为计算每个edge_type下的q，k，v
        q = tf.layers.dense(q, d_model, use_bias=True, name='q')  # (N, len_q, d_model)
        k = tf.layers.dense(k, d_model, use_bias=True, name='k')  # (N, len_k, d_model)
        v = tf.layers.dense(v, d_model, use_bias=True, name='v')  # (N, len_k, d_model)

        # 2. Split and concat
        q_ = tf.concat(tf.split(q, n_head, axis=2), axis=0)  # (h*N, len_q, d_model/h)
        k_ = tf.concat(tf.split(k, n_head, axis=2), axis=0)  # (h*N, len_k, d_model/h)
        v_ = tf.concat(tf.split(v, n_head, axis=2), axis=0)  # (h*N, len_k, d_model/h)

        # Attention
        outputs = ScaledDotProductAttention(q_, k_, v_)  # [h*N, len_q, d_model/h]

        # Restore shape
        outputs = tf.concat(tf.split(outputs, n_head, axis=0), axis=2)  # (N, len_q, d_model)

        # Residual connection
        outputs += q

        # Normalize
        outputs = ln(outputs)

    return outputs


def ScaledDotProductAttention2(q, k, v, dropout=0.1, training=True, scope="scaled_dot_product_attention"):
    """
    :param q: [N, len_q, d_k]
    :param k: [N, len_k, d_k]
    :param v: [N, len_k, d_v]
    :return: output [N, len_q, d_v]. attn [N, len_q, len_k]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = k.get_shape().as_list()[-1]
        # [N, len_q, d_k] * [N, d_k, len_k] -> [N, len_q, len_k]
        attn = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        attn /= d_k ** 0.5
        attn = tf.nn.softmax(attn)
        attn = tf.layers.dropout(attn, rate=dropout, training=training)
        # attn = tf.nn.dropout(attn, 1- dropout)

        # [N, len_q, len_k] * [N, len_k, d_v] -> [N, len_q, d_v]
        output = tf.matmul(attn, v)
    return output, attn


def MultiHeadAttention2(q, k, v, n_head=1, scope="multihead_attention"):
    """
    :param q: [N, len_q, d_model]
    :param k: [N, len_k, d_model]
    :param v: [N, len_k, d_model]
    :param n_head: 1
    len_q=len_k=len_v, 即q=k=v=x，传入参数全是[N, edge_type_cnt, dim]
    :return:  [N, len_q, d_model], [N, nh, len_q, len_k]
    """
    d_model = q.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 1. 由输入计算q.k.v。理解为计算每个edge_type下的q，k，v
        q = tf.layers.dense(q, d_model, use_bias=True, name='q')  # (N, len_q, d_model)
        k = tf.layers.dense(k, d_model, use_bias=True, name='k')  # (N, len_k, d_model)
        v = tf.layers.dense(v, d_model, use_bias=True, name='v')  # (N, len_k, d_model)

        # for debug 常数初始化
        # q = tf.layers.dense(q, d_model, use_bias=False, kernel_initializer=tf.constant_initializer(value=1), name='q')  # (N, len_q, d_model)
        # k = tf.layers.dense(k, d_model, use_bias=False, kernel_initializer=tf.constant_initializer(value=1), name='k')  # (N, len_k, d_model)
        # v = tf.layers.dense(v, d_model, use_bias=False, kernel_initializer=tf.constant_initializer(value=1), name='v')  # (N, len_k, d_model)
        # return q, k, v

        # 2. Split and concat
        q_ = tf.concat(tf.split(q, n_head, axis=2), axis=0)  # (h*N, len_q, d_model/h)
        k_ = tf.concat(tf.split(k, n_head, axis=2), axis=0)  # (h*N, len_k, d_model/h)
        v_ = tf.concat(tf.split(v, n_head, axis=2), axis=0)  # (h*N, len_k, d_model/h)
        # return q_, k_, v_
        # Attention
        outputs, attn = ScaledDotProductAttention2(q_, k_, v_)  # [h*N, len_q, d_model/h], [h*N, len_q, len_k]
        # return outputs, attn

        # Restore shape
        outputs = tf.concat(tf.split(outputs, n_head, axis=0), axis=2)  # (N, len_q, d_model)
        attn = tf.stack(tf.split(attn, n_head, axis=0), axis=1)  # [N, nh, len_q, len_k]

        # Residual connection
        outputs += q
        # return outputs
        # Normalize
        outputs = ln(outputs)

    return outputs, attn

import random
import numpy as np
import os
if __name__ == '__main__':
    def set_seeds(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_random_seed(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.

    set_seeds(2020)
    # input = tf.ones([5, 3, 4], dtype=float)
    input = tf.constant([
        [[1,2,3,4],[2,3,4,5]],
        [[2,3,4,5],[3,4,5,6]]
    ], dtype=float)
    # q, k, v = MultiHeadAttention2(input, input, input)
    # output, attn = ScaledDotProductAttention2(input, input, input)
    output, _ = MultiHeadAttention2(input, input, input)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([output]))
        # print(sess.run([q]))
        # print(sess.run([k]))
        # print(sess.run([v]))

