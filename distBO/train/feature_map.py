from __future__ import division, print_function
from math import sqrt
from functools import partial

import tensorflow as tf
from tensorflow.contrib.kernel_methods import RandomFourierFeatureMapper
from distBO.train.RandomFourierMaternFeatureMapper import RandomFourierMaternFeatureMapper

from distBO.train.gp_utils import repeater, while_repeater
from distBO.train.distbo_net import distbo_net

def nn_features(params, feature, rff_input_dim=None, num_of_rff=100, dtype=tf.float32):
    if 'weights__1' in params and 'weights__2' in params:
        feature = tf.nn.tanh(tf.matmul(feature, params['weights__1']) + params['bias__1'])
        feature = tf.matmul(feature, params['weights__2'])
        if 'weights__3' in params:
            print('Using 2 hidden layer NN')
            feature = tf.matmul(tf.nn.tanh(feature + params['bias__2']), params['weights__3'])
    if num_of_rff > 0:
        feature = tf.divide(feature, tf.exp(params['log_bw']))
        feature = tf.cast(feature, tf.float32)
        rff_mapper = RandomFourierMaternFeatureMapper(rff_input_dim, num_of_rff,
                                                      stddev=1.0,
                                                      seed=23)
        feature = rff_mapper.map(feature)
        feature = tf.cast(feature, dtype)
    return feature

def manual_features(embed, sizes, dtype=tf.float32):
    n_data = tf.shape(sizes)[0]
    start_arr = repeater(embed[0], sizes[0])
    condition = lambda i, x: tf.less(i, n_data)
    body = lambda i, x: (i+1, tf.concat([x, repeater(embed[i], sizes[i])], 0))
    i, embed = tf.while_loop(condition, body, (tf.constant(1), start_arr))
    return embed

def task_features(n_task, sizes, pred=False, dtype=tf.float32):
    task_mat = tf.eye(n_task, dtype=dtype)
    if pred:
        embed = repeater(task_mat[-1], sizes[0])
    else:
        start_arr = repeater(task_mat[0], sizes[0])
        condition = lambda i, x: tf.less(i, n_task)
        body = lambda i, x: (i+1, tf.concat([x, repeater(task_mat[i], sizes[i])], 0))
        i, embed = tf.while_loop(condition, body, (tf.constant(1), start_arr))
    return embed

def dist_features(params, x, y, sizes, sum_matrix,
                  data_sizes=None, embed_sizes=None,
                  max_size=None, embed_op='joint',
                  model_type=None, dtype=tf.float32):
    net = partial(distbo_net, params=params, embed_op=embed_op,
                  sum_matrix=sum_matrix, model_type=model_type, dtype=dtype)
    embed = net(x, y, embed_sizes)
    start_arr = repeater(embed[0], sizes[0])
    n_embed = tf.shape(sizes)[0]
    embed_arr = while_repeater(n_embed, embed, sizes, start_arr)
    if max_size is not None:
        data_ratio = tf.divide(data_sizes, max_size)
        #data_ratio = tf.Print(data_ratio, [data_ratio], message='data_ratio')
        data_ratio_arr = while_repeater(n_embed, data_ratio, sizes,
                                          repeater(data_ratio[0], sizes[0]))
        embed_arr = tf.concat([embed_arr, data_ratio_arr], 1)

    return embed_arr
