from __future__ import division, print_function
from math import sqrt
from functools import partial

import numpy as np
import tensorflow as tf

from distBO.train.gp_utils import repeater, while_repeater
from distBO.train.feature_map import manual_features, dist_features

def rbf_kernel(X, Y, stddev=1.0, scale=1.0):
    X = tf.divide(X, stddev)
    Y = tf.divide(Y, stddev)
    X_sqnorms_row = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
    Y_sqnorms_col = tf.expand_dims(tf.reduce_sum(tf.square(Y), 1), 0)
    XY = tf.matmul(X, Y, transpose_b=True)
    return scale * tf.exp(-0.5 * (-2 * XY + X_sqnorms_row + Y_sqnorms_col))

def matern32_kernel(X, Y, stddev=1.0):
    X = tf.divide(X, stddev)
    Y = tf.divide(Y, stddev)
    X_sqnorms_row = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
    Y_sqnorms_col = tf.expand_dims(tf.reduce_sum(tf.square(Y), 1), 0)
    XY = tf.matmul(X, Y, transpose_b=True)
    dist = tf.sqrt(tf.maximum(-2 * XY + X_sqnorms_row + Y_sqnorms_col, 1e-32))
    value = sqrt(3.0)*dist
    output =  tf.multiply((1.0 + value), tf.exp(-value))
    return output

# Multi-Task Bayesian Optimisation Kernel
# https://papers.nips.cc/paper/5086-multi-task-bayesian-optimization.pdf
def task_kernel(task_triangular, sizes_1, sizes_2=None, dtype=None):
    n_data_1 = tf.shape(sizes_1)[0]
    # This is just LL^T
    task_kernel_single = tf.matmul(task_triangular, tf.transpose(task_triangular))
    # Ensure diagonal is 1.
    diag_sqrt = tf.sqrt(tf.diag_part(task_kernel_single))
    task_kernel_single = task_kernel_single / tf.expand_dims(diag_sqrt, axis=1)
    task_kernel_single = task_kernel_single / tf.expand_dims(diag_sqrt, axis=0)
    # Repeat it vertically
    start_task_kernel_vert = repeater(task_kernel_single[0], sizes_1[0])
    condition = lambda i, x: tf.less(i, n_data_1)
    body = lambda i, x: (i+1, tf.concat([x, repeater(task_kernel_single[i], sizes_1[i])],0))
    i, task_kernel_vert = tf.while_loop(condition, body, (tf.constant(1), start_task_kernel_vert))
    task_kernel_vert = tf.transpose(task_kernel_vert)
    if sizes_2 is not None:
        # Only works for target pred option only.
        t_kernel = repeater(task_kernel_vert[-1], sizes_2[0])
    else:
        # Repeat it horizontally
        start_task_kernel_horz = repeater(task_kernel_vert[0], sizes_1[0])
        condition = lambda i, x: tf.less(i, n_data_1)
        body = lambda i, x: (i+1, tf.concat([x, repeater(task_kernel_vert[i], sizes_1[i])],0))
        i, t_kernel = tf.while_loop(condition, body, (tf.constant(1), start_task_kernel_horz))
    return tf.transpose(t_kernel)

# Manual meta-features kernel
def manual_kernel(params, embed_1, sizes_1, embed_2=None, 
                  sizes_2=None, model_type=None, dtype=None):
    embed_1 = manual_features(embed_1, sizes_1, dtype=dtype)
    if embed_2 is not None and sizes_2 is not None:
        embed_2 = manual_features(embed_2, sizes_2, dtype=dtype)
    else:
        embed_2 = embed_1
    k_man = matern32_kernel(embed_1, embed_2, stddev=tf.exp(params['log_bw_embed']))
    return k_man

# TODO: Update to repeat instead
def dist_kernel(params, x_1, y_1, sizes_1, embed_sizes_1, sum_matrix_1,
                x_2=None, y_2=None, sizes_2=None, embed_sizes_2=None, 
                sum_matrix_2=None, data_sizes_1=None, data_sizes_2=None, 
                max_size=None, embed_op='joint', model_type=None, dtype=tf.float32):
    d_features = partial(dist_features, params=params,
                         embed_op=embed_op, model_type=model_type,
                         max_size=max_size, dtype=dtype)
    embed_1 = d_features(x=x_1, y=y_1, sizes=sizes_1,
                         sum_matrix=sum_matrix_1,
                         data_sizes=data_sizes_1,
                         embed_sizes=embed_sizes_1)
    if x_2 is not None and y_2 is not None:
        embed_2 = d_features(x=x_2, y=y_2, sizes=sizes_2, 
                             sum_matrix=sum_matrix_2,
                             data_sizes=data_sizes_2,
                             embed_sizes=embed_sizes_2)
    else:
        embed_2 = embed_1
    k_dist = matern32_kernel(embed_1, embed_2, stddev=tf.exp(params['log_bw_embed']))
    return k_dist
