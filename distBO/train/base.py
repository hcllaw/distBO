from __future__ import division

import numpy as np
import tensorflow as tf
from collections import namedtuple

SparseInfo = namedtuple('SparseInfo', ['indices', 'values', 'dense_shape'])

def sparse_joint_op(sum_matrix_input, embed_mat, pred=False):
    sum_matrix = tf.SparseTensor(*sum_matrix_input)
    return tf.sparse_tensor_dense_matmul(sum_matrix, embed_mat)

def sparse_matrix_placeholders(dtype=np.float32):
    '''
    Placeholders for a tf.SparseMatrix; use tf.SparseMatrix(*placeholders)
    to make the actual object.
    '''
    return SparseInfo(
        indices=tf.placeholder(tf.int64, [None, 2]),
        values=tf.placeholder(dtype, [None]),
        dense_shape=tf.placeholder(tf.int64, [2]),
    )

def sum_matrix(data, data_sizes, dtype=np.float32):

    length = len(data_sizes) # Number of tasks
    sizes = np.squeeze(data_sizes)
    if length == 1:
        sizes = [sizes]
    total_points = np.sum(sizes)
    '''
    Returns a len(feats) x feats.total_pts matrix to do mean-pooling by bag.
    '''
    bounds = np.r_[0, np.cumsum(sizes)]
    return SparseInfo(
        indices=np.vstack([
            [i, j]
            for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])) 
            for j in range(start, end)
        ]),
        values=[1 / size for size in sizes for _ in range(size)],
        dense_shape=[length, total_points],
    )

class Network(object):
    def __init__(self, in_dim_params, in_dim_x, in_dim_y, model_type, n_tasks=None, in_dim_meta=None,
                       use_dist_kernel=False, use_task_kernel=False, use_manual_kernel=False, 
                       embed_op='joint', concat_data_size=False, num_of_rff=100, dtype=tf.float32):
        self.model_type = model_type
        self.in_dim_params = in_dim_params
        self.dist = use_dist_kernel
        self.num_of_rff = num_of_rff
        self.n_tasks = n_tasks
        self.embed_op = embed_op
        self.multi = use_task_kernel
        self.manual = use_manual_kernel
        self.concat_data_size = concat_data_size
        self.inputs = {
            'h_params': tf.placeholder(dtype, [None, in_dim_params]),
            'obs': tf.placeholder(dtype, [None, 1]),
            'h_params_pred': tf.placeholder(dtype, [None, in_dim_params])
                      }
        if use_dist_kernel:
            self.inputs['X'] = tf.placeholder(dtype, [None, in_dim_x])
            self.inputs['X_pred'] = tf.placeholder(dtype, [None, in_dim_x])
            self.inputs['y'] = tf.placeholder(dtype, [None, in_dim_y])
            self.inputs['y_pred'] = tf.placeholder(dtype, [None, in_dim_y])
            self.inputs['sizes'] = tf.placeholder(dtype, [None, 1]) # one per dataset
            self.inputs['sizes_pred'] = tf.placeholder(dtype, [None, 1]) # one per dataset
            self.inputs['embed_sizes'] = tf.placeholder(dtype, [None, 1]) # one per dataset
            self.inputs['embed_sizes_pred'] = tf.placeholder(dtype, [None, 1]) # one per dataset
            self.inputs['sum_matrix'] = sparse_matrix_placeholders(dtype) # n_datasets, n_pts
            self.inputs['sum_matrix_pred'] = sparse_matrix_placeholders(dtype) # n_datasets, n_pts
            self.inputs['data_sizes'] = tf.placeholder(dtype, [None, 1]) # one per dataset
            self.inputs['data_sizes_pred'] = tf.placeholder(dtype, [None, 1]) # one per dataset
            self.inputs['dist'] = tf.placeholder(dtype, [None, None])
            self.inputs['dist_pred'] = tf.placeholder(dtype, [None, None])
            if concat_data_size:
                self.inputs['max_size'] = tf.placeholder(dtype)
        elif use_task_kernel:
            self.inputs['sizes'] = tf.placeholder(dtype, [None, 1]) # one per dataset
            self.inputs['sizes_pred'] = tf.placeholder(dtype, [None, 1]) # one per dataset
        elif use_manual_kernel:
            self.inputs['sizes'] = tf.placeholder(dtype, [None, 1]) # one per dataset
            self.inputs['sizes_pred'] = tf.placeholder(dtype, [None, 1]) # one per dataset
            self.inputs['embed'] = tf.placeholder(dtype,[None, in_dim_meta])
            self.inputs['embed_pred'] = tf.placeholder(dtype,[None, in_dim_meta])
        self.params = {}
        self.dtype = dtype

    def feed_dict(self, data, update_sum_matrix=False,
                  data_pred=None, sim=False, use_embed=False):
        if self.dist:
            if update_sum_matrix or not hasattr(self, 'sum_matrix_embed'):
                self.sum_matrix_embed = sum_matrix(data, data['embed_sizes'])
            if (data_pred or sim) and (update_sum_matrix or not hasattr(self, 'sum_matrix_embed_pred')):
                self.sum_matrix_embed_pred = sum_matrix(data_pred, data_pred['embed_sizes_pred'])
        i = self.inputs
        d = {
                i['h_params']: data['h_params'],
                i['obs']: data['obs'],
            }
        if self.dist:
            d[i['X']] = data['batch_X']
            d[i['y']] = data['batch_y']
            d[i['sizes']] = data['sizes']
            d[i['data_sizes']] = data['data_sizes']
            d[i['embed_sizes']] = data['embed_sizes']
            if self.concat_data_size:
                d[i['max_size']] = data['max_size']
            for p, v in zip(i['sum_matrix'], self.sum_matrix_embed):
                d[p] = v
        elif self.multi:
            d[i['sizes']] = data['sizes']
        elif self.manual:
            d[i['sizes']] = data['sizes']
            d[i['embed']] = data['embed']

        if data_pred is not None and not sim:
            d[i['h_params_pred']] = data_pred['h_params_pred']
            if self.dist:
                d[i['X_pred']] = data_pred['X_pred']
                d[i['y_pred']] = data_pred['y_pred']
                d[i['sizes_pred']] = data_pred['sizes_pred']
                d[i['data_sizes_pred']] = data_pred['data_sizes_pred']
                d[i['embed_sizes_pred']] = data_pred['embed_sizes_pred']
                for p, v in zip(i['sum_matrix_pred'], self.sum_matrix_embed_pred):
                    d[p] = v
            if use_embed:
                d[i['dist']] = data['dist']
                d[i['dist_pred']] = data_pred['dist_pred']
            elif self.multi:
                d[i['sizes_pred']] = data_pred['sizes_pred']
            elif self.manual:
                d[i['sizes_pred']] = data_pred['sizes_pred']
                d[i['embed_pred']] = data_pred['embed_pred']
        elif sim:
            d[i['sizes']] = np.array([ [1] * len(data['sizes']) ]).T
            d[i['sizes_pred']] = np.array([[1]])
            if self.dist:
                d[i['X_pred']] = data_pred['X_pred']
                d[i['y_pred']] = data_pred['y_pred']
                #d[i['data_sizes']] = data['data_sizes']
                #d[i['embed_sizes']] = data['embed_sizes']
                d[i['data_sizes_pred']] = data_pred['data_sizes_pred']
                d[i['embed_sizes_pred']] = data_pred['embed_sizes_pred']
                for p, v in zip(i['sum_matrix_pred'], self.sum_matrix_embed_pred):
                    d[p] = v
                if self.concat_data_size:
                    d[i['max_size']] = data['max_size']
            elif self.manual:
                d[i['embed']] = data['embed']
                d[i['embed_pred']] = data_pred['embed_pred']
        return d
