# Gaussian Process Log Marginal Likelihood Optimisation
from __future__ import division, print_function
from math import log, sqrt
from functools import partial

import tensorflow as tf
import numpy as np

from distBO.train.base import Network
from distBO.train.gp_utils import multivariate_normal, nn_weight_setter
from distBO.train.kernels import (matern32_kernel, task_kernel, 
                                  manual_kernel, dist_kernel)

def build_log_marg_likhd(in_dim_params, n_tasks=None,
                         in_dim_x=None, in_dim_y=None,
                         weight_dim_xy=None, weight_dim_x=None, weight_dim_y=None,
                         model_type='regression',
                         log_sigma_init=log(0.001), bw_params_init=1.0,
                         use_task_kernel=False, use_dist_kernel=False,
                         use_manual_kernel=False, in_dim_meta=None,
                         concat_data_size=False, embed_op='joint',
                         dtype=tf.float32, seed=23):
    ### PARAMETERS ###
    # Construct network class
    net = Network(in_dim_params, in_dim_x, in_dim_y, model_type, in_dim_meta=in_dim_meta,
                  use_dist_kernel=use_dist_kernel, use_task_kernel=use_task_kernel, 
                  use_manual_kernel=use_manual_kernel, concat_data_size=concat_data_size, 
                  embed_op=embed_op, dtype=dtype)
    inputs = net.inputs
    params = net.params

    # Distribution kernel
    if use_dist_kernel:
        if embed_op in ['none', 'joint', 'concat']:
            # X weights
            params = nn_weight_setter(params, in_dim_x, weight_dim_x, 'x',
                                      seed=seed, dtype=dtype)
            # For P_XY and P_Y|X P_X, y weights
            if embed_op in ['joint', 'concat'] and model_type == 'regression':
                # Y weights
                params = nn_weight_setter(params, in_dim_y, weight_dim_y, 'y',
                                          seed=seed, dtype=dtype)
                if embed_op == 'joint': # Outer product
                    num_bw = weight_dim_x[-1] * weight_dim_y[-1]
                elif embed_op == 'concat': # Additional marginal embedding
                    num_bw = weight_dim_x[-1] * (weight_dim_y[-1] + 1)
                    params['log_reg'] = tf.Variable(tf.log(tf.constant(1.0, dtype=dtype)), 
                                                    name='log_reg',dtype=dtype)
            else: # Classification, embed by class, Regression, embed data
                num_bw = weight_dim_x[-1] * in_dim_y
        elif embed_op == 'nn':
            total_dim = in_dim_x + in_dim_y
            params = nn_weight_setter(params, total_dim, weight_dim_xy, 'xy',
                                      seed=seed, dtype=dtype)
            num_bw = weight_dim_xy[-1]
        if concat_data_size: # Add datasize ratio
            num_bw = num_bw + 1
        if model_type == 'classification': # Add datasize ratio
            num_bw = num_bw + in_dim_y
        # Matern kernel bandwidth
        params['log_bw_embed'] = tf.Variable(tf.log(tf.constant([1.0] * num_bw, dtype=dtype)),
                                             name='log_bw_embed')
    # Manual meta-features kernel
    elif use_manual_kernel:
        params['log_bw_embed'] = tf.Variable(tf.log(tf.constant([1.0] * in_dim_meta, dtype=dtype)), 
                                             name='log_bw_embed')

    # Multi-task kernel
    if use_task_kernel:
        np.random.seed(seed)
        params['log_triangular_task'] = tf.Variable(tf.constant(np.random.normal(size=n_tasks*(n_tasks+1)//2 ), dtype=dtype), 
                                                    dtype=dtype, name='log_triangular_task')
        # https://papers.nips.cc/paper/5086-multi-task-bayesian-optimization.pdf
        # Sample each element of cholesky in the log space, to ensure positive correlation
        net.task_triangular = task_triangular = tf.contrib.distributions.fill_triangular(tf.exp(params['log_triangular_task']))
    
    # Hyperparameter kernel
    bw_ard_init = [ bw_params_init for _ in range(in_dim_params) ]
    params['log_bw'] = tf.Variable(tf.log(tf.constant(bw_ard_init, dtype=dtype)), name='log_bw')
    
    # GP hyper-parameters
    params['log_scale'] = tf.Variable(tf.constant([0], dtype=dtype), name='log_scale')
    params['mean'] = tf.Variable(tf.constant([0], dtype=dtype), name='mean')
    params['log_sigma'] = tf.Variable(tf.constant(log_sigma_init, dtype=dtype), name='log_sigma')

    scale = tf.exp(params['log_scale'])
    stddev = tf.exp(params['log_bw'])
    sigma_sq = tf.exp(2.0 * params['log_sigma'])
    n_datapoints = tf.shape(inputs['h_params'])[0]

    ### CONSTRUCT GRAPHS FOR KERNELS ###
    k_dist = k_man = k_task = 1 # Preset them to 1

    # Hyperparameter Kernel
    net.k_params = k_params = matern32_kernel(inputs['h_params'], inputs['h_params'], stddev=stddev)
    
    # Distribution Kernel
    if use_dist_kernel:
        dist_kernel_base = partial(dist_kernel, params, inputs['X'], inputs['y'], 
                                   inputs['sizes'], inputs['embed_sizes'],
                                   net.inputs['sum_matrix'],
                                   data_sizes_1=inputs['data_sizes'],
                                   embed_op=embed_op,
                                   model_type=net.model_type, dtype=dtype)
        if concat_data_size:
            net.k_dist = k_dist = dist_kernel_base(max_size=inputs['max_size'])
        else:
            net.k_dist = k_dist = dist_kernel_base()
    # Manual meta-features Kernel
    if use_manual_kernel:
        net.k_man = k_man = manual_kernel(params, inputs['embed'], inputs['sizes'],
                                          model_type=net.model_type, dtype=dtype)

    # Multi-task Kernel
    if use_task_kernel:
        net.k_task = k_task = task_kernel(task_triangular, inputs['sizes'], dtype=dtype)

    # Product of Kernels.
    net.k = k = scale * k_params * k_task * k_dist * k_man

    # Log GP Marg Likelihood, we add manual jittter incase sigma_sq goes to 0
    net.k_sigma_sq = k_sigma_sq = k + (sigma_sq + 0.000001) * tf.eye(n_datapoints, dtype=dtype)
    net.L = L = tf.cholesky(k_sigma_sq)
    mean = tf.expand_dims(params['mean'] * tf.ones(n_datapoints, dtype=dtype), 1)
    net.loss = - tf.squeeze(multivariate_normal(inputs['obs'], mean, L))
    return net

def build_predict(net):
    """
    Xnew is a data matrix, point at which we want to predict

    This method computes

        p(F* | Y )

    where F* are points on the GP at Xnew, Y are noisy observations at X.

    """
    inputs = net.inputs
    params = net.params
    scale = tf.exp(params['log_scale'])
    k_diag = scale
    k_dist_pred = k_man_pred = k_task_pred = 1.0

    # Distribution kernel (diagonal is identity)
    if net.dist:
        dist_kernel_base = partial(dist_kernel, params, inputs['X'], inputs['y'], 
                                   inputs['sizes'], inputs['embed_sizes'],
                                   inputs['sum_matrix'],
                                   x_2=inputs['X_pred'], y_2=inputs['y_pred'],
                                   sizes_2=inputs['sizes_pred'],
                                   data_sizes_1=inputs['data_sizes'],
                                   data_sizes_2=inputs['data_sizes_pred'],
                                   embed_sizes_2=inputs['embed_sizes_pred'],
                                   sum_matrix_2=inputs['sum_matrix_pred'],
                                   embed_op=net.embed_op,
                                   model_type=net.model_type, dtype=net.dtype)
        if net.concat_data_size:
            net.k_dist_pred = k_dist_pred = dist_kernel_base(max_size=inputs['max_size'])
        else:
            net.k_dist_pred = k_dist_pred = dist_kernel_base()

    # Manual meta-features kernel (diagonal is identity)
    if net.manual:
        net.k_man_pred = k_man_pred = manual_kernel(params, inputs['embed'], inputs['sizes'], 
                                                    embed_2=inputs['embed_pred'], 
                                                    sizes_2=inputs['sizes_pred'], 
                                                    model_type=net.model_type, 
                                                    dtype=net.dtype)
    # Multi-task Kernel
    if net.multi:
        # Assume prediction only on target, otherwise need to indexes of tasks being predicted.
        net.k_task_pred = k_task_pred = task_kernel(net.task_triangular,
                                                    inputs['sizes'],
                                                    sizes_2=inputs['sizes_pred'])

    # Hyperparameter kernel
    k_params_pred = matern32_kernel(inputs['h_params'], inputs['h_params_pred'], 
                                    stddev=tf.exp(params['log_bw']))
    # Product kernel
    k_pred = scale * k_params_pred * k_dist_pred * k_task_pred * k_man_pred

    A = tf.matrix_triangular_solve(net.L, k_pred, lower=True)
    V = tf.matrix_triangular_solve(net.L, inputs['obs'] - params['mean'])
    net.pred_mean = tf.matmul(A, V, transpose_a=True) + tf.expand_dims(params['mean'],1)
    fvar = k_diag - tf.reduce_sum(tf.square(A), 0)
    net.pred_var = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(inputs['obs'])[1]])
    return net

def build_predict_dist(net):
    inputs = net.inputs
    params = net.params

    scale = tf.exp(params['log_scale'])
    stddev = tf.exp(params['log_bw'])
    sigma_sq = tf.exp(2.0 * params['log_sigma'])
    n_datapoints = tf.shape(inputs['h_params'])[0]

    k_diag = scale

    # Hyperparameter kernel
    k_params = matern32_kernel(inputs['h_params'], inputs['h_params'], stddev=stddev)
    k_params_pred = matern32_kernel(inputs['h_params'], inputs['h_params_pred'], stddev=stddev)
    
    # Product kernel
    k = scale * k_params * inputs['dist']
    k_pred = scale * k_params_pred * inputs['dist_pred']

    # Log GP Marg Likelihood, we add manual jittter incase sigma_sq goes to 0
    k_sigma_sq = k + (sigma_sq + 0.000001) * tf.eye(n_datapoints, dtype=net.dtype)
    L = tf.cholesky(k_sigma_sq)

    A = tf.matrix_triangular_solve(L, k_pred, lower=True)
    V = tf.matrix_triangular_solve(L, inputs['obs'] - params['mean'])
    net.pred_mean = tf.matmul(A, V, transpose_a=True) + tf.expand_dims(params['mean'],1)
    fvar = k_diag - tf.reduce_sum(tf.square(A), 0)
    net.pred_var = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(inputs['obs'])[1]])
    return net

