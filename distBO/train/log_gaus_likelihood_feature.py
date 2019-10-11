# Gaussian Process Log Marginal Likelihood Optimisation
from __future__ import division, print_function
from functools import partial 
from math import log, sqrt

import tensorflow as tf
import numpy as np

from distBO.train.base import Network
from distBO.train.gp_utils import multivariate_normal, nn_weight_setter
from distBO.train.feature_map import (nn_features, dist_features, manual_features, task_features)

def build_log_marg_likhd_feature(in_dim_params, n_tasks=None,
                                 in_dim_x=None, in_dim_y=None, weight_dim_xy=None,
                                 weight_dim=None, weight_dim_x=None, weight_dim_y=None,
                                 model_type='regression', embed_op='joint', num_of_rff=100,
                                 log_sigma_init=log(0.001), in_dim_meta=None,
                                 use_dist_kernel=False, use_task_kernel=False, use_manual_kernel=False,
                                 concat_data_size=False, dtype=tf.float32, seed=23):
    ### PARAMETERS ###
    # Construct network class
    net = Network(in_dim_params, in_dim_x, in_dim_y, model_type, n_tasks=n_tasks,
                  in_dim_meta=in_dim_meta, concat_data_size=concat_data_size,
                  use_dist_kernel=use_dist_kernel, use_task_kernel=use_task_kernel, 
                  use_manual_kernel=use_manual_kernel, num_of_rff=num_of_rff,
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
                    embed_dim = weight_dim_x[-1] * weight_dim_y[-1]
                elif embed_op == 'concat': # Additional marginal embedding
                    embed_dim = weight_dim_x[-1] * (weight_dim_y[-1] + 1)
                    params['log_reg'] = tf.Variable(tf.log(tf.constant(1.0, dtype=dtype)), 
                                                    name='log_reg',dtype=dtype)
            else: # Classification, embed by class, Regression, embed data
                 embed_dim = weight_dim_x[-1] * in_dim_y
        elif embed_op == 'nn':
            total_dim = in_dim_x + in_dim_y
            params = nn_weight_setter(params, total_dim, weight_dim_xy, 'xy',
                                      seed=seed, dtype=dtype)
            embed_dim = weight_dim_xy[-1]
        if concat_data_size: # Add datasize ratio
            embed_dim = embed_dim + 1
        if model_type == 'classification':
            embed_dim = embed_dim + in_dim_y
    # Manual meta-features kernel
    elif use_manual_kernel:
        embed_dim = in_dim_meta
    elif use_task_kernel:
        embed_dim = n_tasks
    else:
        embed_dim = 0
    
    total_dim = in_dim_params + embed_dim
    
    # Neural net for hyperparameters
    params = nn_weight_setter(params, total_dim, weight_dim, '',
                                         seed=seed, dtype=dtype)
    if len(weight_dim) != 0:
        rff_input_dim = net.rff_input_dim = weight_dim[-1]
    else:
        rff_input_dim = net.rff_input_dim = total_dim

    # BLR hyper-parameters
    #params['mean'] = tf.Variable(tf.constant([0], dtype=dtype), name='mean')
    params['log_sigma'] = tf.Variable(tf.constant(log_sigma_init, dtype=dtype), name='log_sigma')
    params['log_bw'] = tf.Variable(tf.constant([0] * rff_input_dim, dtype=dtype), name='log_bw')
    params['log_alpha'] = tf.Variable(tf.constant(0, dtype=dtype), name='log_alpha')

    sigma_sq = tf.exp(2.0 * params['log_sigma'])
    alpha = tf.exp(params['log_alpha'])
    n_datapoints = tf.shape(inputs['h_params'])[0]

    # Distribution Kernel
    if use_dist_kernel:
        dist_features_base = partial(dist_features, params, inputs['X'], inputs['y'], 
                                     inputs['sizes'], net.inputs['sum_matrix'],
                                     embed_sizes=inputs['embed_sizes'],
                                     data_sizes=inputs['data_sizes'],
                                     embed_op=embed_op,
                                     model_type=net.model_type, dtype=dtype)
        if concat_data_size:
            net.dist_fea = dist_fea = dist_features_base(max_size=inputs['max_size'])
        else:
            net.dist_fea = dist_fea = dist_features_base()
        net.feature = feature = tf.concat([dist_fea, inputs['h_params']], 1)
    # Manual meta-features Kernel
    elif use_manual_kernel:
        net.man_fea = man_fea = manual_features(inputs['embed'], inputs['sizes'], dtype=dtype)
        net.feature = feature = tf.concat([man_fea, inputs['h_params']], 1)
    elif use_task_kernel:
        net.task_fea = task_fea = task_features(n_tasks, inputs['sizes'], dtype=dtype)
        net.feature = feature = tf.concat([task_fea, inputs['h_params']], 1)
    else:
        net.feature = feature = inputs['h_params']
    # Feature map transformation
    net.nn_fea = nn_fea = nn_features(params, feature, rff_input_dim=rff_input_dim,
                                      num_of_rff=num_of_rff, dtype=dtype)
    nn_fea_dim = tf.shape(nn_fea)[1]
    # http://www.gaussianprocess.org/gpml/chapters/RW8.pdf Pg 2
    # https://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning (Appendix)
    # Sylvester's determinant identity https://en.wikipedia.org/wiki/Sylvester%27s_determinant_identity
    
    # negative Log GP Marg Likelihood (utilising that K = \Phi \Phi^T)
    # 1.0 / \sigma^ 2 [ || y ||^2_2 - (alpha / \sigma^2 ) || L^-1 \Phi^T y ||^2_2]
    # + N log(sigma^2) + 2 \sum_i log(L_ii)
    # e_term = L^-1 \Phi^T y
    # ( I + (alpha/sigma^2) \Phi^T \Phi )^-1 = LL^T (cholesky decomposition)
    # we add manual jittter incase sigma_sq goes to 0

    #mean = tf.expand_dims(params['mean'] * tf.ones(n_datapoints, dtype=dtype), 1)
    y = inputs['obs'] #- mean
    sigma_sq_inv = 1.0 / sigma_sq
    # Term 1: || y ||^2_2
    term_1 = tf.square(tf.norm(y, ord=2))

    # Term 2: (alpha / \sigma^2 ) || L^-1 \Phi^T y ||^2_2
    phi_t_phi = tf.matmul(tf.transpose(nn_fea), nn_fea)
    # If BLR crashes, let's add jitter, also note we need this for noneBLR protein dataset.
    inside_term = (1.0) * tf.eye(nn_fea_dim, dtype=dtype) + alpha * sigma_sq_inv * phi_t_phi
    L = tf.cholesky(inside_term )
    net.L_inv = L_inv = tf.linalg.inv(L)
    phi_t_y = tf.matmul(tf.transpose(nn_fea), y)
    net.e_term = e_term = tf.matmul(L_inv, phi_t_y)
    term_2 =  alpha * sigma_sq_inv * tf.square(tf.norm(e_term, ord=2))

    # Term 3: N log(sigma^2)
    term_3 = tf.log(sigma_sq) * tf.cast(n_datapoints, dtype=dtype)

    # Term 4: 2 \sum_i log(L_ii)
    term_4 = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
    #term_4 = tf.Print(term_4, [term_4], 'term_4')
    # Negative of marg likelihood
    net.loss = (sigma_sq_inv * (term_1 - term_2) + term_3 + term_4)
    return net

def build_predict_feature(net):
    """
    Xnew is a data matrix, point at which we want to predict
    This method computes
        p(F* | Y )
    where F* are points on the GP at Xnew, Y are noisy observations at X.
    """
    inputs = net.inputs
    params = net.params
    sigma_sq = tf.exp(2.0 * params['log_sigma'])
    alpha = tf.exp(params['log_alpha'])
    # Distribution kernel (diagonal is identity)
    if net.dist:
        dist_features_base = partial(dist_features, params, inputs['X_pred'], inputs['y_pred'], 
                                     inputs['sizes_pred'], inputs['sum_matrix_pred'],
                                     embed_sizes=inputs['embed_sizes_pred'],
                                     data_sizes=inputs['data_sizes_pred'],
                                     embed_op=net.embed_op,
                                     model_type=net.model_type, 
                                     dtype=net.dtype)
        if net.concat_data_size:
            net.dist_fea_pred = dist_fea_pred = dist_features_base(max_size=inputs['max_size'])
        else:
            net.dist_fea_pred = dist_fea_pred = dist_features_base()
        net.feature_pred = feature_pred = tf.concat([dist_fea_pred, inputs['h_params_pred']], 1)

    # Manual meta-features kernel (diagonal is identity)
    elif net.manual:
        net.man_fea_pred = man_fea_pred = manual_features(inputs['embed_pred'], inputs['sizes_pred'], 
                                                          dtype=net.dtype)
        net.feature_pred = feature_pred = tf.concat([man_fea_pred, inputs['h_params_pred']], 1)
    elif net.multi:
        net.task_fea_pred = task_fea_pred = task_features(net.n_tasks, inputs['sizes_pred'], pred=True,
                                                          dtype=net.dtype)
        net.feature_pred = feature_pred = tf.concat([task_fea_pred, inputs['h_params_pred']], 1)
    else:
        net.feature_pred = feature_pred = inputs['h_params_pred']
    
    # Hyperparameter
    net.nn_fea_pred = nn_fea_pred = nn_features(params, feature_pred, rff_input_dim=net.rff_input_dim,
                                                num_of_rff=net.num_of_rff, dtype=net.dtype)
    # Mean: alpha / sigma^2 e_term^T L^-1 \Phi^T
    # Var alpha L^-1 \Phi^T, then || L^-1 \Phi^T||^2_2 across dimensions.
    L_inv_fea_pred_t = tf.matmul(net.L_inv, tf.transpose(nn_fea_pred))
    net.pred_mean = (alpha / sigma_sq) * tf.transpose(tf.matmul(tf.transpose(net.e_term), 
                                                 L_inv_fea_pred_t)) #+ tf.expand_dims(params['mean'],1)
    net.pred_var = alpha * tf.expand_dims(tf.reduce_sum(tf.square(L_inv_fea_pred_t), 0), 1)
    return net

def build_predict_dist_feature(net):
    inputs = net.inputs
    params = net.params
    sigma_sq = tf.exp(2.0 * params['log_sigma'])
    alpha = tf.exp(params['log_alpha'])
    # Distribution kernel (diagonal is identity)
    net.feature = feature = tf.concat([inputs['dist'], inputs['h_params']], 1)
    net.feature_pred = feature_pred = tf.concat([inputs['dist_pred'], inputs['h_params_pred']], 1)

    net.nn_fea = nn_fea = nn_features(params, feature, rff_input_dim=net.rff_input_dim,
                                      num_of_rff=net.num_of_rff, dtype=net.dtype)
    net.nn_fea_pred = nn_fea_pred = nn_features(params, feature_pred, rff_input_dim=net.rff_input_dim,
                                                num_of_rff=net.num_of_rff, dtype=net.dtype)
    
    sigma_sq_inv = 1.0 / sigma_sq
    phi_t_phi = tf.matmul(tf.transpose(nn_fea), nn_fea)
    nn_fea_dim = tf.shape(nn_fea)[1]
    inside_term = tf.eye(nn_fea_dim, dtype=net.dtype) + alpha * sigma_sq_inv * phi_t_phi
    L = tf.cholesky(inside_term)
    net.L_inv = L_inv = tf.linalg.inv(L)
    phi_t_y = tf.matmul(tf.transpose(nn_fea), inputs['obs'])
    net.e_term = e_term = tf.matmul(L_inv, phi_t_y)

    L_inv_fea_pred_t = tf.matmul(L_inv, tf.transpose(nn_fea_pred))
    net.pred_mean = (alpha / sigma_sq) * tf.transpose(tf.matmul(tf.transpose(net.e_term), 
                                                      L_inv_fea_pred_t))
    net.pred_var = alpha * tf.expand_dims(tf.reduce_sum(tf.square(L_inv_fea_pred_t), 0), 1)
    return net

