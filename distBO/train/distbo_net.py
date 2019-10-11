# DistBO network
import tensorflow as tf 

from distBO.train.gp_utils import repeater
from distBO.train.base import sparse_joint_op

def nn_net(params, input_array, name, dtype=tf.float32):
    hidden = tf.matmul(input_array, params['weights_{}_1'.format(name)]) + params['bias_{}_1'.format(name)]
    hidden = tf.nn.tanh(hidden)
    output = tf.matmul(hidden, params['weights_{}_2'.format(name)])
    if 'weights_{}_3'.format(name) in params:
        output = tf.nn.tanh(output + params['bias_{}_2'.format(name)])
        output = tf.matmul(output, params['weights_{}_3'.format(name)])
    return output

def distbo_net(x, y, embed_sizes, params, embed_op='joint',
               model_type=None, sum_matrix=None, dtype=tf.float32):
    if embed_op == 'nn':
        xy = tf.concat([x, y], axis=1)
        feature_xy = nn_net(params, xy, 'xy', dtype=dtype)
    else:
        # X network, None uses feature_xy
        feature_xy = feature_x = nn_net(params, x, 'x', dtype=dtype)
        # For P_XY and P_Y|X P_X, y weights
        if embed_op in ['joint', 'concat'] and model_type == 'regression':
            # Y network for regression
            feature_y = nn_net(params, y, 'y', dtype=dtype)
        elif model_type == 'classification':
            feature_y = y

    # Joint or classification, we compute outer product
    if embed_op == 'joint' or (model_type == 'classification' and embed_op != 'nn'):
        feature_x_dim = tf.shape(feature_x)[1]
        feature_y_dim = tf.shape(feature_y)[1]
        dim_all = feature_x_dim * feature_y_dim
        feature_x_expand = tf.expand_dims(feature_x, 2)
        feature_y_expand = tf.expand_dims(feature_y, 1)
        # Batch-wise perform outer product between b x i x 1 and b x 1 x j
        # Imagine compute i x j after extract it
        outer_product = tf.einsum('bil,blj->bij', feature_x_expand, feature_y_expand)
        n_datapoints = tf.shape(outer_product)[0]
        feature_xy = tf.reshape(outer_product, [n_datapoints, dim_all])
    if embed_op != 'concat' or model_type == 'classification':
        pool_embed = sparse_joint_op(sum_matrix, feature_xy)
        if model_type == 'classification': # Add class ratios
            class_ratio = sparse_joint_op(sum_matrix, y)
            #class_ratio = tf.Print(class_ratio, [class_ratio], message='class_ratio', summarize=1000)
            pool_embed = tf.concat([pool_embed, class_ratio], axis=1)
    else:
        c_size = tf.concat([tf.constant([0], tf.int32), 
                            tf.cumsum(tf.cast(tf.reshape(embed_sizes, [-1]), 
                                                      tf.int32))], 0)
        feature_x_dim = tf.shape(feature_x)[1]
        feature_y_dim = tf.shape(feature_y)[1]
        dim_all = feature_x_dim * feature_y_dim
        # Cond P_Y|X Operator
        def cond_embed_func(k):
            reg = tf.exp(params['log_reg'])
            fea_task_x = feature_x[c_size[k]:c_size[k+1]]
            fea_task_y = feature_y[c_size[k]:c_size[k+1]]
            # Let fea_task_x = \Phi (n by s), fea_task_y = \Psi (n by t)
            # Co_OP = \Psi^T (\Phi \Phi^T + \lambda I)^-1 \Phi
            # Equivalent to:
            # \Psi^T (\lambda^-1 I - \lambda^-1 \Phi( \lambda I + \Phi^T\Phi)^-1\Phi^T) \Phi
            # http://www.gaussianprocess.org/gpml/chapters/RW8.pdf Pg 2
            # https://www.cc.gatech.edu/~lsong/papers/SonFukGre13.pdf Pg 13
            # Dimensions
            feature_x_expand = tf.expand_dims(feature_x, 2)
            feature_y_expand = tf.expand_dims(feature_y, 1)
            # Compute
            fea_y_t_fea_x = tf.matmul(tf.transpose(fea_task_y), fea_task_x) # t by n
            fea_x_t_fea_x = tf.matmul(tf.transpose(fea_task_x), fea_task_x) # s by s
            inv_term = tf.linalg.inv(fea_x_t_fea_x + reg * tf.eye(feature_x_dim, dtype=dtype)) # s by s
            lambda_inv = 1.0 / reg # t by s - t by s (s by s * s by s) = t by s
            cond_embed = lambda_inv * (fea_y_t_fea_x - tf.matmul(fea_y_t_fea_x, tf.matmul(inv_term, fea_x_t_fea_x)))
            return tf.reshape(cond_embed, [dim_all])
        # Marg P_X embedding
        def marg_embed_func(k):
            fea_task_x = feature_x[c_size[k]:c_size[k+1]]
            return tf.reshape(tf.reduce_mean(fea_task_x, axis=0), [feature_x_dim])

        n_datasets = tf.shape(embed_sizes)[0]
        marginal_embed = tf.map_fn(fn=lambda k: marg_embed_func(k),
                                   elems=tf.range(tf.cast(n_datasets, dtype=tf.int32)), 
                                   dtype=dtype)
        cond_embed = tf.map_fn(fn=lambda k: cond_embed_func(k),
                               elems=tf.range(tf.cast(n_datasets, dtype=tf.int32)), 
                               dtype=dtype)
        pool_embed = tf.concat([marginal_embed, cond_embed], 1)
    return pool_embed