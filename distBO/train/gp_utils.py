# Some GP utility
import random
from copy import deepcopy
from functools import partial 

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from distBO.utils import check_dims, binarize_labels

def nn_weight_setter(params, in_dim, weight_dim, name,
                     seed=23, dtype=tf.float32):
    if len(weight_dim) >= 2:
        cst = partial(tf.cast, dtype=dtype)
        initializer = tf.keras.initializers.glorot_normal(seed=seed)
        params['weights_{}_1'.format(name)] = tf.Variable(cst(initializer([in_dim, weight_dim[0]])), dtype=dtype)
        params['bias_{}_1'.format(name)] = tf.Variable(tf.constant([0] * weight_dim[0], dtype=dtype), dtype=dtype)
        params['weights_{}_2'.format(name)] = tf.Variable(cst(initializer([weight_dim[0], weight_dim[1]])), dtype=dtype)
        if len(weight_dim) == 3:
            params['bias_{}_2'.format(name)] = tf.Variable(tf.constant([0]*weight_dim[1], dtype=dtype), 
                                                           name='bias_2', dtype=dtype)
            params['weights_{}_3'.format(name)] = tf.Variable(cst(initializer([weight_dim[1], weight_dim[2]])),
                                                              name='weights_2', dtype=dtype)
    return params

def loop_batches(train, batch_size, max_epochs, rs, model_type, stratify=False):
    if 'X' in train and 'y' in train:
        X_list = train['X']
        Y_list = train['y']
        n_datasets = len(X_list)
        if model_type == 'classification':
            lb = preprocessing.LabelBinarizer()
            n_classes = Y_list[0].shape[1]
            lb.fit(range(0, n_classes))
        if model_type == 'classification' and stratify:
            dataset_class_list = []
            dataset_class_ratio_list = []
            dataset_class_index_list = []
            for i in range(0, n_datasets):
                dataset_class = []
                dataset_class_ratio = []
                dataset_class_index = []
                dataX = X_list[i]
                dataY = Y_list[i]
                if n_classes == 2:
                    class_y = lb.inverse_transform(dataY[:,0])
                else:
                    class_y = lb.inverse_transform(dataY)
                for l in range(0, n_classes):
                    index = np.where(class_y == l)[0].tolist()
                    random.shuffle(index)
                    dataset_class.append(index)
                    ratio = int((float(len(index))/len(dataY)) * (batch_size + 4))
                    dataset_class_ratio.append(ratio)
                    dataset_class_index.append(0)
                dataset_class_list.append(dataset_class)
                dataset_class_index_list.append(dataset_class_index)
                dataset_class_ratio_list.append(dataset_class_ratio)
        while range(max_epochs):
            dataX_all = []
            dataY_all = []
            for i in range(0, n_datasets):
                #print(i)
                dataX = X_list[i]
                dataY = Y_list[i]
                n_datapoints = len(dataX)
                if n_datapoints <= batch_size:
                    sampX = dataX
                    sampY = dataY
                elif stratify:
                    if model_type == 'classification':
                        class_index = []
                        for k in range(0, n_classes):
                            end_index = dataset_class_index_list[i][k] + dataset_class_ratio_list[i][k]
                            if end_index > len(dataset_class_list[i][k]):
                                random.shuffle(dataset_class_list[i][k])
                                dataset_class_index_list[i][k] = 0
                                end_index = dataset_class_ratio_list[i][k]
                            permute_index = dataset_class_list[i][k][ dataset_class_index_list[i][k]:end_index ]
                            dataset_class_index_list[i][k] = end_index
                            class_index = class_index + permute_index
                        class_index = class_index[:batch_size]
                        sampX = dataX[class_index, :]
                        sampY = dataY[class_index, :]
                    elif model_type == 'regression':
                        _, sampX, _, sampY = train_test_split(dataX, dataY,
                                                              stratify=binarize_labels(dataY),
                                                              test_size=batch_size,
                                                              random_state=rs)
                else:
                    permute_index = rs.permutation(n_datapoints)[:batch_size]
                    sampX = dataX[permute_index, :]
                    sampY = dataY[permute_index, :]
                dataX_all.append(sampX)
                dataY_all.append(sampY)
            embed_sizes = check_dims([len(source) for source in dataX_all])
            max_epochs = max_epochs - 1
            yield np.vstack(dataX_all), np.vstack(dataY_all), embed_sizes
    else:
        while range(max_epochs):
            max_epochs = max_epochs - 1
            yield None, None, None

def repeater(embed, size):
    size = tf.cast(size, tf.int32)
    tile_embed = tf.tile(embed, size, name='tile')
    output = tf.reshape(tile_embed, [tf.squeeze(size), tf.shape(embed)[0]])
    return output

def while_repeater(n_embed, object_repeat, sizes, start_arr):
    condition = lambda i, x: tf.less(i, n_embed)
    body = lambda i, x: (i+1, tf.concat([x, repeater(object_repeat[i], sizes[i])], 0))
    i, repeat_arr = tf.while_loop(condition, body, (tf.constant(1), start_arr))
    return repeat_arr
        
def multivariate_normal(x, mu, L):
    """
    Computes the log-density of a multivariate normal.
    :param x  : Dx1 or DxN sample(s) for which we want the density
    :param mu : Dx1 or DxN mean(s) of the normal distribution
    :param L  : DxD Cholesky decomposition of the covariance matrix
    :return p : (1,) or (N,) vector of log densities for each of the N x's and/or mu's
    x and mu are either vectors or matrices. If both are vectors (N,1):
    p[0] = log pdf(x) where x ~ N(mu, LL^T)
    If at least one is a matrix, we assume independence over the *columns*:
    the number of rows must match the size of L. Broadcasting behaviour:
    p[n] = log pdf of:
    x[n] ~ N(mu, LL^T) or x ~ N(mu[n], LL^T) or x[n] ~ N(mu[n], LL^T)
    """
    if x.shape.ndims is None:
        logger.warn('Shape of x must be 2D at computation.')
    elif x.shape.ndims != 2:
        raise ValueError('Shape of x must be 2D.')
    if mu.shape.ndims is None:
        logger.warn('Shape of mu may be unknown or not 2D.')
    elif mu.shape.ndims != 2:
        raise ValueError('Shape of mu must be 2D.')

    d = x - mu
    alpha = tf.matrix_triangular_solve(L, d, lower=True)
    num_dims = tf.cast(tf.shape(d)[0], L.dtype)
    p = - 0.5 * tf.reduce_sum(tf.square(alpha), 0)
    p -= 0.5 * num_dims * np.log(2 * np.pi)
    p -= tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
    return p

def make_stable(A):
    A = tf.check_numerics(A, 'A')
    A = (A + tf.transpose(A))/2.
    e,v = tf.self_adjoint_eig(A)
    e = tf.where(e > 1e-6, e, 1e-6*tf.ones_like(e))
    A_stable = tf.matmul(tf.matmul(v,tf.matrix_diag(e)),tf.transpose(v))
    return A_stable

def loop(a):
    while range(a):
        a = a - 1
        yield None