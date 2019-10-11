# Utilities for DistBO
from __future__ import division, print_function
from contextlib import contextmanager
import os 
import pickle

import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances

from distBO.data.data import data

def binarize_labels(labels, min_bin=0, max_bin=1.01, intervals=4):
    bins = np.linspace(min_bin, max_bin, intervals)
    y_binned = np.digitize(np.squeeze(labels), bins)
    return y_binned

def round_list(list_num, decimal_place):
    return [ round(num, decimal_place) for num in list_num]
    
def similarity_l2(source_data, target):
    source_embed = [s_x.embed for s_x in source_data]
    return [ -np.linalg.norm(target.embed - s_embed, ord = 2)
                                 for s_embed in source_embed ]

def PSD(A, reg=1e-3, tol=0.0):
    evals, eV = np.linalg.eig(A)
    evals = np.real(evals) #due to numerical error
    eV = np.real(eV)
    if not np.all(evals > tol): #small tolerance allowed
        if isinstance(reg, float) and reg > 0.0:
            ev_small = np.sort(evals[evals > 0])[0]
            evals[evals <= 0] = min(reg, ev_small) #if reg too large
        else:
            raise ValueError('float {} is not positive float'.format(reg))
    psd_A = eV.dot(np.diag(evals)).dot(eV.T) # reconstruction
    return psd_A

def prev_extractor(dataset, s_data, path, bo_source_type,
                   rand_it, bo_it, data_seed, opt_seed):
    folder_path = os.path.join(path, 'pre_eval_source', dataset)
    filename = 'd_seed_{}_o_seed_{}_s_type_{}_rand_it_{}_bo_it_{}_name_{}.pkl'.format(data_seed, opt_seed, 
                                                                                      bo_source_type,
                                                                                      rand_it, bo_it, s_data.name)
    evals = pickle.load(open(os.path.join(folder_path, filename), "rb" ))
    return evals

def triangular_vec(L, n=5, log_init=False):
    size = n * (n + 1) // 2 #int division
    x = np.arange(size)
    m = x.shape[0]
    x_tail = x[(m - (n**2 - m)):]
    triangular = np.concatenate([x_tail, x[::-1]], 0).reshape(n, n)
    vec = np.zeros(int(size))
    if L is not None:
        for i in range(n):
            vec[triangular[i:, i]] = L[i:, i]
    else:
        diag = np.diag(triangular)
        if log_init:
            vec = np.ones(int(size)) * -20.0
            vec[diag] = 0
        else:
            vec[diag] = 1
    vec.astype(np.float32)
    return vec

    
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def tf_session(n_cpus=1, config_args={}, **kwargs):
    import tensorflow as tf
    config = tf.ConfigProto(intra_op_parallelism_threads=n_cpus,
                            inter_op_parallelism_threads=n_cpus, **config_args)
    return tf.Session(config=config)

def sample(data_X, data_Y, embed_ind=None, source_list=True):
    assert len(data_X) == len(data_Y)
    if not source_list:
        return data_X[embed_ind], data_Y[embed_ind]
    else:
        samp_X = []
        samp_Y = []
        for i in range(0, len(data_X)):
            assert len(data_X[i]) == len(data_Y[i])
            samp_X.append(data_X[i][embed_ind[i]])
            samp_Y.append(data_Y[i][embed_ind[i]])
        return samp_X, samp_Y

def data_split(p_data, s_label, n_train, n_test,
               rs, embed_size=10000, name=None):
    n_total = n_train + n_test
    permute = rs.permutation(n_total)
    tr_ind = permute[:n_train]
    test_ind = permute[n_train:]
    return data(p_data[tr_ind], p_data[test_ind],
                s_label[tr_ind], s_label[test_ind], 
                embed_size=embed_size, name=name)

def check_dims(array):
    if np.ndim(array) == 1:
        array = np.expand_dims(array, 1)
    return array 

def standardise(labels, low=0.0, high=1.0):
    min_l = np.min(labels)
    max_l = np.max(labels)
    zero_one_range = (labels - min_l) / (max_l - min_l)
    return low + (zero_one_range * (high-low))

def increase_dim(X, dim):
    data_size, latent_dim = X.shape
    zeros_dim = dim - latent_dim
    zeros_data = np.zeros((data_size, zeros_dim))
    X_dim = np.hstack((X, zeros_data))
    return X_dim

def rotate_orth(data, return_orth=False, transpose=False, seed=23):
    rs = check_random_state(seed)
    dim = data.shape[1]
    random_matrix = rs.uniform(0, 2, (dim, dim))
    orth_matrix, R = np.linalg.qr(random_matrix)
    if transpose:
        orth_matrix = orth_matrix.T
    trans_data = np.dot(data, orth_matrix)
    if return_orth:
        return trans_data, orth_matrix
    else:
        return trans_data

def stack_tr_test(data_list, test=False):
    list_x = []
    list_y = []
    for data in data_list:
        list_x.append(data.train_x)
        list_y.append(data.train_y)
        if test:
            list_x.append(data.test_x)
            list_y.append(data.test_y) 
    return np.vstack(list_x), np.vstack(list_y)

def get_median_sqdist(all_Xs, n_sub=5000, seed=23):
    np.random.seed(seed)
    N = all_Xs.shape[0]
    sub = all_Xs[np.random.choice(N, min(n_sub, N), replace=False)]
    D2 = euclidean_distances(sub, squared=True)
    return np.median(D2[np.triu_indices_from(D2, k=1)], overwrite_input=True)
    
