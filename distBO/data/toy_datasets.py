from __future__ import print_function, division
from functools import partial
from itertools import combinations
import os
import pickle 

import numpy as np
from sklearn import preprocessing
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel

from distBO.utils import (increase_dim, rotate_orth, standardise, 
                          data_split, get_median_sqdist, unpickle, 
                          PSD, round_list)
from distBO.data.data import data
from distBO.model_embed.meta_fea import counter_dataset

def counter_dataset_make(n_train, n_test, dim, sig_dim=3, seed=23,
                         noise=0.5, name=None, embed_size=10000,
                         preprocess='standardise'):
    n_total = n_train + n_test
    data, label = counter_dataset(sig_dim, noise=noise, data_size=n_total,
                                  dimension=dim, random_seed=seed)
    if preprocess == 'standardise':
        data = preprocessing.scale(data)
        label = standardise(label)
    rs = check_random_state(seed)
    dataset = data_split(data, label, n_train, n_test, rs, 
                         embed_size=embed_size, name=name)
    return dataset
 
def one_d_split(n_train, n_test, sd=1.0, dim=None, 
                preprocess='standardise', embed_size=10000,
                true_dist=True, seed=23):
    rs = check_random_state(seed)
    n_total = n_train + n_test
    if true_dist:
        mu = rs.normal(loc=0.0, scale=sd, size=1)[0]
    else:
        mu = rs.normal(loc=4.0, scale=sd, size=1)[0]
    X = rs.normal(loc=mu, scale=1.0, size=(n_total, 1))
    # Y is ignored later.
    Y = np.array([rs.normal(loc=x, scale=1.0, size=1) for x in np.nditer(X)])
    dataset = data_split(X, Y, n_train, n_test, rs, embed_size=embed_size)
    return dataset, mu

def dim_bw_dataset(n_train, n_test, dim, prob_type='regression', embed_size=10000, name=None,
                   preprocess='standardise', bw_set=None, noise_sd=0.5, seed=23):
    n_total = n_train + n_test
    rs = check_random_state(seed)
    X = rs.normal(loc=0.0, scale=1.0, size=(n_total, dim))
    data_x = preprocessing.scale(X)
    signal_x_bw = np.divide(X, np.array(bw_set))
    #  1 / (2 sigma^2) = gamma i.e sigma = 1 implies gamma = 0.5
    # sigma = sqrt( 1.0 / 2.0 * gamma)
    rbf_feature = RBFSampler(gamma=0.5, n_components=200, random_state=0)
    trans_signal_x_bw = rbf_feature.fit_transform(signal_x_bw)
    alpha = rs.normal(loc=0.0, scale=1.0, size=(200))
    y_0 = np.matmul(trans_signal_x_bw, alpha)
    if prob_type == 'regression':
        y_0 = standardise(y_0, low=0.0, high=1.0)
        y = y_0 + rs.normal(loc=0.0, scale=noise_sd, size=(n_total))
        label = standardise(y)
        dataset = data_split(data_x, label, n_train, n_test, rs)
    elif prob_type == 'classification':
        y_0 = standardise(y_0, low=-6.0, high=6.0)
        y = y_0
        prob = 1.0 / (1.0 + np.exp(-y))
        uni_values = rs.uniform(low=0.0, high=1.0, size=len(prob))
        label = (uni_values > prob).astype(int)
        train_x, test_x, train_y, test_y = train_test_split(data_x, label, stratify=label,
                                                            test_size=float(n_test)/n_total,
                                                            random_state=rs)
        dataset = data(train_x, test_x, train_y, test_y, name=name,
                       embed_size=embed_size, prob_type='classification')
    return dataset

def bw_source_gen(args, base_config, seeds, rm_source_near_t=None):
    rs = check_random_state(args.data_seed)
    bw_choices = np.exp(np.linspace(np.log(2.0**-1), np.log(2.0**4), num=6))
    bw_set_array = rs.choice(bw_choices,
                             size=(args.source_num, 6),
                             replace=True)
    noise_sd_list = [0.0] * args.source_num
    source_list = []
    supp_list = []
    for i in range(0, len(noise_sd_list)):
        name = '{}_log_bw_{}_noise_{}'.format(args.prob_type, 
                                              round_list(bw_set_array[i], 2),
                                              round(noise_sd_list[i], 2))
        source_data = dim_bw_dataset(bw_set=bw_set_array[i], noise_sd=noise_sd_list[i],
                                     seed=seeds[i], prob_type=args.prob_type, name=name, 
                                     **base_config)
        source_list.append(source_data)
        supp_list.append(name)
    return source_list, supp_list