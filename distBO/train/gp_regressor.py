# GP class
from __future__ import division, print_function
from functools import partial 
from math import log
import os

import numpy as np
import tensorflow as tf
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from distBO.train.log_gaus_likelihood import build_log_marg_likhd, build_predict, build_predict_dist
from distBO.train.log_gaus_likelihood_feature import build_log_marg_likhd_feature, build_predict_feature, build_predict_dist_feature
from distBO.train.train import train_network, eval_network, sim_network, embed_network
from distBO.utils import tf_session, check_dims, sample, get_median_sqdist

# Training, Prediction, Similarity

class gp_regressor(object):
    def __init__(self, kernel, n_cpus=1, target_X=None, target_Y=None, 
                 model_type=None, seed=None, target_embed_ind=None, float64=False):
        self.model_type = model_type
        self.kernel = kernel
        self.n_cpus = n_cpus
        self.dtype = tf.float64 if float64 else tf.float32
        self.seed = seed
        self.initialise = True
        self.params_scaler = None
        if target_X is not None:
           self.data_sizes_target = check_dims([ len(target_X) ])
           self.target_embed_ind = target_embed_ind
           self.target_X, self.target_Y = target_X, target_Y

    def maximise_likhd(self, params, y, data_X=None, data_Y=None, data_embed=None, 
                             source_embed_ind=None, embed_op='joint', algorithm='GP',
                             weight_dim=None, weight_dim_x=None, num_of_rff=100,
                             weight_dim_xy=None, weight_dim_y=None, batch_size=250,
                             sizes=None, target_embed=None, concat_data_size=False,
                             learning_rate=0.001, max_epochs=200, max_size=None,
                             bw_params_init=1.0, likhd_var_init=0.01, stratify=True,
                             gradient_clip=False, first_early_stop_epoch=30):
        assert len(params) == len(y), 'Configurations and y must be same length'
        self.target_embed = target_embed
        self.algorithm = algorithm
        if self.kernel == 'dist':
            data_sizes = check_dims([ len(data) for data in data_X ])
            assert len(params) == np.sum(sizes), 'Total sizes must match with number of configurations'
            assert len(data_X) == len(sizes), 'Number of datasets must equal number of sizes'
            assert len(data_X) == len(data_Y), 'Number of datasets X must be equal to number of labels'
            for m, n in zip(data_X, data_Y):
                assert len(m) == len(n), 'Number of datapoints must equal number of labels'
            self.in_dim_x = np.shape(data_X[0])[1]
            self.in_dim_y = np.shape(data_Y[0])[1]
        if sizes is not None:
            sizes = check_dims(sizes)
        self.in_dim_params = np.shape(params)[1]

        if self.params_scaler is None:
            print('Scaling Parameters')
            self.params_scaler = StandardScaler()
            self.params_scaler.fit(params)
            params = self.params_scaler.transform(params)
        else:
            params = self.params_scaler.transform(params)

        if algorithm == 'GP':
            print('Using GP')
            build_likhd_func = partial(build_log_marg_likhd,
                                       bw_params_init=bw_params_init)
        elif algorithm == 'BLR':
            print('Using BLR')
            build_likhd_func = partial(build_log_marg_likhd_feature,
                                       weight_dim=weight_dim,
                                       num_of_rff=num_of_rff)
        
        build_likhd = partial(build_likhd_func,
                              model_type=self.model_type,
                              in_dim_params=self.in_dim_params,
                              weight_dim_x=weight_dim_x,
                              weight_dim_y=weight_dim_y,
                              weight_dim_xy=weight_dim_xy,
                              embed_op=embed_op,
                              log_sigma_init=0.5*log(likhd_var_init),
                              dtype=self.dtype, seed=self.seed)

        train = {'h_params': params, 'obs': y}
        # Distribution kernel
        if self.kernel == 'dist':
            if self.initialise:
                self.sess = tf_session(n_cpus=self.n_cpus)
                self.net = build_likhd(in_dim_x=self.in_dim_x,
                                       in_dim_y=self.in_dim_y,
                                       concat_data_size=concat_data_size,
                                       use_dist_kernel=True)
                self.samp_target_X, self.samp_target_Y = sample(self.target_X, self.target_Y,
                                                                embed_ind=self.target_embed_ind,
                                                                source_list=False)
            self.source_X, self.source_Y = sample(data_X, data_Y,
                                                  embed_ind=source_embed_ind,
                                                  source_list=True)
            self.samp_source_X = np.vstack(self.source_X)
            self.samp_source_Y = np.vstack(self.source_Y)
            train['X'] = data_X #self.source_X #data_X
            train['y'] = data_Y
            train['data_sizes'] = data_sizes
            train['sizes'] = sizes
            if concat_data_size:
                train['max_size'] = max_size
            else:
                train['max_size'] = None
        # Manual kernel 
        elif self.kernel == 'manual':
            if self.initialise:
                self.sess = tf_session(n_cpus=self.n_cpus)
                self.net = build_likhd(in_dim_meta=len(target_embed[0]),
                                       use_manual_kernel=True)
            train['embed'] = data_embed
            train['sizes'] = sizes
        # Hyperparameter kernel
        elif self.kernel == 'params':
            if self.initialise:
                self.sess = tf_session(n_cpus=self.n_cpus)
                self.net = build_likhd()
        # Multi-task kernel
        elif self.kernel == 'multi':
            if self.initialise:
                self.sess = tf_session(n_cpus=self.n_cpus)
                self.net = build_likhd(use_task_kernel=True, n_tasks=len(sizes))
            train['sizes'] = sizes

        # Training
        print('Training')
        self.train = train
        loss = train_network(self.sess, self.net, self.train,
                             initialise=self.initialise,
                             model_type=self.model_type,
                             batch_size=batch_size,
                             max_epochs=max_epochs, stratify=stratify,
                             gradient_clip=gradient_clip,
                             first_early_stop_epoch=first_early_stop_epoch,
                             optimizer=tf.train.AdamOptimizer, seed=self.seed,
                             lr=learning_rate, display_every=100)
        self.initialise = False
        self.update_sum_matrix = True
        self.pre_compute_embed = True
        return loss

    def initialise_predict(self): # So we dont intialise the net multiple times.
        if self.algorithm == 'GP':
            self.predict_net = build_predict(self.net)
            if self.kernel == 'dist':
                self.dist_predict_net = build_predict_dist(self.net)
        elif self.algorithm == 'BLR':
            self.predict_net = build_predict_feature(self.net)
            if self.kernel == 'dist':
                self.dist_predict_net = build_predict_dist_feature(self.net)

    # TODO: Prediction time should not re-feed embedding.
    def predict_y(self, params_pred, refresh_net=False, compute_embed=False):
        # Transform also the predicted hyperparameters.
        params_pred = self.params_scaler.transform(params_pred)
        if refresh_net or not hasattr(self, 'predict_net'):
            initialise_predict(self)
        test = {'h_params_pred': params_pred}
        if self.kernel == 'dist':
            self.train['batch_X'] = self.samp_source_X
            self.train['batch_y'] = self.samp_source_Y
            self.train['embed_sizes'] = check_dims([len(source) for source in self.source_X])
            test['X_pred'], test['y_pred'] = self.samp_target_X, self.samp_target_Y
            test['embed_sizes_pred'] = check_dims([len(self.target_embed_ind)])
            test['sizes_pred'] = check_dims([len(params_pred)])
            test['data_sizes_pred'] = self.data_sizes_target
        elif self.kernel == 'multi':
            test['sizes_pred'] = check_dims([len(params_pred)])
        elif self.kernel == 'manual':
            test['sizes_pred'] = check_dims([len(params_pred)])
            test['embed_pred'] = self.target_embed
        if (compute_embed or self.pre_compute_embed) and self.kernel == 'dist':
            self.dist, self.dist_pred = embed_network(self.sess, self.predict_net, 
                                                      self.train, test, algorithm=self.algorithm,
                                                      update_sum_matrix=self.update_sum_matrix)
            if compute_embed:
                self.pre_compute_embed = True
            else:
                self.pre_compute_embed = False
        if self.kernel == 'dist':
            self.train['dist'] = self.dist
            test['dist_pred'] = self.dist_pred
            mean, var = eval_network(self.sess, self.dist_predict_net, 
                                     self.train, test, use_embed=True,
                                     update_sum_matrix=self.update_sum_matrix)
        else:
            mean, var = eval_network(self.sess, self.predict_net, 
                                     self.train, test,
                                     update_sum_matrix=self.update_sum_matrix)
        self.update_sum_matrix = False
        std = np.sqrt(np.maximum(var, 1e-32)) # stability
        return mean, std

    def close(self):
        self.initialise = True
        tf.reset_default_graph()
        self.sess.close()

    def similarity(self, sample_sim=False):
        if self.algorithm == 'GP':
            if self.kernel == 'multi':
                test = None
            elif self.kernel == 'manual':
                test = {'embed_pred': self.target_embed}
            else:
                self.train['batch_X'] = self.samp_source_X
                self.train['batch_y'] = self.samp_source_Y
                self.train['embed_sizes'] = check_dims([len(source) for source in self.source_X])
                test = {'X_pred': self.samp_target_X, 'y_pred': self.samp_target_Y,
                        'embed_sizes_pred': check_dims([len(self.target_embed_ind)]),
                        'data_sizes_pred': self.data_sizes_target}
            sim = sim_network(self.sess, self.net, self.train, test, update_sum_matrix=True)
        elif self.algorithm == 'GP_check':
            sim = ['none']
        elif self.algorithm == 'BLR':
            sim = ['none']
        return sim