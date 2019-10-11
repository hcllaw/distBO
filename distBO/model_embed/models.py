# Models to run
from __future__ import division, print_function
import os
from functools import partial

import numpy as np 
import tensorflow as tf
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist

def set_model(model, data, init):
    previous_evals = None
    if model == 'ridge':
        target = partial(ridge, data)
        alpha_bounds = (np.log(10.0**-10), np.log(0.1))
        gamma_bounds = (np.log(2.0**-7), np.log(2.0**5))
        param_bounds = {'log_alpha': alpha_bounds, 'log_gamma': gamma_bounds}
        model_type = 'regression'
        if init is not None:
            previous_evals = {'target': init[:,-1], 
                              'log_alpha': init[:,0],
                              'log_gamma': init[:,1]}
    elif model == 'toy':
        target = partial(n_toy, data)
        param_bounds = {'theta': (-8, 8)}
        if init is not None:
            previous_evals = {'target': init[:,-1], 'theta': init[:,0]}
        model_type = 'regression'
    elif model == 'forest':
        target = partial(random_forest, data)
        model_type = 'classification'
        n_trees_bounds = (1, 200)
        depth_bounds = (1, 32)
        min_samples_split = (0.01, 1.0)
        min_samples_leaf = (0.01, 0.5)
        param_bounds = {'n_trees': n_trees_bounds, 'depth': depth_bounds, 
                        'min_samples_leaf': min_samples_leaf,
                        'min_samples_split': min_samples_split}
        if init is not None:
            previous_evals = {'target': init[:,-1], 'depth': init[:,0], 
                              'min_samples_leaf': init[:,1],
                              'min_samples_split': init[:,2], 
                              'n_trees': init[:,3]}
    elif model == 'svm':
        target = partial(svm, data)
        model_type = 'classification'
        log_C_bounds = (np.log(2**-7), np.log(2**10))
        log_gamma_bounds = (np.log(2.0**-7), np.log(2**5))
        param_bounds = {'log_C': log_C_bounds, 'log_gamma': log_gamma_bounds}
        if init is not None:
            previous_evals = {'target': init[:,-1], 'log_C': init[:,0], 
                              'log_gamma': init[:,1]}
    elif model == 'jaccard_svm':
        target = partial(jaccard_svm, data)
        model_type = 'classification'
        log_C_bounds = (np.log(2**-7), np.log(2**10))
        param_bounds = {'log_C': log_C_bounds}
        if init is not None:
            previous_evals = {'target': init[:,-1], 'log_C': init[:,0]}
    elif model == 'logistic':
        target = partial(logistic, data)
        model_type = 'classification'
        log_C_bounds = (np.log(2**-7), np.log(2**10))
        log_gamma_bounds = (np.log(2.0**-7), np.log(2**5))
        param_bounds = {'log_C': log_C_bounds, 'log_gamma': log_gamma_bounds}
        if init is not None:
            previous_evals = {'target': init[:,-1], 'log_C': init[:,0], 
                              'log_gamma': init[:,1]}
    elif model == 'dim_ridge':
        target = partial(dim_ridge, data)
        model_type = 'regression'
        log_alpha = (np.log(10.0**-8), np.log(0.1))
        bounds = (np.log(2.0**-7), np.log(2.0**6))
        param_bounds = {'log_bw1': bounds, 'log_bw2': bounds,
                        'log_bw3': bounds, 'log_bw4': bounds,
                        'log_bw5': bounds, 'log_bw6': bounds,
                        'log_alpha': log_alpha}
        if init is not None:
            previous_evals = {'target': init[:,-1], 'log_bw1': init[:,1], 
                              'log_bw2': init[:,2], 'log_bw3': init[:,3],
                              'log_bw4': init[:,4], 'log_bw5': init[:,5],
                              'log_bw6': init[:,6], 'log_alpha': init[:,0]}
    elif model == 'dim_logistic':
        target = partial(dim_logistic, data)
        model_type = 'classification'
        log_C_bounds = (np.log(2**-7), np.log(2**10))
        bounds = (np.log(2.0**-3), np.log(2.0**5))
        param_bounds = {'log_bw1': bounds, 'log_bw2': bounds,
                        'log_bw3': bounds, 'log_bw4': bounds,
                        'log_bw5': bounds, 'log_bw6': bounds,
                        'log_C': log_C_bounds}
        if init is not None:
            previous_evals = {'target': init[:,-1], 'log_C': init[:,0], 
                              'log_bw1': init[:,1], 'log_bw2': init[:,2], 
                              'log_bw3': init[:,3], 'log_bw4': init[:,4],
                              'log_bw5': init[:,5], 'log_bw6': init[:,6]}
    return target, param_bounds, previous_evals, model_type

def logistic(data, log_C, log_gamma):
    lb = data.lb
    train_y = lb.inverse_transform(data.train_y)
    test_y = lb.inverse_transform(data.test_y)
    print('Running Logistic Regression')
    C = np.exp(log_C)
    gamma = np.exp(log_gamma)
    print('Training with C:{}, gamma:{}'.format(C, gamma))
    rbf_feature = RBFSampler(gamma=gamma, n_components=200, random_state=0)
    trans_tr_x = rbf_feature.fit_transform(data.train_x)
    trans_test_x = rbf_feature.transform(data.test_x)
    clf = LogisticRegression(random_state=0, solver='lbfgs', 
                             multi_class='multinomial', C=C)
    clf.fit(trans_tr_x, train_y)
    te_predict = clf.predict_proba(trans_test_x)
    return roc_auc_score(data.test_y, te_predict)

def jaccard_kernel(features1, features2):
    kernel = 1 - cdist(features1, features2, 'jaccard')
    return kernel
    
def jaccard_svm(data, log_C):
    lb = data.lb
    train_y = lb.inverse_transform(data.train_y)
    test_y = lb.inverse_transform(data.test_y)
    print('Running SVM')
    C = np.exp(log_C)
    print('Training with C:{}'.format(C))
    jacaard_train_k = jaccard_kernel(data.train_x, data.train_x)
    jacaard_test_k = jaccard_kernel(data.test_x, data.train_x)
    clf = SVC(C=C, kernel='precomputed')
    #clf = CalibratedClassifierCV(clf)
    clf.fit(jacaard_train_k, train_y)
    return clf.score(jacaard_test_k, test_y)

def svm(data, log_C, log_gamma):
    lb = data.lb
    train_y = lb.inverse_transform(data.train_y)
    test_y = lb.inverse_transform(data.test_y)
    print('Running SVM')
    C = np.exp(log_C)
    gamma = np.exp(log_gamma)
    print('Training with C:{}, gamma:{}'.format(C, gamma))
    rbf_feature = RBFSampler(gamma=gamma, n_components=50, random_state=0)
    trans_tr_x = rbf_feature.fit_transform(data.train_x)
    trans_test_x = rbf_feature.transform(data.test_x)
    clf = LinearSVC(C=C)
    clf = CalibratedClassifierCV(clf)
    clf.fit(trans_tr_x, train_y)
    return clf.score(trans_test_x, test_y)

def dim_logistic(data, log_C, log_bw1, log_bw2, 
                 log_bw3, log_bw4, log_bw5, log_bw6):
    lb = data.lb
    train_y = lb.inverse_transform(data.train_y)
    test_y = lb.inverse_transform(data.test_y)
    C = np.exp(log_C)
    bw1 = np.exp(log_bw1)
    bw2 = np.exp(log_bw2)
    bw3 = np.exp(log_bw3)
    bw4 = np.exp(log_bw4)
    bw5 = np.exp(log_bw5)
    bw6 = np.exp(log_bw6)
    bw = np.array([bw1, bw2, bw3, bw4, bw5, bw6])
    print('Training with C:{}, bw:{}'.format(C, bw))
    rbf_feature = RBFSampler(gamma=0.5, n_components=200, random_state=0)
    trans_tr_x = rbf_feature.fit_transform(np.divide(data.train_x, bw))
    trans_test_x = rbf_feature.transform(np.divide(data.test_x, bw))
    clf = LogisticRegression(random_state=0, solver='lbfgs', C=C)
    clf.fit(trans_tr_x, train_y)
    te_predict = clf.predict_proba(trans_test_x)
    return roc_auc_score(data.test_y, te_predict)

def dim_ridge(data, log_alpha, log_bw1, log_bw2, 
              log_bw3, log_bw4, log_bw5, log_bw6):
    alpha = np.exp(log_alpha)
    bw1 = np.exp(log_bw1)
    bw2 = np.exp(log_bw2)
    bw3 = np.exp(log_bw3)
    bw4 = np.exp(log_bw4)
    bw5 = np.exp(log_bw5)
    bw6 = np.exp(log_bw6)
    bw = np.array([bw1, bw2, bw3, bw4, bw5, bw6])
    print('Training with alpha:{}, bw:{}'.format(alpha, bw))
    rbf_feature = RBFSampler(gamma=0.5, n_components=200, random_state=0)
    trans_tr_x = rbf_feature.fit_transform(np.divide(data.train_x, bw))
    trans_test_x = rbf_feature.transform(np.divide(data.test_x, bw))
    clf = Ridge(alpha=alpha)
    clf.fit(trans_tr_x, data.train_y)
    score = clf.score(trans_test_x, data.test_y)
    return max(score, -1.0)

def ridge_gamma(data, log_gamma):
    alpha = 5.0e-07#2.0e-06#6.25e-07 # sigma^2/n
    gamma = np.exp(log_gamma)
    print('Training with alpha:{}, gamma:{}'.format(alpha, gamma))
    np.random.seed(23)
    rbf_feature = RBFSampler(gamma = gamma, n_components=200)
    trans_tr_x = rbf_feature.fit_transform(data.train_x)
    trans_test_x = rbf_feature.transform(data.test_x)
    clf = Ridge(alpha=alpha)
    clf.fit(trans_tr_x, data.train_y)
    score = clf.score(trans_test_x, data.test_y)
    return max(score, -1.0)

def ridge(data, log_alpha, log_gamma):
    alpha = np.exp(log_alpha)
    gamma = np.exp(log_gamma)
    print('Training with alpha:{}, gamma:{}'.format(alpha, gamma))
    np.random.seed(23)
    rbf_feature = RBFSampler(gamma = gamma, n_components=200)
    trans_tr_x = rbf_feature.fit_transform(data.train_x)
    trans_test_x = rbf_feature.transform(data.test_x)
    clf = Ridge(alpha=alpha)
    clf.fit(trans_tr_x, data.train_y)
    score = clf.score(trans_test_x, data.test_y)
    test_predict_y = clf.predict(trans_test_x)
    return max(score, -1.0)

def random_forest(data, depth, min_samples_leaf, min_samples_split, n_trees):
    print('Training with depth:{}, min_samples_leaf:{},\
           min_samples_leaf:{}, min_samples_split, n_trees'.format(int(depth), 
                                                                   min_samples_leaf, 
                                                                   min_samples_split,
                                                                   int(n_trees)))
    forest = RandomForestClassifier(n_estimators=int(n_trees), 
                                    max_depth=int(depth),
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf)
    forest.fit(data.train_x, data.train_y[:,0])
    te_predict = np.array(forest.predict_proba(data.test_x)[:,1])
    return roc_auc_score(data.test_y[:,0], te_predict)

def n_toy(data, theta):
    tr_test_x = np.vstack((data.train_x, data.test_x))
    sample_mean = np.squeeze(np.mean(tr_test_x))
    return np.exp(-0.5 * (theta - sample_mean) ** 2) 
