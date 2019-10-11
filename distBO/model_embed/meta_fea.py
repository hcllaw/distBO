# Script for computing meta features
from __future__ import print_function, division

import math

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from distBO.utils import get_median_sqdist

def meta(data_list, model, max_size=None, scaler=None, random_state=None, include_corr_meta=True):
    meta_data = []
    number = len(data_list)
    if model in ['forest', 'svm', 'logistic', 'jaccard_svm', 
                 'logistic_gamma', 'dim_logistic']:
        model_type = 'classification'
    elif model in ['dim_ridge', 'ridge', 'toy', 
                   'ridge_gamma', 'ridge_alpha']:
        model_type = 'regression'
    else:
        raise ValueError('Model not recognised')

    for k, source in enumerate(data_list):
        x = source.train_x[source.embed_ind]
        y = source.train_y[source.embed_ind]
        if model != 'toy':
            skew_data_meta = skewness_meta(x)
            kur_data_meta = kurtosis_meta(x)
            corr_cov_data_meta = corr_cov_meta(x, include_corr_meta=include_corr_meta)
            pca_data_meta = pca_meta(x, random_state=random_state)
            land_data_meta = landmark_meta(x, y, model_type=model_type, lb=source.lb,
                                           random_state=random_state)
            label_data_meta = label_meta(y, model_type=model_type, lb=source.lb)
            source_meta = skew_data_meta + kur_data_meta + pca_data_meta + \
                          land_data_meta + corr_cov_data_meta + label_data_meta
            if max_size is not None:
                size_ratio = len(source.train_x) / max_size
                source_meta = source_meta + [size_ratio]
            print(source_meta)
        else:
            mean_data = np.squeeze(np.mean(x, axis=0))
            std_data = np.squeeze(np.std(x, axis=0))
            skew_data = np.squeeze(skew(x, axis=0))
            kurt_data = np.squeeze(kurtosis(x, axis=0))
            source_meta = [mean_data, std_data, skew_data, kurt_data]
        meta_data.append(source_meta)
    meta_data = np.array(meta_data)
    # Remove columns that are all the same.
    meta_data = meta_data[:, ~np.all(meta_data[1:] == meta_data[:-1], axis=0)]
    print('Meta-Data shape', meta_data.shape)
    if scaler is None: # Map to [0,1]
        scaler = MinMaxScaler()
        scaler.fit(meta_data)
    meta_data = scaler.transform(meta_data)
    for i, source in enumerate(data_list):
        source.embed = meta_data[i,:]
    return scaler

def corr_cov_meta(dataset, include_corr_meta=True):
    # Correlation
    corr_mat = np.corrcoef(dataset.T)
    corr_vec = corr_mat[np.triu_indices(dataset.shape[1], k = 1)]
    if include_corr_meta:
        corr_meta = [np.min(corr_vec), np.max(corr_vec), np.mean(corr_vec), np.std(corr_vec)]
    else:
        corr_meta = []
    # Covariance
    cov_mat = np.cov(dataset.T)
    cov_vec = cov_mat[np.triu_indices(dataset.shape[1], k = 1)]
    cov_meta = [np.min(cov_vec), np.max(cov_vec), np.mean(cov_vec), np.std(cov_vec)]
    print('corr_cov_meta', corr_meta + cov_meta)
    return corr_meta + cov_meta

def skewness_meta(dataset):
    skew_data = skew(dataset, axis=0)
    skew_vec = [np.min(skew_data), np.max(skew_data), np.mean(skew_data), np.std(skew_data)]
    print('skewness_meta', skew_vec)
    return skew_vec 

# Uses fisher, subtracts 3 so that normal has 0 kurtosis
def kurtosis_meta(dataset):
    kurtosis_data = kurtosis(dataset, axis=0)
    kurtosis_vec = [np.min(kurtosis_data), np.max(kurtosis_data), np.mean(kurtosis_data), np.std(kurtosis_data)]
    return kurtosis_vec

def pca_meta(dataset, random_state=None):
    num_fea = np.shape(dataset)[1]
    pca = PCA(random_state=random_state)
    trans_data = pca.fit_transform(dataset)
    first_pc = trans_data[:,0] # First PC
    pc_skew = skew(first_pc)
    kurt_skew = kurtosis(first_pc)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    for i, per in enumerate(cumsum):
        if per > 0.95:
            intrinsic_dim = i + 1
            break
    pca_vec = [pc_skew, kurt_skew, float(intrinsic_dim)/num_fea]
    print('pca_meta', pca_vec)
    return pca_vec

def landmark_meta(dataset_x, dataset_y, lb=None, model_type='regression', random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, 
                                                        test_size=0.25, 
                                                        random_state=random_state)
    if model_type == 'regression':
        # 1 nearest neighbour
        neigh = KNeighborsRegressor(n_neighbors=1)
        neigh.fit(X_train, y_train)
        neigh_score = neigh.score(X_test, y_test)
        # Linear regression 
        linear = LinearRegression()
        linear.fit(X_train, y_train)
        linear_score = linear.score(X_test, y_test)
        # Decision Tree 
        tree = DecisionTreeRegressor(random_state=random_state)
        tree.fit(X_train, y_train)
        tree_score = tree.score(X_test, y_test)
        scores = [neigh_score, linear_score, tree_score]
    elif model_type == 'classification':
        y_train = lb.inverse_transform(y_train)
        y_test = lb.inverse_transform(y_test)
        # 1 nearest neighbour
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(X_train, y_train)
        neigh_score = neigh.score(X_test, y_test)
        # LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        lda_score = lda.score(X_test, y_test)
        # Decision Tree
        tree = DecisionTreeClassifier(random_state=random_state)
        tree.fit(X_train, y_train)
        tree_score = tree.score(X_test, y_test)
        # Naive Bayes (Gaussian)
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        nb_score = tree.score(X_test, y_test)
        scores = [neigh_score, lda_score, tree_score, nb_score]
    else:
        raise ValueError('Must be regression or classification')
    return scores

def label_meta(dataset_y, lb=None, model_type='regression'):
    if model_type == 'regression':
        label_vec = [np.mean(dataset_y), np.std(dataset_y), skew(dataset_y)[0], kurtosis(dataset_y)[0]]
    elif model_type == 'classification':
        # More features can be added
        dataset_y = lb.inverse_transform(dataset_y)
        dataset_y = dataset_y.astype(np.int64)
        count = np.bincount(dataset_y)
        ratio = count[count!=0] / len(dataset_y)
        entropy_data = entropy(ratio)
        label_vec = ratio.tolist() + [entropy_data]
    else:
        raise ValueError('Must be regression or classification')
    return label_vec

def counter_dataset(sig_dim, noise=0.5, data_size=1000, dimension=5, random_seed=23):
    assert sig_dim > 2
    assert sig_dim <= dimension
    np.random.seed(random_seed)
    data_x = np.random.normal(loc=0.0, scale=2.0, size=(data_size, dimension))
    data_x[:, sig_dim-1] = np.sign(np.multiply(data_x[:,0], data_x[:,1])) * np.absolute(data_x[:, sig_dim-1])
    data_y = np.log( 1 + (data_x[:, 0] * data_x[:, 1] * data_x[:, sig_dim-1])**3 ) + np.random.normal(size=data_size, scale=noise)
    return data_x, data_y

def all_meta(x, y, random_state=None):
    skew_data_meta = skewness_meta(x)
    kur_data_meta = kurtosis_meta(x)
    pca_data_meta = pca_meta(x, random_state=random_state)
    land_data_meta = landmark_meta(x, y, random_state=random_state)
    label_data_meta = label_meta(y)
    corr_cov_data_meta = corr_cov_meta(x)




