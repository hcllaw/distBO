# Data class
from __future__ import print_function, division

import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import LabelBinarizer

def check_dims(array):
    if np.ndim(array) == 1:
        array = np.expand_dims(array, 1)
    return array

class data():
    def __init__(self, train_x, test_x,
                       train_y, test_y,
                       name=None, path=None, index=None,
                       embed_size=None, random_state=None,
                       prob_type='regression'):
        self.train_x = train_x
        self.test_x = test_x
        self.prob_type = prob_type
        if prob_type == 'regression':
            self.train_y = check_dims(train_y)
            self.test_y = check_dims(test_y)
            self.lb = None
        elif prob_type == 'classification':
            num_classes = len(np.unique(train_y))
            self.lb = LabelBinarizer()
            if num_classes == 2: #Ensure we make two columns even for binary problem.
                self.lb.fit(np.squeeze(train_y).tolist() + [2])
            else:
                self.lb.fit(train_y)
            self.train_y = self.lb.transform(train_y)
            self.test_y = self.lb.transform(test_y)
            if num_classes == 2: # remove last column, to ensure format in n_datapoints by 2
                self.train_y = self.train_y[:, :-1]
                self.test_y = self.test_y[:, :-1]
        else:
            raise ValueError('Must be regression or classification')
        if embed_size is None or len(train_x) <= embed_size:
            self.embed_ind = range(0, len(train_x))
        else:
            if random_state is None:
                random_state = check_random_state(23)
            full_ind = np.arange(len(train_x))
            if prob_type == 'regression': # TODO: Make this stratify, if we have large dataset
                self.embed_ind = random_state.permutation(full_ind)[:embed_size].tolist()
            elif prob_type == 'classification':
                embed_ind = []
                for i in range(0, num_classes):
                    class_y_ind = full_ind[np.squeeze(train_y) == i]
                    class_prob = len(class_y_ind) / len(train_y)
                    class_embed_size = int(class_prob * (embed_size+10)) # HACK so that we always have more!
                    print('class {}: {}, embed_size: {}'.format(i, class_prob, class_embed_size))
                    embed_ind = embed_ind + random_state.permutation(class_y_ind)[:class_embed_size].tolist()
                self.embed_ind = embed_ind[:embed_size] 
        self.embed = None 
        if name is not None:
            self.name = name
        if path is not None:
            self.path = path
        if index is not None:
            self.index = index

    def __len__(self):
        print('Training and Testing total length')
        return len(self.train_x) + len(self.test_x)

    def train_len(self):
        return len(self.train_x)

    def dim(self):
        return self.train_x.shape[1]