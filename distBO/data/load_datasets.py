from __future__ import print_function, division
import os
from os.path import dirname as up

import numpy as np
from sklearn import preprocessing
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from distBO.utils import (standardise, data_split, stack_tr_test, 
                          unpickle, get_median_sqdist, binarize_labels)
from distBO.data.data import data

def load_protein(target, random_state=None, preprocess='standardise', test_size=0.2):
    dir_path = [os.path.join(up(up(up(os.path.abspath(__file__)))), 'protein')]
    print(dir_path)
    data_list = []
    supp_list = []
    for path in dir_path:
        if os.path.exists(path):
            for i, filename in enumerate(sorted(os.listdir(path))):
                file_path = os.path.join(path, filename)
                protein = np.genfromtxt(file_path, delimiter=',')
                supp_name = filename
                protein_x = protein[:,1:-1]
                # Print binary data, so we do not rescale.
                protein_y = np.expand_dims(protein[:, -1], 1)
                print('Data {}'.format(i), 'Size:', len(protein_y),
                      'Ratio:', np.mean(protein_y==0), np.mean(protein_y==1))
                # Stratify train/test split
                train_x, test_x, train_y, test_y = train_test_split(protein_x, protein_y, stratify=protein_y,
                                                                    test_size=test_size,
                                                                    random_state=random_state)
                print('Data {}'.format(i), 'Ratio:', np.mean(train_y==0), np.mean(train_y==1))
                protein_data = data(train_x, test_x, train_y, test_y,
                                    prob_type='classification',
                                    random_state=random_state, name=supp_name)
                data_list.append(protein_data)
                supp_list.append(supp_name)
    total_group = 7 
    assert target < total_group, 'Target index is more than total group.'
    index = range(0, total_group)
    index.remove(target)
    return data_list[target], [data_list[i] for i in index], [supp_list[i] for i in [target] + index]

def load_parkinson(target, random_state=None, preprocess='standardise', 
                   label='total', test_size=0.2):
    dir_path = [os.path.join(up(up(up(os.path.abspath(__file__)))), 'parkinson')]
    data_list = []
    supp_list = []
    if label == 'both':
        label_list = ['total', 'motor']
        total_group = 84
    else:
        label_list = [label]
        total_group = 42
    for label in label_list:
        for ind in range(0, 42):
            for path in dir_path:
                if os.path.exists(path):
                    file_path = os.path.join(path, 'patient_{}.npy'.format(ind))
                    patient = np.load(file_path)
            patient_x = patient[:,:-2]
            patient_x = patient_x[:,2:] # Dropping age and sex, same for all patients.
            print(patient_x.shape)
            if preprocess == 'standardise':
                patient_x = preprocessing.scale(patient_x)
            if label == 'motor':
                patient_y = np.expand_dims(patient[:, -2], 1)
            elif label == 'total':
                patient_y = np.expand_dims(patient[:, -1], 1)
            patient_y = standardise(patient_y)
            train_x, test_x, train_y, test_y = train_test_split(patient_x, patient_y,
                                                                test_size=test_size,
                                                                stratify=binarize_labels(patient_y),
                                                                random_state=random_state)
            name = 'pat_{}_{}'.format(label, ind)
            patient_data = data(train_x, test_x, train_y, test_y, name=name)
            data_list.append(patient_data)
            supp_list.append(name)
    assert target < total_group, 'Target index is more than total group.'
    index = range(0, total_group)
    index.remove(target)
    return data_list[target], [data_list[i] for i in index], [supp_list[i] for i in [target] + index]