from __future__ import print_function, division
from functools import partial

import numpy as np
from sklearn.utils import check_random_state

import distBO.data.toy_datasets as toy_datasets
import distBO.data.load_datasets as load_datasets
from distBO.utils import round_list

def generate_data(args):
    # Real Data
    if args.dataset in ['patient_vary', 'protein_vary']:
        rs = check_random_state(args.data_seed)
        if args.dataset == 'patient_vary':
            args.model = 'ridge'
            target, source_list, supp = load_datasets.load_parkinson(args.target_group, random_state=rs,
                                                                     preprocess=args.preprocess,
                                                                     label=args.park_labels,
                                                                     test_size=args.test_size)
        elif args.dataset == 'protein_vary':
            args.model = 'forest' # Can also run with 'jaccard_svm', currently hard-coded...
            args.label_kernel = 'none'
            target, source_list, supp = load_datasets.load_protein(args.target_group, random_state=rs, 
                                                                   preprocess=args.preprocess,
                                                                   test_size=args.test_size)
    # Toy Dataset
    else:
        rs = check_random_state(args.data_seed)
        base_config = {'dim': args.dim, 'preprocess': args.preprocess,
                       'embed_size': args.max_embed_size, 'n_train': args.n_train,
                       'n_test': args.n_test}
        if args.dataset == 'one_dim_split':
            args.model = 'toy'
            source_num = args.fake_source_num + args.true_source_num
            data_seeds = rs.randint(2**32, size=source_num + 1)
            target, target_mu = toy_datasets.one_d_split(sd=args.mu_sd,
                                                         seed=data_seeds[0],
                                                         true_dist=True,
                                                         **base_config)
            source_seeds = data_seeds[1:]
            supp = [target_mu]
            source_list = []
            # Generate not same source
            for i in range(0, args.fake_source_num):
                source_fake, mu_fake = toy_datasets.one_d_split(sd=args.mu_sd, seed=source_seeds[i], 
                                                            true_dist=False, **base_config)
                source_list.append(source_fake)
                supp.append(mu_fake)
            # Generate same source
            for i in range(args.fake_source_num, source_num):
                source_true, mu_true = toy_datasets.one_d_split(sd=args.mu_sd, seed=source_seeds[i], 
                                                            true_dist=True, **base_config)
                source_list.append(source_true)
                supp.append(mu_true)
        # bw toy dataset
        elif args.dataset == 'bw_dim_vary':
            data_seeds = rs.randint(2**32, size=args.source_num + 1)
            source_seeds = data_seeds[1:]
            assert args.target_group in [-1] + range(0, args.source_num), 'must be -1 or less than source_num'
            remove_nearby = None
            base_config['dim'] = 6
            rs = check_random_state(args.data_seed)
            if args.target_group != -1:
                bw_choices = np.exp(np.linspace(np.log(2.0**-1), np.log(2.0**4), num=6))
                bw_set_array = rs.choice(bw_choices,
                                     size=(args.source_num, 6),
                                     replace=True)
                target_bw_set = bw_set_array[args.target_group]
            else:
                target_bw_set = np.array([1.5] * 6)
            if args.prob_type == 'regression':
                noise_sd = 0.05
            else:
                noise_sd = 0.0
            target = toy_datasets.dim_bw_dataset(noise_sd=noise_sd,
                                                 bw_set=target_bw_set,
                                                 seed=data_seeds[0],
                                                 prob_type=args.prob_type,
                                                 **base_config)
            target_name = '{}_log_bw_{}_noise_{}'.format(args.prob_type, 
                                                         round_list(target_bw_set, 2),
                                                         round(noise_sd, 2))

            source_list, supp = toy_datasets.bw_source_gen(args, base_config, source_seeds,
                                                           rm_source_near_t=remove_nearby)
            supp = [target_name] + supp
            if args.prob_type == 'regression':
                args.model = 'dim_ridge'
            elif args.prob_type == 'classification':
                args.model = 'dim_logistic'
                
        # Counter manual dataset
        elif args.dataset == 'counter_manual':
            print('Using dimension 6 for counter manual dataset')
            args.dim = 6
            base_config['dim'] = 6
            args.model = 'dim_ridge'
            data_seeds = rs.randint(2**32, size=4+1)
            source_seeds = data_seeds[1:]
            target_name = 'counter_noise_{}_dim_{}'.format(args.noise, 4)
            target = toy_datasets.counter_dataset_make(sig_dim=4, seed=data_seeds[0],
                                                       noise=args.noise, name=target_name, 
                                                       **base_config)
            source_list = []
            supp = [target_name]
            for i in range(1, args.dim - 1):
                source_name = 'counter_noise_{}_dim_{}'.format(args.noise, i+2)
                source = toy_datasets.counter_dataset_make(sig_dim=i+2, seed=data_seeds[i],
                                                           noise=args.noise, name=source_name,
                                                           **base_config)
                source_list.append(source)
                supp.append(source_name)
        else:
            raise ValueError("unknown dataset {}".format(args.dataset))
    print('Target dataset :', supp[0])
    print('\n'.join('Source dataset {}: {}'.format(*k) for k in enumerate(supp[1:])))
    return target, source_list, supp
