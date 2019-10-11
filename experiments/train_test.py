from __future__ import print_function, division
import argparse
import os
import sys
import time
from copy import deepcopy

import numpy as np
import multiprocessing as mp
from sklearn.utils import check_random_state

# Data
from distBO.data.generate_data import generate_data
# Models and embeddings
from distBO.model_embed.bayes_optimise import bayes_opt_wrap
from distBO.model_embed.meta_fea import meta
# Utilities
from distBO.utils import merge_two_dicts, prev_extractor

__author__ = 'leon'

def get_adder(g):
    def f(*args, **kwargs):
        kwargs.setdefault('help', "Default %(default)s.")
        return g.add_argument(*args, **kwargs)
    return f

def _add_args(subparser):
    me = subparser.add_argument_group("BO configuration")
    m = get_adder(me)
    m('--use-prev-results', action='store_true', default=False) # Possible to load previous results
    m('--bo-source-type', choices=['BO', 'RS'], default='RS')
    m('--bo-target-type', choices=['distBO', 'allBO', 'manualBO',
                                   'noneBO', 'RS', 'multiBO',
                                   'initBO'], default='distBO')
    m('--acq-one-shot', choices=['lcb', 'ei', 'ucb', 'es'], default='lcb')
    m('--acq-source', choices=['ei', 'ucb', 'es'], default='ei')
    m('--acq-target', choices=['ei', 'ucb', 'es'], default='ei')
    m('--source-rand-iterations', type=int, default=30)
    m('--source-iterations', type=int, default=0)
    m('--target-rand-iterations', type=int, default=0)
    m('--target-iterations', type=int, default=30)
    m('--warm-start-tasks', type=int, default=3)
    m('--n-warm-per-task', type=int, default=3)
    m('--warm-start-method', choices=['dist', 'manual'], default='manual')
    m('--data-seed', type=int, default=np.random.randint(2**32))
    m('--opt-seed', type=int, default=np.random.randint(2**23))
    
    kernel = subparser.add_argument_group('Kernel')
    r = get_adder(kernel)
    r('--embed-op', choices=['none', 'joint', 'concat', 'nn'], default='joint',
                      help='none uses P_X\
                            nn for neural net phi([x,y])\
                            joint for P_XY with phi(x), phi(y)\
                            concat for P_X P_Y|X with phi(x), phi(y)\
                            classification uses delta-kernel for y automatically,\
                            hence it is same as using joint or concat.')
    r('--concat-data-size', action='store_true', default=False)
    r('--max-embed-size', type=int, default=20000, help='Max number of data for embedding (Testing time)')
    
    optim_acq = subparser.add_argument_group("optimisation parameters")
    o = get_adder(optim_acq)
    o('--xi', type=float, default=0.01, help='control the level of exploration')
    o('--kappa', type=float, default=2.58, help='control the level of few shot LCB sd')
    o('--algorithm', choices=['GP', 'BLR'], default='GP')
    o('--weight_dim_1', type=int, default=50, help='Number of NN units for layer 1 if BLR deep feature')
    o('--weight_dim_2', type=int, default=50, help='Number of NN units for layer 2 if BLR deep feature')
    o('--weight_dim_3', type=int, default=50, help='Number of NN units for layer 3 if BLR deep feature')
    o('--num-of-rff', type=int, default=0, help='Number of Random Fourier Features for BLR, \
                                                   if 0 then dont use RFF')
    o('--weight_dim_1_x', type=int, default=20, help='Number of NN units for layer 1 for covariates')
    o('--weight_dim_2_x', type=int, default=10, help='Number of NN units for layer 2 for covariates')
    o('--weight_dim_1_y', type=int, default=20, help='Number of NN units for layer 1 for labels')
    o('--weight_dim_2_y', type=int, default=10, help='Number of NN units for layer 2 for labels')
    o('--weight_dim_1_xy', type=int, default=20, help='Number of NN units for layer 1 for covariates, labels')
    o('--weight_dim_2_xy', type=int, default=20, help='Number of NN units for layer 2 for covariates, labels')
    o('--weight_dim_3_xy', type=int, default=0, help='Number of NN units for layer 3 for covariates, labels')
    o('--gradient-clip', action='store_true', default=False)
    o('--lkh_var_init', type=float, default=0.00001)
    o('--dont-stratify', action='store_true', default=False)
    o('--n-warmup', type=int, default=300000, help='Number of random iterations for locate acq max')
    o('--n-opt-iterations', type=int, default=10, help='Number of repeats for optimisation of acq')
    o('--n-epochs', type=int, default=5000, help='Number of training epochs for Marginal likelihood')
    o('--learning-rate', type=float, default=0.005, help='Learning rate for Marginal likelihood optimisation')
    o('--first_early_stop_epoch', type=int, default=1000, help='First early stop epoch')
    o('--batch-size', type=int, default=1000, help='Batch size to compute embedding (Training time)')
    o('--float64', action='store_true', default=False)
    io = subparser.add_argument_group("I/O parameters")
    i = get_adder(io)
    io.add_argument('out_dir')
    i('--n-cpus', type=int, default=min(1, mp.cpu_count()))

def make_parser(rest_of_args=_add_args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="The dataset to run on")
    # Subparser chosen by the first argument of your parser
    def add_subparser(name, **kwargs):
        subparser = subparsers.add_parser(name, **kwargs)
        subparser.set_defaults(dataset=name)
        data = subparser.add_argument_group('Data parameters')
        rest_of_args(subparser)
        return data, get_adder(data)

    def add_sim_args(g): # For Simulated/Toy Datasets
        a = g.add_argument
        a('--preprocess', choices=['None', 'standardise'], default='standardise')
        a('--n-train', type=int, default=1000)
        a('--n-test',  type=int, default=1000)
        a('--dim', '-d', type=int, default=10)

    def add_split_args(g): # For Real Datasets
        a = g.add_argument
        a('--preprocess', choices=['None', 'standardise'], default='standardise')
        a('--test-size', type=float, default=.4,
          help="Number or portion of overall data to use for testing "
               "(default %(default)s).")
        
    ### TOY DATASETS ###
    # One dimensional toy dataset 
    one_d_split, d = add_subparser('one_dim_split')
    d('--fake_source_num', type=int, default=12)
    d('--true_source_num', type=int, default=3)
    d('--mu_sd', type=float, default=0.5)
    add_sim_args(one_d_split)

    # Counter example to manual meta-features
    counter_manual, d = add_subparser('counter_manual')
    d('--noise', type=int, default=0.5) # Noise level
    add_sim_args(counter_manual)

    # bw toy dataset
    bw_gamma_vary, d = add_subparser('bw_dim_vary')
    d('--source-num', type=int, default=10)
    d('--target-group', type=int, default=-1, help='Use -1 to set it to an unseen target')
    d('--prob-type', choices=['classification', 'regression'], default='classification')
    add_sim_args(bw_gamma_vary)

    ### REAL DATASETS ###
    # Parkinson regression
    patient, d = add_subparser('patient_vary') 
    d('--target-group', type=int, default=36)
    d('--park-labels', choices=['motor', 'total', 'both'], default='total')
    add_split_args(patient)

    # Protien dataset (Default is forest, change to Jaccard SVM in distBO/data/generate_data.py)
    protein, d = add_subparser('protein_vary')  
    d('--target-group', type=int, default=0) 
    add_split_args(protein)

    return parser

def check_output_dir(dirname, parser):
    if os.path.exists(dirname):
        files = set(os.listdir(dirname)) - {'output'}
        if files:
            parser.error(("Output directory {} exists. Change the name or delete it.")
                         .format(dirname))
    else:
        os.makedirs(dirname)

def parse_args(rest_of_args=_add_args):
    parser = make_parser(rest_of_args)
    args = parser.parse_args()
    check_output_dir(args.out_dir, parser)
    return args

def main():
    ### COMMAND LINE ARGUMENTS ###
    args = parse_args()
    d = {'args': args}
    
    ### DATA GENERATION ###
    print("Loading target and source dataset: {} ...".format(args.dataset))
    target_data, source_datas, supp = generate_data(args)
    d['supp'] = supp # addtional info for reference
    if args.dataset == 'protein_vary':
        args.include_corr_meta = False
    else:
        args.include_corr_meta = True
    ### SETUP ###
    opt_rs = check_random_state(args.opt_seed) # Optimisation seed
    source_elapsed = None # Some methods no source time.

    # Some methods do not require random starts.
    if args.bo_target_type not in ['noneBO', 'RS', 'multiBO']:
        args.target_rand_iterations = 0
    
    bo_config = {'model': args.model, 'rs': opt_rs,
                 'xi': args.xi, 'kappa': args.kappa}

    # Weights setup for GP and BLR
    if args.weight_dim_1 != 0 and args.weight_dim_2 != 0:
        weight_dim = [args.weight_dim_1, args.weight_dim_2]
        if args.weight_dim_3 != 0:
            weight_dim = [args.weight_dim_1, args.weight_dim_2, args.weight_dim_3]
    else:
        weight_dim = []

    if args.weight_dim_3_xy != 0:
        weight_dim_xy = [args.weight_dim_1_xy, args.weight_dim_2_xy, args.weight_dim_3_xy]
    else:
        weight_dim_xy = [args.weight_dim_1_xy, args.weight_dim_2_xy]

    weight_dim_x = [args.weight_dim_1_x, args.weight_dim_2_x]
    weight_dim_y = [args.weight_dim_1_y, args.weight_dim_2_y]

    # Normal Bayesian optimisation do not require so much iterations, avoid numerical issues
    if args.bo_target_type == ['noneBO', 'initBO']:
        args.first_early_stop_epoch = 1000
    optim_config = {'weight_dim': weight_dim, 'weight_dim_x': weight_dim_x, 
                    'weight_dim_y': weight_dim_y, 'weight_dim_xy': weight_dim_xy,
                    'n_warmup': args.n_warmup, 'n_opt_iterations': args.n_opt_iterations,
                    'n_epochs': args.n_epochs, 'lkh_var_init': args.lkh_var_init, 'lr': args.learning_rate,
                    'float64': args.float64,  'n_cpus':args.n_cpus, 'stratify': not args.dont_stratify,
                    'embed_op': args.embed_op, 'concat_data_size': args.concat_data_size, 
                    'algorithm': args.algorithm, 'gradient_clip': args.gradient_clip,
                    'num_of_rff': args.num_of_rff, 'batch_size': args.batch_size,
                    'first_early_stop_epoch': args.first_early_stop_epoch, 'max_size': None}

    source_optim_config = deepcopy(optim_config)

    ### Source Hyperparameter Search ###
    # For those that uses source datasets
    if args.bo_target_type in ['distBO', 'allBO', 'multiBO',
                               'manualBO', 'initBO']:
        if args.concat_data_size: # Calculate the max dataset size.
            optim_config['max_size'] = max_size = np.max([target_data.train_len()] + \
                                                         [source.train_len() \
                                                         for source in source_datas])
            print('Max train size:', max_size)
        else:
            max_size = None
        # Random search sets BO iterations to 0
        if args.bo_source_type == 'RS': # Random search sets BO iterations to 0
            args.source_iterations = 0
        print("Hyperparameter search for source with {}".format(args.bo_source_type))
        bo_config_para = {'model': args.model, 'rs': opt_rs,
                          'xi': 0.01, 'kappa': args.kappa} # always use xi=0.01 for source 
        source_bo_config = {'acq': args.acq_source,
                            'bo_it': args.source_iterations,
                            'rand_it': args.source_rand_iterations}
        source_optim_config['algorithm'] = 'GP' # Always use GP for source.
        # noneBO does not require so much early stop, avoid numerical issues.
        source_optim_config['first_early_stop_epoch'] = 1000
        source_bo_config = merge_two_dicts(bo_config_para, source_bo_config)
        source_bo_config['optim_config'] = source_optim_config
        source_evals = []
        source_start = time.time()
        for s_data in source_datas:
            if args.use_prev_results:
                path = os.path.dirname(os.path.abspath(__file__))
                evals = prev_extractor(args.dataset, s_data, path, args.bo_source_type,
                                       args.source_rand_iterations, args.source_iterations,
                                       args.data_seed, args.opt_seed)
            else:
                evals = bayes_opt_wrap('params', s_data, include_init=True, **source_bo_config)
            source_evals.append(evals) # last column is evaluations, other columns are hyper-parameters
        source_elapsed = (time.time() - source_start)
        d['source_params'] = [ source_p[:,:-1] for source_p in source_evals]
        d['source_evals'] = [ source_p[:,-1] for source_p in source_evals]
        d['source_time'] = source_elapsed

    ### TARGET HYPERPARAMETER SEARCH ###
    print("Hyperparameter search for target with {}".format(args.bo_target_type))
    target_start = time.time()
    target_bo_config = {'acq': args.acq_target,
                        'bo_it': args.target_iterations,
                        'rand_it': args.target_rand_iterations}
    # Make sure that RS, noneBO use the same start seed
    bo_config['rs'] = check_random_state(args.opt_seed + 1)
    target_bo_config = merge_two_dicts(bo_config, target_bo_config)
    target_bo_config['optim_config'] = optim_config

    # include_init = False refers to not including source tasks evaluations
    if args.bo_target_type == 'distBO':
        target_summary, target_sim = bayes_opt_wrap('dist', target_data, init_list=source_evals,
                                                    include_init=False,
                                                    source_data=source_datas,
                                                    acq_one_shot=args.acq_one_shot,
                                                    **target_bo_config)
        d['target_sim'] = target_sim
    elif args.bo_target_type == 'manualBO':
        _ = meta(source_datas + [target_data], args.model, 
                 max_size=max_size, random_state=opt_rs, include_corr_meta=args.include_corr_meta)
        target_summary, target_sim = bayes_opt_wrap('manual', target_data, init_list=source_evals, 
                                                    include_init=False,
                                                    source_data=source_datas,
                                                    acq_one_shot=args.acq_one_shot,
                                                    **target_bo_config)
        d['target_sim'] = target_sim
    elif args.bo_target_type == 'initBO':
        args.target_rand_iterations = 0
        assert args.warm_start_tasks <= len(source_datas), 'Warm start tasks should not be \
                                                            greater than the number of source \
                                                            data.'
        if args.warm_start_method == 'manual':
            # Compute manual meta-features
            _ = meta(source_datas + [target_data], args.model,
                     max_size=max_size, random_state=opt_rs, include_corr_meta=args.include_corr_meta)
        target_summary = bayes_opt_wrap('params', target_data, init_list=source_evals,
                                        include_init=False,
                                        warm_start_tasks=args.warm_start_tasks,
                                        n_warm_per_task=args.n_warm_per_task,
                                        warm_start_method=args.warm_start_method,
                                        source_data=source_datas,
                                        **target_bo_config)
    elif args.bo_target_type == 'allBO':
        target_summary = bayes_opt_wrap('params', target_data, init_list=source_evals,
                                        source_data=source_datas,
                                        include_init=False,
                                        acq_one_shot=args.acq_one_shot,
                                        **target_bo_config)
    elif args.bo_target_type == 'multiBO':
        target_summary, target_sim = bayes_opt_wrap('multi', target_data, init_list=source_evals,
                                                    source_data=source_datas,
                                                    include_init=False, **target_bo_config)
        d['target_sim'] = target_sim
    elif args.bo_target_type == 'noneBO':
        target_summary = bayes_opt_wrap('params', target_data, include_init=True,
                                        **target_bo_config)
    elif args.bo_target_type == 'RS':
        args.target_iterations = 0
        target_bo_config['bo_it'] = 0 #Random Search does not perform BO
        target_summary = bayes_opt_wrap('params', target_data, include_init=True,
                                        **target_bo_config)
    target_elapsed = (time.time() - target_start)
    
    if args.bo_target_type == 'initBO':
        total_it = args.warm_start_tasks * args.n_warm_per_task + args.target_iterations
    else:
        total_it = args.target_rand_iterations + args.target_iterations
    
    d['target_params'] = target_summary[-total_it:,:-1]
    d['target_evals'] = target_summary[-total_it:,-1]
    d['target_time'] = target_elapsed
    
    ### SAVE RESULTS ###
    np.savez(os.path.join(args.out_dir, 'results.npz'), **d)

if __name__ == '__main__':
    main()