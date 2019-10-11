from __future__ import print_function, division
from copy import deepcopy 

import numpy as np
import pandas as pd

from distBO.bayes_opt import BayesianOptimization
from distBO.model_embed.models import set_model
from distBO.utils import similarity_l2

def bayes_opt_wrap(method, target_data, acq, model, rs, optim_config, xi=0, kappa=1.0, bo_it=100, warm_start_tasks=0, 
                   n_warm_per_task=1, warm_start_method='manual', rand_it=10, init_list=None, max_size=None,
                   acq_one_shot='ei', include_init=False, source_data=None):
    if init_list is not None:
        size_evals = [len(init) for init in init_list]
        print('Previous size evals', size_evals)
        init_list_stack = np.vstack(init_list)
    else:
        init_list_stack = None
    target, param_bounds, previous_evals, model_type = set_model(model, deepcopy(target_data), init_list_stack)
    if source_data is not None:
        bo = BayesianOptimization(target, param_bounds, source_data=source_data, target_data=target_data, 
                                  size_evals=deepcopy(size_evals), include_init=include_init, model_type=model_type,
                                  random_state=rs)
    else:
        bo = BayesianOptimization(target, param_bounds, include_init=include_init,
                                  model_type=model_type, random_state=rs)
    
    if warm_start_tasks != 0 and source_data is not None:
        # Currently only for normal hyper-parameter optimisation
        assert n_warm_per_task > 0, 'n_warm_per_task must be greater than 0'
        # Only care about similarity
        if optim_config['algorithm'] in ['GP', 'BLR']:
            # Let's use L2 distance with manual meta-features
            similarity = np.array(similarity_l2(source_data, target_data))
        top_sim = similarity.argsort()[-warm_start_tasks:][::-1]
        cumsum_size = [0] + np.cumsum(size_evals).tolist()
        init_loc = []
        for sim in top_sim:
            assert n_warm_per_task < len(init_list[sim]), 'n_warm_per_task exceeds source evaluations'
            loc = cumsum_size[sim] + init_list[sim][:,-1].argsort()[-n_warm_per_task:][::-1]
            init_loc = init_loc + loc.tolist()
        del previous_evals['target'] # pointer remove
        for key, value in previous_evals.iteritems():
            previous_evals[key] = value[init_loc]
        bo = BayesianOptimization(target, param_bounds,
                                  model_type=model_type, random_state=rs)
        bo.explore(previous_evals) # A list of hyperparameters to explore, included as part of target evaluations.
        init_list = None # Warm-start only.
    if init_list is not None:
        bo.initialize(previous_evals) # Source task hyperparamters and evaluations.
    bo.maximize(method, optim_config, init_points=rand_it,
                n_iter=bo_it, acq=acq, xi=xi, kappa=kappa,
                acq_one_shot=acq_one_shot) 
    df = pd.DataFrame(bo.res['all']['params'])
    df = df.sort_index(axis=1)
    params = df.values
    params_eval = np.expand_dims(bo.res['all']['values'], 1)
    if method in ['dist', 'manual', 'multi']:
        similarity = np.array(bo.res['all']['similarity'])
        return np.hstack((params, params_eval)), similarity
    else:
        return np.hstack((params, params_eval))
