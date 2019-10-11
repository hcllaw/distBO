# Adapted from https://github.com/fmfn/BayesianOptimization
from __future__ import print_function, division
from copy import deepcopy
import warnings

import numpy as np
from sklearn.utils import check_random_state

from .helpers import (UtilityFunction, PrintLog, acq_max, ensure_rng)
from .target_space import TargetSpace
from distBO.train.gp_regressor import gp_regressor
from distBO.utils import get_median_sqdist

class BayesianOptimization(object):
    def __init__(self, f, pbounds, target_data=None, source_data=None, size_evals=None,
                 model_type=None, include_init=False, random_state=None, verbose=1):
        """
        :param f:
            Function to be maximized.

        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        """
        # Store the source and target data if provided
        if source_data is not None and source_data[0].embed is not None:
            self.source_embed = [s_x.embed for s_x in source_data]
        else:
            self.source_embed = [None]

        if target_data is not None and target_data.embed is not None:
            self.target_embed = np.expand_dims(target_data.embed, 0)
        else:
            self.target_embed = None

        if source_data is not None:
            self.source_X = [s_x.train_x for s_x in source_data]
            self.source_Y = [s_y.train_y for s_y in source_data]
            self.source_embed_ind = [s_ind.embed_ind for s_ind in source_data]
        else:
            self.source_X = [None]
            self.source_Y = [None]
            self.source_embed_ind = [None]
        
        if target_data is not None:
            self.target_X = target_data.train_x
            self.target_Y = target_data.train_y
            self.target_embed_ind = target_data.embed_ind
        else:
            self.target_X = None
            self.target_Y = None
            self.target_embed_ind = None

        if size_evals is not None:
            self.size_evals = size_evals
        else:
            self.size_evals = [0]
        
        # Store the original dictionary
        self.model_type = model_type
        self.pbounds = pbounds
        self.para_dim = len(pbounds)
        self.include_init = include_init
        self.random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self.space = TargetSpace(f, pbounds, random_state)

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Current Target data x and ys
        self.current_x = []
        self.current_y = []

        # Counter of iterations
        self.i = 0

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.space.keys)

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [] , 'params': [], 'similarity': []}

        # Verbose
        self.verbose = verbose

    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """
        # Concatenate new random points to possible existing
        # points from self.explore method.
        rand_points = self.space.random_points(init_points)
        self.init_points.extend(rand_points)

        # Add the points from `self.initialize` to the observations
        if self.x_init: # Source task evaluations 
            x_init = np.vstack(self.x_init)
            y_init = np.hstack(self.y_init)
            for x, y in zip(x_init, y_init):
                self.space.add_observation(x, y)
                if self.include_init: # Set to be False for target task 
                    self.res['all']['values'].append(y)
                    self.res['all']['params'].append(dict(zip(self.space.keys, x)))
                if self.verbose:
                    self.plog.print_step(x, y)

        # Evaluate target function at all initialization points
        # Any previous points from warm start or random search.
        for x in self.init_points: # Any previous points from warm start or random search.
            y = self._observe_point(x)
            # Add to current target y!
            self.current_y.append(y)
            #if self.include_init:
            self.res['all']['values'].append(y)
            self.res['all']['params'].append(dict(zip(self.space.keys, x)))
        # Updates the flag
        self.initialized = True

    def _observe_point(self, x):
        y = self.space.observe_point(x)
        if self.verbose:
            self.plog.print_step(x, y)
        return y

    def explore(self, points_dict, eager=False): 
        """Method to explore user defined points.

        :param points_dict:
        :param eager: if True, these points are evaulated immediately
        """
        if eager:
            self.plog.reset_timer()
            if self.verbose:
                self.plog.print_header(initialization=True)

            points = self.space._dict_to_points(points_dict)
            for x in points:
                self._observe_point(x)
        else:
            points = self.space._dict_to_points(points_dict)
            self.init_points = points

    def initialize(self, points_dict):
        """
        Method to introduce points for which the target function value is known

        :param points_dict:
            dictionary with self.keys and 'target' as keys, and list of
            corresponding values as values.

        ex:
            {
                'target': [-1166.19102, -1142.71370, -1138.68293],
                'alpha': [7.0034, 6.6186, 6.0798],
                'colsample_bytree': [0.6849, 0.7314, 0.9540],
                'gamma': [8.3673, 3.5455, 2.3281],
            }

        :return:
        """

        self.y_init.extend(points_dict['target'])
        for i in range(len(points_dict['target'])):
            all_points = []
            for key in self.space.keys:
                all_points.append(points_dict[key][i])
            self.x_init.append(all_points)

    def initialize_df(self, points_df):
        """
        Method to introduce point for which the target function
        value is known from pandas dataframe file

        :param points_df:
            pandas dataframe with columns (target, {list of columns matching
            self.keys})

        ex:
              target        alpha      colsample_bytree        gamma
        -1166.19102       7.0034                0.6849       8.3673
        -1142.71370       6.6186                0.7314       3.5455
        -1138.68293       6.0798                0.9540       2.3281
        -1146.65974       2.4566                0.9290       0.3456
        -1160.32854       1.9821                0.5298       8.7863

        :return:
        """

        for i in points_df.index:
            self.y_init.append(points_df.loc[i, 'target'])

            all_points = []
            for key in self.space.keys:
                all_points.append(points_df.loc[i, key])

            self.x_init.append(all_points)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds:
            A dictionary with the parameter name and its new bounds

        """
        # Update the internal object stored dict
        self.pbounds.update(new_bounds)
        self.space.set_bounds(new_bounds)

    def maximize(self, kernel, optim_config, warm_start=False,
                 init_points=5, n_iter=25, acq='ucb', kappa=1.0,
                 xi=0.0, acq_one_shot='ei'):
        # Reset timer
        self.plog.reset_timer()

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)
        
        # For any warm-start vs random search points
        if len(self.current_y) != 0:
            y_max = np.max(self.current_y)
            print('Current y_max:', y_max)
        else:
            y_max = 0.0
        
        gp_dict = {'target_X': self.target_X, 'target_Y': self.target_Y,
                   'n_cpus': optim_config['n_cpus'], 'model_type': self.model_type,
                   'target_embed_ind': self.target_embed_ind,
                   'float64': optim_config['float64']}

        max_lkd_dict = {'data_X': self.source_X, 'data_Y': self.source_Y, 
                        'source_embed_ind': self.source_embed_ind,
                        'data_embed': self.source_embed, 
                        'target_embed': self.target_embed,
                        'weight_dim': optim_config['weight_dim'],
                        'weight_dim_x': optim_config['weight_dim_x'],
                        'weight_dim_y': optim_config['weight_dim_y'],
                        'weight_dim_xy': optim_config['weight_dim_xy'],
                        'sizes': self.size_evals, 'stratify': optim_config['stratify'],
                        'learning_rate': optim_config['lr'],
                        'batch_size': optim_config['batch_size'],
                        'num_of_rff': optim_config['num_of_rff'],
                        'algorithm': optim_config['algorithm'],
                        'embed_op': optim_config['embed_op'],
                        'max_epochs': optim_config['n_epochs'],
                        'concat_data_size': optim_config['concat_data_size'],
                        'max_size': optim_config['max_size'],
                        'likhd_var_init': optim_config['lkh_var_init'],
                        'gradient_clip': optim_config['gradient_clip'],
                        'first_early_stop_epoch': optim_config['first_early_stop_epoch']}
        
        # Note Ordering is different in self.space.X! Keep it consistent by using self.space.X
        print('Using one shot acquisition function: {} with kapp: {} xi: {}'.format(acq_one_shot, kappa, xi))
        self.util_one_shot = UtilityFunction(kind=acq_one_shot, kappa=kappa, xi=xi)
        
        if kernel == 'multi':
            self.size_evals.append(init_points) # Multi-task has some intialisations already
        
        if n_iter != 0: 
            seed = self.random_state.randint(2**32)
            self.gp = gp_regressor(kernel, seed=seed, **gp_dict)
            best_loss = self.gp.maximise_likhd(self.space.X, np.expand_dims(self.space.Y,1), **max_lkd_dict)
            # Obtain previous evaluations and any random + warm-start evaluations
            init_X = deepcopy(self.space.X)
            init_Y = deepcopy(self.space.Y)
            # Finding argmax of the acquisition function.
            self.gp.initialise_predict()
            x_max = acq_max(ac=self.util_one_shot.utility,
                            algorithm=optim_config['algorithm'],
                            kernel=kernel,
                            gp=self.gp,
                            prev_X=init_X,
                            prev_Y=np.squeeze(init_Y),
                            size_evals=self.size_evals,
                            y_max=y_max,
                            bounds=self.space.bounds,
                            random_state=self.random_state,
                            n_warmup=optim_config['n_warmup'],
                            n_iter=optim_config['n_opt_iterations'])
        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        
        # Add embeddings and update size of new points.
        if kernel == 'manual':
            self.source_embed.append(np.squeeze(self.target_embed))
            self.size_evals.append(1)
        elif kernel == 'dist':
            self.source_X.append(self.target_X)
            self.source_Y.append(self.target_Y)
            self.size_evals.append(1)
            max_lkd_dict['source_embed_ind'] = self.source_embed_ind + [self.target_embed_ind]
        elif kernel == 'multi':
            self.size_evals[-1] = self.size_evals[-1] + 1

        # Print similarities
        if kernel in ['dist', 'manual', 'multi']:
            similarity = np.squeeze(self.gp.similarity())
            self.res['all']['similarity'].append(similarity)
            print('similarity', similarity)

        # Bayesian Optimisation steps
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            while (x_max in self.space.X[int(np.sum(self.size_evals[:-1])):]):
                print('Repeated Hyperparameter, using random search.')
                x_max = self.space.random_points(1)[0]
                pwarning = True

            # Append most recently generated values to X and Y arrays
            y = self.space.observe_point(x_max)
            self.current_y.append(y)

            if self.verbose:
                self.plog.print_step(x_max, y, pwarning)
            
            # Update the best params seen so far
            self.res['all']['values'].append(y)
            self.res['all']['params'].append(dict(zip(self.space.keys, x_max)))

            # End here once reach number of iterations.
            if i == n_iter - 1:
                break

            # Maximise marginal likelihood with new things.
            loss = self.gp.maximise_likhd(self.space.X, np.expand_dims(self.space.Y,1), **max_lkd_dict)
            # Use less first early stop epoch after first two iterations, as roughly already converge.
            max_lkd_dict['first_early_stop_epoch'] = 600

            # Update maximum value to search for next probe point.
            print('Current Y', self.current_y)
            y_max = np.max(self.current_y)
            print('Current maximum:', y_max)
            
            # Maximize acquisition function to find next probing point
            print('Using {}'.format(acq))
            self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

            self.gp.initialise_predict() # Build predict net
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp, y_max=y_max,
                            bounds=self.space.bounds, kappa=kappa,
                            evaluatedX=self.space.X[int(np.sum(self.size_evals[:-1])):],
                            random_state=self.random_state,
                            n_warmup=optim_config['n_warmup'],
                            n_iter=optim_config['n_opt_iterations'])
            # Similarity
            if kernel in ['dist', 'manual', 'multi']:
                similarity = np.squeeze(self.gp.similarity())
                self.res['all']['similarity'].append(similarity)
                print('similarity', similarity)
                self.size_evals[-1] = self.size_evals[-1] + 1

        if hasattr(self, 'gp'):
            self.gp.close()
    
        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

    def points_to_csv(self, file_name):
        """
        After training all points for which we know target variable
        (both from initialization and optimization) are saved

        :param file_name: name of the file where points will be saved in the csv
            format

        :return: None
        """

        points = np.hstack((self.space.X, np.expand_dims(self.space.Y, axis=1)))
        header = ','.join(self.space.keys + ['target'])
        np.savetxt(file_name, points, header=header, delimiter=',', comments='')

    # --- API compatibility ---

    @property
    def X(self):
        warnings.warn("use self.space.X instead", DeprecationWarning)
        return self.space.X

    @property
    def Y(self):
        warnings.warn("use self.space.Y instead", DeprecationWarning)
        return self.space.Y

    @property
    def keys(self):
        warnings.warn("use self.space.keys instead", DeprecationWarning)
        return self.space.keys

    @property
    def f(self):
        warnings.warn("use self.space.target_func instead", DeprecationWarning)
        return self.space.target_func

    @property
    def bounds(self):
        warnings.warn("use self.space.dim instead", DeprecationWarning)
        return self.space.bounds

    @property
    def dim(self):
        warnings.warn("use self.space.dim instead", DeprecationWarning)
        return self.space.dim