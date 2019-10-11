# Experiment Script
from itertools import product
import subprocess

experiment = 'one_dim_split'
file_path = 'output_results_file'

#### Fixed settings ####
n_train = 500
n_test = 500
max_embed_size = 500

n_epochs = 10000

kappa = 2.58
warmup = 300000
acq_it = 0

source_rand_it = 10
source_bo_it = 20

warm_start_tasks = 3
n_warm_per_task = 3

#### General Configuration ####
# Seeding for data and optimisation
seed = range(0, 30)

# Seperate into distBO-GP distBO-BLR rest-GP rest-BLR RS noneBO

# DistBO-GP
weight_dim_x = [(20, 10)]
weight_dim_y = [(0, 0)]
batch_size = [500]

# BLR only
xi = [0.01]
weight_dim = [(50, 50, 50)]
rff_dim = [0]

# BO kernel / method except distBO
bo_type = ['manualBO', 'multiBO',
           'initBO', 'noneBO']

# Loop over distBO-GP
to_loop = (seed, ['GP'], ['distBO'], weight_dim_x, [(0, 0)], 
           [(0, 0, 0)], [(0, 0, 0)], [0], batch_size, xi, ['none'])
grid_distbo_gp = list(product(*to_loop))

# Loop over distBO-BLR
to_loop = (seed, ['BLR'], ['distBO'], weight_dim_x, [(0, 0)],
           [(0, 0, 0)], weight_dim, rff_dim, batch_size, xi, ['none'])
grid_distbo_blr = list(product(*to_loop))

# Loop over all-GP
to_loop = (seed, ['GP'], bo_type, [(0, 0)], [(0, 0)], 
           [(0, 0, 0)], [(0, 0, 0)], [0], [0], xi, ['none'])
grid_all_gp = list(product(*to_loop))

# Loop over all-BLR
to_loop = (seed, ['BLR'], bo_type, [(0, 0)], [(0, 0)],
           [(0, 0, 0)], weight_dim, rff_dim, [0], xi, ['none'])
grid_all_blr = list(product(*to_loop))

# Loop over RS
to_loop = (seed, ['BLR'], ['RS'], [(0, 0)], [(0, 0)], 
           [(0, 0, 0)], [(0, 0, 0)], [0], [0], xi, ['none'])
grid_rs = list(product(*to_loop))

grid_all = grid_distbo_gp + grid_distbo_blr + grid_all_gp + grid_all_blr +  grid_rs

for selected in grid_all:
    seed_c, algorithm_c, bo_type_c, weight_dim_x_c, weight_dim_y_c,  \
    weight_dim_xy_c, weight_dim_c, rff_dim_c, batch_size_c, xi_c, embed_op_c = selected

    #### Iterations for different bo_types ####
    if bo_type_c in ['distBO', 'allBO', 'manualBO']:
        target_it = 15
        rand_it = 0
    elif bo_type_c == 'RS':
        target_it = 0
        rand_it = 15
    elif bo_type_c == 'noneBO':
        rand_it = 9
        target_it = 6
    elif bo_type_c == 'multiBO':
        rand_it = 1
        target_it = 14
    elif bo_type_c == 'initBO':
        rand_it = 0
        target_it = 6

    command = [
        "python", "train_test.py",
         experiment,
        '--n-train', str(n_train),
        '--n-test', str(n_test),
        '--data-seed', str(seed_c),
        '--opt-seed', str(seed_c),
        '--float64',
        '--bo-source-type', 'BO',
        '--source-rand-iterations', str(source_rand_it),
        '--source-iterations', str(source_bo_it),
        '--algorithm', algorithm_c,
        '--bo-target-type', bo_type_c,
        '--target-iterations', str(target_it),
        '--target-rand-iterations', str(rand_it),
        '--kappa', str(kappa),
        '--xi', str(xi_c),
        '--embed-op', embed_op_c,
        '--max-embed-size', str(max_embed_size),
        '--batch-size', str(batch_size_c),
        '--dont-stratify',
        '--n-epochs', str(n_epochs),
        '--learning-rate', str(0.005),
        '--n-warmup', str(warmup),
        '--n-opt-iterations', str(acq_it),
        '--num-of-rff', str(rff_dim_c),
        '--first_early_stop_epoch', str(5000),
        '--warm-start-method', 'manual',
        '--warm-start-tasks', str(warm_start_tasks),
        '--n-warm-per-task', str(n_warm_per_task),
        '--weight_dim_1', str(weight_dim_c[0]),
        '--weight_dim_2', str(weight_dim_c[1]),
        '--weight_dim_3', str(weight_dim_c[2]),
        '--weight_dim_1_x', str(weight_dim_x_c[0]),
        '--weight_dim_2_x', str(weight_dim_x_c[1]),
        '--weight_dim_1_y', str(weight_dim_y_c[0]),
        '--weight_dim_2_y', str(weight_dim_y_c[1]),
        '--weight_dim_1_xy', str(weight_dim_xy_c[0]),
        '--weight_dim_2_xy', str(weight_dim_xy_c[1]),
        '--weight_dim_3_xy', str(weight_dim_xy_c[2]),
        file_path
    ]
    cmd = subprocess.list2cmdline(command)
    print(cmd)



