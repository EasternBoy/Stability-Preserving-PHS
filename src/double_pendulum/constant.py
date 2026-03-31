import os

# Common parameters
nncells = 64
dt      = 0.05
nx      = 4

batch_size     = 32
learning_rate  = 10e-3
l2_lambda      = 1e-7
equi_lambda = 1e-2  

scheduler_step = 1000
num_epochs     = 5000
nndepth   = 3
patience  = 500
data_path = os.path.join("data", "double_pendulum", f"double_pendulum_train_dt{dt}.npz")

grid_size = 10
n_traj    = grid_size**2*200
layers_type = ["gelu", "gelu", "gelu"]