import os

# Common parameters
nncells = 32
stime   = 0.1
nx      = 10

batch_size     = 8
learning_rate  = 1e-3
l2_lambda      = 1e-7
grad_h0_lambda = 1e-1
scheduler_step = 2000
num_epochs     = 5000
patience       = 1000
nndepth        = 3


path_to_data = os.path.join("data", "toda_lattice", "TodaLat_data_train.npz")

