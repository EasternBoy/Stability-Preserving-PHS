import os
from datetime import datetime

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import constant as cs

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

def activation_select(act_type):
    if act_type == "gelu":
        return nn.gelu
    if act_type == "relu":
        return nn.relu
    if act_type == "softplus":
        return nn.softplus

act_func = [activation_select(cs.layers_type[i]) for i in range(cs.nndepth)]


class NonNegDense(nn.Module):
    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        kernel = self.param(
            'kernel',
            lambda key, shape, dtype=jnp.float64: -4.0
            + 1e-2 * jax.random.normal(key, shape, dtype=dtype),
            (in_features, self.features),
        )
        kernel_pos = nn.softplus(kernel)
        y = jnp.dot(x, kernel_pos)

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            y = y + bias
        return y
        
class ICNNHNet(nn.Module):
    nncells: int
    depth:   int = 3

    @nn.compact
    def __call__(self, x):
        z = nn.softplus(nn.Dense(features=self.nncells, name="x_to_z_0")(x))
        for i in range(self.depth - 1):
            z_from_x = nn.Dense(features=self.nncells, name=f"x_to_z_{i + 1}")(x)
            z_from_z = NonNegDense(features=self.nncells, use_bias=False, name=f"z_to_z_{i + 1}")(z)
            z        = nn.softplus(z_from_x + z_from_z)

        h_from_x = nn.Dense(features=1, name="x_to_h")(x)
        h_from_z = NonNegDense(features=1, use_bias=True, name="z_to_h")(z)
        return (h_from_x + h_from_z).squeeze()
    
class PHNN(nn.Module):
    n: int
    nncells: int

    def setup(self):
        self.H_net = ICNNHNet(nncells=self.nncells)

    @nn.compact
    def __call__(self, x):
        n_half = self.n // 2
        J = jnp.block([
            [jnp.zeros((n_half, n_half)), jnp.eye(n_half)],
            [-jnp.eye(n_half), jnp.zeros((n_half, n_half))],
        ])
        gamma_val = self.param("gamma", nn.initializers.ones, (n_half,))
        gamma_val = jnp.array([0.5, 0.5])


        R = jnp.block([
            [jnp.zeros((n_half, n_half)), jnp.zeros((n_half, n_half))],
            [jnp.zeros((n_half, n_half)), jnp.diag(gamma_val)],
        ])

        def compute_raw_hamiltonian(x_):
            return self.H_net(x_)

        x0 = jnp.zeros_like(x)
        raw_h_x = compute_raw_hamiltonian(x)
        
        grad_raw_h_x = jax.grad(compute_raw_hamiltonian)(x)
        grad_raw_h_0 = jax.grad(compute_raw_hamiltonian)(x0)

        dHdx = grad_raw_h_x - grad_raw_h_0
        h_x = raw_h_x - jnp.dot(grad_raw_h_0, x)
        dxdt = (J - R) @ dHdx 

        return dxdt, h_x

def train_model():
    train_key = key
    data_path = cs.data_path
    print(f"Loading data from {data_path}...")
    data_raw = np.load(data_path)
    
    xk_flat = jnp.array(data_raw["Xk"].T, dtype=jnp.float64)
    xkp1_flat = jnp.array(data_raw["Xkp1"].T, dtype=jnp.float64)
    nx = xk_flat.shape[1]

    num_total_traj = getattr(cs, 'num_traj', cs.n_traj) 
    seq_len = xk_flat.shape[0] // num_total_traj

    print(f"Reshaping data into {num_total_traj} trajectories of length {seq_len}...")

    xk_traj = xk_flat.reshape(num_total_traj, seq_len, nx)
    xkp1_traj = xkp1_flat.reshape(num_total_traj, seq_len, nx)

    x0_full = xk_traj[:, 0, :]               # Shape: (num_traj, 4)
    x_target_full = xkp1_traj[:, :, :]       # Shape: (num_traj, seq_len, 4)

    val_ratio = 0.1
    val_size = int(num_total_traj * val_ratio)
    train_size = num_total_traj - val_size

    shuffled_indices = jax.random.permutation(train_key, num_total_traj)
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_dataset = (
        x0_full[train_indices],
        x_target_full[train_indices],
    )
    val_dataset = (
        x0_full[val_indices],
        x_target_full[val_indices],
    )

    mor_ph = PHNN(n=nx, nncells=cs.nncells)
    init_key, _ = jax.random.split(train_key, 2)
    params = mor_ph.init(init_key, jnp.ones((nx,)))["params"]

    patience = 500
    
    scheduler_step = 500
    lr_schedule = optax.linear_schedule(
        init_value = 1e-2,
        end_value  = 2e-4,
        transition_steps=scheduler_step,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), 
        optax.adamw(learning_rate=lr_schedule)
    )
    opt_state = optimizer.init(params)


    def euler_rollout(params_, x0_single, length):
        def step_fn(x, _):
            def get_dx(state):
                return mor_ph.apply({"params": params_}, state)[0]
            
            k1 = get_dx(x)
            x_next = x + cs.dt*k1
            return x_next, x_next 
        
        _, trajectory = jax.lax.scan(step_fn, x0_single, None, length=length)
        return trajectory

    vmapped_rollout = jax.vmap(euler_rollout, in_axes=(None, 0, None))

    def mse_trajectory_loss(params_, x0_batch, x_target_batch):
        seq_length = x_target_batch.shape[1]
        x_pred_traj = vmapped_rollout(params_, x0_batch, seq_length)
        return jnp.mean((x_pred_traj[:, :, :2] - x_target_batch[:, :, :2]) ** 2)

    def l2_regularization(params_):
        return sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(params_))

    def train_loss(params_, x0_batch, x_target_batch):
        data_loss = mse_trajectory_loss(params_, x0_batch, x_target_batch)
        return data_loss + cs.l2_lambda * l2_regularization(params_)

    @jax.jit
    def train_step(params_, opt_state_, x0_batch, x_target_batch):
        loss, grads = jax.value_and_grad(train_loss)(params_, x0_batch, x_target_batch)
        updates, opt_state_ = optimizer.update(grads, opt_state_, params_)
        params_ = optax.apply_updates(params_, updates)
        return params_, opt_state_, loss

    @jax.jit
    def eval_step(params_, x0_batch, x_target_batch):
        return mse_trajectory_loss(params_, x0_batch, x_target_batch)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting Full-Trajectory RK4 PHNN training...\n")

    for epoch in range(1, cs.num_epochs + 1):
        total_train_loss = 0.0
        num_train_batches = int(np.ceil(train_size / cs.batch_size))

        perm_key, train_key = jax.random.split(train_key)
        train_perm = jax.random.permutation(perm_key, train_size)

        for i in range(num_train_batches):
            batch_indices = train_perm[i * cs.batch_size:(i + 1) * cs.batch_size]
            x0_b = train_dataset[0][batch_indices]
            x_target_b = train_dataset[1][batch_indices]

            params, opt_state, loss = train_step(params, opt_state, x0_b, x_target_b)
            total_train_loss += loss

        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        total_val_loss = 0.0
        num_val_batches = int(np.ceil(val_size / cs.batch_size))
        for i in range(num_val_batches):
            batch_indices = slice(i * cs.batch_size, (i + 1) * cs.batch_size)
            x0_v = val_dataset[0][batch_indices]
            x_target_v = val_dataset[1][batch_indices]
            
            val_loss = eval_step(params, x0_v, x_target_v)
            total_val_loss += val_loss

        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        if avg_val_loss < 0.99 * best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (patience {patience} reached)\n")
            break

        if epoch % 100 == 0:
            current_step = epoch 
            current_lr = lr_schedule(current_step)
            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6e}, "
                f"Val Loss = {avg_val_loss:.6e}, LR = {current_lr:.6e}"
            )

    print("\nTraining completed!\n")

    save_dir = "model/double_pendulum"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    msgpack_path = os.path.join(save_dir, f"ICNN_DPen_FullTraj_dt{cs.dt}.msgpack")
    with open(msgpack_path, "wb") as f:
        f.write(serialization.to_bytes(params))
    print(f"Saved trained model MSGPACK to: {msgpack_path}")

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Full Trajectory MSE Loss")
    plt.title("RK4 Full Trajectory Rollout Loss (Double Pendulum)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    train_model()