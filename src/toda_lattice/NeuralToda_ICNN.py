import os
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import constant as cs
from datetime import datetime
from flax import serialization

# JAX setup
key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

# ==================================
# Define ICNN-based Hamiltonian model
# ==================================
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
    def _raw_icnn(self, x):
        # First layer depends only on x.
        z = nn.softplus(nn.Dense(features=self.nncells, name="x_to_z_0")(x))

        # Subsequent layers combine x and z, with nonnegative z-weights.
        for i in range(self.depth - 1):
            z_from_x = nn.Dense(features=self.nncells, name=f"x_to_z_{i + 1}")(x)
            z_from_z = NonNegDense(features=self.nncells, use_bias=False, name=f"z_to_z_{i + 1}")(z)
            z        = nn.softplus(z_from_x + z_from_z)

        h_from_x = nn.Dense(features=1, name="x_to_h")(x)
        h_from_z = NonNegDense(features=1, use_bias=True, name="z_to_h")(z)
        return (h_from_x + h_from_z).squeeze()

    @nn.compact
    def __call__(self, x):
        def raw_hamiltonian(x_):
            return self._raw_icnn(x_)

        x0 = jnp.zeros_like(x)
        h_raw_x = raw_hamiltonian(x)
        h_raw_0 = raw_hamiltonian(x0)
        grad_raw_0 = jax.grad(raw_hamiltonian)(x0)

        # Affine centering enforces H(0)=0 and dH/dx|_{x=0}=0 by construction.
        h_centered = h_raw_x - h_raw_0 - jnp.dot(grad_raw_0, x)

        # Keep the Hamiltonian coercive to improve gradient quality at large |x|.
        regularizer = 5e-4 * jnp.sum(x ** 2)
        return (nn.softplus(h_centered) - jnp.log(2.0)) + regularizer


class PHNN(nn.Module):
    n: int
    nncells: int
    depth:   int = 3

    def setup(self):
        self.H_net = ICNNHNet(nncells=self.nncells, depth=self.depth)

    @nn.compact
    def __call__(self, x, u):
        n_half = self.n // 2

        J = jnp.block([
            [jnp.zeros((n_half, n_half)), jnp.eye(n_half)],
            [-jnp.eye(n_half), jnp.zeros((n_half, n_half))],
        ])

        gamma_raw = self.param('gamma_raw', nn.initializers.zeros, ())
        gamma_val = nn.sigmoid(gamma_raw)

        R = jnp.block([
            [jnp.zeros((n_half, n_half)), jnp.zeros((n_half, n_half))],
            [jnp.zeros((n_half, n_half)), gamma_val * jnp.eye(n_half)],
        ])

        e1 = jnp.zeros(n_half).at[0].set(1.0)
        B = jnp.concatenate([jnp.zeros(n_half), e1]).reshape(self.n, 1)

        def compute_h(x_):
            return self.H_net(x_)

        h_x = compute_h(x)
        dHdx = jax.grad(compute_h)(x)

        dxdt = (J - R) @ dHdx + B @ u
        y = (B.T @ dHdx).squeeze()

        return dxdt, y, h_x


def train_model():
    train_key = key

    # ==================================
    # Load training data
    # ==================================
    data_raw = np.load(cs.path_to_data)
    dx_raw = data_raw["Xs"]
    du_raw = data_raw["Us"]

    x_full = jnp.array(dx_raw.T, dtype=jnp.float64)
    du     = jnp.array(du_raw.reshape(-1, 1), dtype=jnp.float64)

    xk_full   = x_full[:-1]
    xkp1_full = x_full[1:]
    uk        = du[:-1]

    dt           = cs.stime
    dt_batch     = jnp.full((xk_full.shape[0], 1), dt, dtype=jnp.float64)

    val_ratio    = 0.2
    dataset_size = len(xk_full)
    val_size     = int(dataset_size * val_ratio)
    train_size   = dataset_size - val_size

    shuffled_indices = jax.random.permutation(train_key, dataset_size)
    train_indices    = shuffled_indices[:train_size]
    val_indices      = shuffled_indices[train_size:]

    train_dataset = (
        xk_full[train_indices],
        uk[train_indices],
        dt_batch[train_indices],
        xkp1_full[train_indices],
    )
    val_dataset = (
        xk_full[val_indices],
        uk[val_indices],
        dt_batch[val_indices],
        xkp1_full[val_indices],
    )

    # ==================================
    # Construct network and optimizer
    # ==================================
    nx         = x_full.shape[1]
    nncells    = cs.nncells
    icnn_depth = cs.nndepth

    mor_ph      = PHNN(n=nx, nncells=nncells, depth=icnn_depth)
    init_key, _ = jax.random.split(train_key, 2)
    params      = mor_ph.init(init_key, jnp.ones((nx,)), jnp.ones((1,)))['params']

    l2_weight_decay = 1e-5
    patience        = 500
    scheduler_step  = 500
    batch_size      = cs.batch_size
    num_epochs      = cs.num_epochs
    lr              = cs.learning_rate


    lr_schedule = optax.linear_schedule(
        init_value = lr,
        end_value  = 2e-3,
        transition_steps=scheduler_step,
    )
    optimizer = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay =l2_weight_decay,
    )
    opt_state = optimizer.init(params)

    vmapped_dxdt = jax.vmap(lambda p, x, u: mor_ph.apply({'params': p}, x, u)[0], in_axes=(None, 0, 0))

    def euler_one_step_projected(params_, x_full_batch, u_batch, dt_local):
        x_dot_pred = vmapped_dxdt(params_, x_full_batch, u_batch)
        return x_full_batch + dt_local * x_dot_pred

    def mse_state_loss(params_, x_full_batch, u_batch, dt_local, x_next_full_batch):
        x_next_pred = euler_one_step_projected(params_, x_full_batch, u_batch, dt_local)
        return jnp.mean((x_next_pred - x_next_full_batch) ** 2)

    @jax.jit
    def train_step(params_, opt_state_, x_full_batch, u_batch, dt_local, x_next_full_batch):
        loss, grads = jax.value_and_grad(mse_state_loss)(
            params_, x_full_batch, u_batch, dt_local, x_next_full_batch
        )
        updates, opt_state_ = optimizer.update(grads, opt_state_, params_)
        params_ = optax.apply_updates(params_, updates)
        return params_, opt_state_, loss

    @jax.jit
    def eval_step(params_, x_full_batch, u_batch, dt_local, x_next_full_batch):
        return mse_state_loss(params_, x_full_batch, u_batch, dt_local, x_next_full_batch)

    train_losses = []
    val_losses   = []
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting ICNN state-based training with projected Euler one-step loss...\n")

    for epoch in range(1, num_epochs + 1):
        total_train_loss = 0.0
        num_train_batches = int(np.ceil(train_size / batch_size))

        perm_key, train_key = jax.random.split(train_key)
        train_perm = jax.random.permutation(perm_key, train_size)

        for i in range(num_train_batches):
            batch_indices = train_perm[i * batch_size:(i + 1) * batch_size]
            x_b      = train_dataset[0][batch_indices]
            u_b      = train_dataset[1][batch_indices]
            dt_b     = train_dataset[2][batch_indices]
            x_next_b = train_dataset[3][batch_indices]

            params, opt_state, loss = train_step(params, opt_state, x_b, u_b, dt_b, x_next_b)
            total_train_loss += loss

        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        total_val_loss = 0.0
        num_val_batches = int(np.ceil(val_size / batch_size))
        for i in range(num_val_batches):
            batch_indices = slice(i * batch_size, (i + 1) * batch_size)
            x_v      = val_dataset[0][batch_indices]
            u_v      = val_dataset[1][batch_indices]
            dt_v     = val_dataset[2][batch_indices]
            x_next_v = val_dataset[3][batch_indices]
            val_loss = eval_step(params, x_v, u_v, dt_v, x_next_v)
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

        if epoch % 500 == 0:
            current_lr = lr_schedule(epoch - 1)
            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                f"Val Loss = {avg_val_loss:.6f}, LR = {current_lr:.6e}"
            )

    print("\nTraining complete!\n")

    save_dir = "model/toda_lattice"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # msgpack_path = os.path.join(save_dir, f"NeuralToda_ICNN_params_{timestamp}.msgpack")
    msgpack_path = os.path.join(save_dir, f"NeuralToda_ICNN_params.msgpack")

    with open(msgpack_path, "wb") as f:
        f.write(serialization.to_bytes(params))

    print(f"Saved trained ICNN model MSGPACK to: {msgpack_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("State MSE Loss")
    plt.title("ICNN Projected One-Step State Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_model()
