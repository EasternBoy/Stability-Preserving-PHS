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

# ==================================
# Constants and Helper Functions
# ==================================
# Proposed
sms_del = 1e-4
sms_cm  = 1 #Default 0.2
sms_or  = 1   #Default 2

@jax.jit
def comb_manual(n, k):
    """A JAX-compatible implementation of nCr."""
    k = jnp.minimum(k, n - k)

    def body_fun(i, val):
        return val * (n - i) / (i + 1)

    return jax.lax.fori_loop(0, k, body_fun, 1.0)

def sig(r):
    return jnp.sqrt(r**2 + sms_del**2) - sms_del

@jax.jit
def smstep(sigma, b = sig(sms_cm)):

    def true_fn(s):
        temp = 0.0
        for k in range(sms_or + 1):
            term1 = comb_manual(sms_or + k, k)
            term2 = comb_manual(2 * sms_or + 1, sms_or - k)
            term3 = (-s / b) ** k
            temp += term1 * term2 * term3
        return temp * (s / b) ** (sms_or + 1)

    def false_fn(s):
        return 1.0

    return jax.lax.cond(sigma <= b, true_fn, false_fn, sigma)

@jax.jit
def d_smstep(sigma, b = sig(sms_cm)):

    def true_fn(s):
        temp = 0.0
        for k in range(sms_or + 1):
            term1 = comb_manual(sms_or + k, k)
            term2 = comb_manual(2 * sms_or + 1, sms_or - k)
            term3 = (-1) ** k * (s / b) ** (sms_or + k) * (sms_or + k + 1) / b
            temp += term1 * term2 * term3
        return temp

    def false_fn(s):
        return 0.0

    return jax.lax.cond(sigma <= b, true_fn, false_fn, sigma)


# ==================================
# Define model
# ==================================
class ONet(nn.Module):
    nncells: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=cs.nncells)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=cs.nncells)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=1)(x) #Output
        x = nn.softplus(x)
        return x.squeeze()
    
class INet(nn.Module): #smaller network
    nncells: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=int(cs.nncells/2))(x)
        x = nn.gelu(x)
        x = nn.Dense(features=int(cs.nncells/2))(x)
        x = nn.gelu(x)
        x = nn.Dense(features=1)(x) #Output
        x = nn.softplus(x)
        return x.squeeze()
    

class PHNN(nn.Module):
    n: int
    nncells: int

    def setup(self):
        self.outer_net = ONet(nncells=self.nncells)
        self.inner_net = INet(nncells=self.nncells)

    def inner_only(self, x, u=None):
        return self.inner_net(x)

    @nn.compact
    def __call__(self, x, u):
        n_half = self.n // 2

        J = jnp.block([
            [jnp.zeros((n_half, n_half)), jnp.eye(n_half)],
            [-jnp.eye(n_half), jnp.zeros((n_half, n_half))],
        ])

        gamma_val = self.param('gamma', nn.initializers.ones, ())
        gamma_val = jnp.clip(gamma_val, 0.0, 1.0)


        beta = self.param('beta', nn.initializers.ones, ())
        beta = jnp.clip(beta, 0.0, np.inf)

        R = jnp.block([
            [jnp.zeros((n_half, n_half)), jnp.zeros((n_half, n_half))],
            [jnp.zeros((n_half, n_half)), gamma_val * jnp.eye(n_half)],
        ])

        e1 = jnp.zeros(n_half).at[0].set(1.0)
        B  = jnp.concatenate([jnp.zeros(n_half), e1]).reshape(self.n, 1)

        def compute_outer_nn(x_):
            return self.outer_net(x_)
        def compute_inner_nn(x_):
            return self.inner_net(x_)

        outer_nn   = compute_outer_nn(x)
        inner_nn   = compute_inner_nn(x)

        mag2  = jnp.dot(x, x)
        mag   = jnp.sqrt(mag2)
        s_mag = sig(mag)

        h_x        = outer_nn * smstep(s_mag) + inner_nn * (1 - smstep(s_mag))
        d_outer_nn = jax.grad(compute_outer_nn)(x)
        d_inner_nn = jax.grad(compute_inner_nn)(x)
        dHdx = d_outer_nn * smstep(s_mag) + d_inner_nn * (1 - smstep(s_mag)) + \
                (outer_nn - inner_nn) * d_smstep(s_mag) * x/jnp.sqrt(mag2 + sms_del ** 2)

        # def computer_H_nn(x_):
        #     mag = jnp.sqrt(jnp.dot(x, x))
        #     return self.outer_net(x_) * smstep(mag) + self.inner_net(x_) * (1 - smstep(mag))
        # Hx, dHdx = jax.value_and_grad(computer_H_nn)(x)

        dxdt = (J - R) @ dHdx + B @ u
        y    = (B.T @ dHdx).squeeze()

        return dxdt, y, h_x


def train_model():
    train_key = key

    # ==================================
    # Load the training data
    # ==================================
    data_raw = np.load(cs.path_to_data)
    dx_raw = data_raw["Xs"]
    du_raw = data_raw["Us"]

    # Shape conventions:
    # full-order X: (n, T) -> (T, n)
    x_full = jnp.array(dx_raw.T)
    du = jnp.array(du_raw.reshape(-1, 1))

    # One-step supervised dataset for Euler integration
    xk_full = x_full[:-1]
    xkp1_full = x_full[1:]
    uk = du[:-1]

    dt = cs.stime
    dt_batch = jnp.full((xk_full.shape[0], 1), dt)

    # Create train-validation split
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
    # Construct the network and optimizer
    # ==================================
    nx = x_full.shape[1]
    nncells = cs.nncells

    mor_ph      = PHNN(n=nx, nncells=nncells)
    init_key, _ = jax.random.split(train_key, 2)
    params = mor_ph.init(init_key, jnp.ones((nx,)), jnp.ones((1,)))['params']

    patience       = cs.patience
    scheduler_step = cs.scheduler_step
    batch_size     = cs.batch_size
    lr             = cs.learning_rate
    num_epochs     = cs.num_epochs
    l2_lambda      = cs.l2_lambda
    weight_decay   = 1e-3
    inner_grad_lambda = getattr(cs, "inner_grad_lambda", 1e-2)


    lr_schedule = optax.linear_schedule(
        init_value = lr,
        end_value  = 2e-4,
        transition_steps = scheduler_step,
    )
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    vmapped_dxdt = jax.vmap(lambda p, x, u: mor_ph.apply({'params': p}, x, u)[0], in_axes=(None, 0, 0))

    def euler_one_step_projected(params_, x_full_batch, u_batch, dt_local):
        x_dot_pred = vmapped_dxdt(params_, x_full_batch, u_batch)
        return x_full_batch + dt_local * x_dot_pred

    def mse_state_loss(params_, x_full_batch, u_batch, dt_local, x_next_full_batch):
        x_next_pred = euler_one_step_projected(params_, x_full_batch, u_batch, dt_local)
        return jnp.mean((x_next_pred - x_next_full_batch) ** 2)

    def l2_regularization(params_):
        return sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(params_))

    def inner_grad_norm2(params_, x_full_batch):
        def inner_scalar(x_):
            return mor_ph.apply({"params": params_}, x_, jnp.zeros((1,)), method=PHNN.inner_only)
        grads = jax.vmap(jax.grad(inner_scalar))(x_full_batch)
        return jnp.mean(jnp.sum(grads ** 2, axis=-1))

    def train_loss(params_, x_full_batch, u_batch, dt_local, x_next_full_batch):
        data_loss = mse_state_loss(params_, x_full_batch, u_batch, dt_local, x_next_full_batch)
        grad_penalty = inner_grad_norm2(params_, x_full_batch)
        return data_loss + l2_lambda * l2_regularization(params_) + inner_grad_lambda * grad_penalty

    @jax.jit
    def train_step(params_, opt_state_, x_full_batch, u_batch, dt_local, x_next_full_batch):
        loss, grads = jax.value_and_grad(train_loss)(
            params_, x_full_batch, u_batch, dt_local, x_next_full_batch
        )
        updates, opt_state_ = optimizer.update(grads, opt_state_, params_)
        params_ = optax.apply_updates(params_, updates)
        return params_, opt_state_, loss

    @jax.jit
    def eval_step(params_, x_full_batch, u_batch, dt_local, x_next_full_batch):
        return mse_state_loss(params_, x_full_batch, u_batch, dt_local, x_next_full_batch)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting state-based training with projected Euler one-step loss...\n")

    for epoch in range(1, num_epochs + 1):
        total_train_loss = 0.0
        num_train_batches = int(np.ceil(train_size / batch_size))

        perm_key, train_key = jax.random.split(train_key)
        train_perm = jax.random.permutation(perm_key, train_size)

        for i in range(num_train_batches):
            batch_indices = train_perm[i * batch_size:(i + 1) * batch_size]
            x_b = train_dataset[0][batch_indices]
            u_b = train_dataset[1][batch_indices]
            dt_b = train_dataset[2][batch_indices]
            x_next_b = train_dataset[3][batch_indices]

            params, opt_state, loss = train_step(params, opt_state, x_b, u_b, dt_b, x_next_b)
            total_train_loss += loss

        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        total_val_loss = 0.0
        num_val_batches = int(np.ceil(val_size / batch_size))
        for i in range(num_val_batches):
            batch_indices = slice(i * batch_size, (i + 1) * batch_size)
            x_v     = val_dataset[0][batch_indices]
            u_v     = val_dataset[1][batch_indices]
            dt_v    = val_dataset[2][batch_indices]
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

        if epoch % 100 == 0:
            current_lr = lr_schedule(opt_state[-1].count)
            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                f"Val Loss = {avg_val_loss:.6f}, LR = {current_lr:.6e}"
            )

    print("\nTraining complete!\n")

    save_dir = "model/toda_lattice"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # msgpack_path = os.path.join(save_dir, f"NeuralToda_NN_sms_params_{timestamp}.msgpack")
    msgpack_path = os.path.join(save_dir, f"NeuralToda_NN_dual_sms_params{sms_or}-{sms_cm}.msgpack")

    with open(msgpack_path, "wb") as f:
        f.write(serialization.to_bytes(params))

    print(f"Saved trained model MSGPACK to: {msgpack_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("State MSE Loss")
    plt.title("Projected One-Step State Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_model()
