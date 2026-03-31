import os
import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import constant as cs
from NeuralDpen_NN_sms import PHNN as PHNN_NNsms
from jax.experimental.ode import odeint

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    "font.size": 16,        # base size
    "axes.titlesize": 24,   # title
    "axes.labelsize": 22,   # x/y labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 14
})

jax.config.update("jax_enable_x64", True)

def activation_select(act_type):
    if act_type == "gelu":
        return nn.gelu
    if act_type == "relu":
        return nn.relu
    if act_type == "softplus":
        return nn.softplus

act_func = [activation_select(cs.layers_type[i]) for i in range(cs.nndepth)]

class HNet(nn.Module):
    nncells: int

    @nn.compact
    def __call__(self, x):
        for i in range(cs.nndepth - 1):
            x = nn.Dense(features=self.nncells)(x)
            x = act_func[i](x)
            
        x = nn.Dense(features=1)(x)
        x = act_func[cs.nndepth - 1](x)
        return x.squeeze()

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
    model_type: str = "standard" 

    def setup(self):
        if self.model_type == "icnn":
            self.H_net = ICNNHNet(nncells=self.nncells, depth=cs.nndepth)
        else:
            self.H_net = HNet(nncells=self.nncells)

    @nn.compact
    def __call__(self, x):
        n_half = self.n // 2
        J = jnp.block([
            [jnp.zeros((n_half, n_half)), jnp.eye(n_half)],
            [-jnp.eye(n_half), jnp.zeros((n_half, n_half))],
        ])
        
        gamma_val = self.param("gamma", nn.initializers.ones, (n_half,))
        gamma_val = jnp.clip(gamma_val, 0.49, 0.5) 
        print("gamma_val:", gamma_val)
        
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

def test_models(nn_model_path, icnn_model_path, NNsms_MODEL_FILE, test_data_path):
    print(f"Loading test data from {test_data_path}...")
    data_raw = np.load(test_data_path)
    
    xk_flat = jnp.array(data_raw["Xk"].T, dtype=jnp.float64)
    xkp1_flat = jnp.array(data_raw["Xkp1"].T, dtype=jnp.float64)
    nx = xk_flat.shape[1]
    seq_len = 200 
    num_test_traj = xk_flat.shape[0] // seq_len
    
    print(f"Detected {num_test_traj} trajectories of length {seq_len}...")
    
    xk_traj = xk_flat[:num_test_traj * seq_len].reshape(num_test_traj, seq_len, nx)
    xkp1_traj = xkp1_flat[:num_test_traj * seq_len].reshape(num_test_traj, seq_len, nx)

    x0_test = xk_traj[:, 0, :]
    target_traj = xkp1_traj[:, :, :]
    print("target_taj:", target_traj)
    dummy_key = jax.random.PRNGKey(0)

    def odeint_rollout(model_apply_fn, params_, x0_single, length):
        def get_dx(state, t, p):
            return model_apply_fn({"params": p}, state)[0]
        t_eval = jnp.arange(length + 1) * cs.dt
        trajectory = odeint(get_dx, x0_single, t_eval, params_)
        return trajectory[1:]

    print("Initializing Standard PHNN...")
    phnn_nn = PHNN(n=nx, nncells=cs.nncells, model_type="standard")
    empty_params_nn = phnn_nn.init(dummy_key, jnp.ones((nx,)))["params"]

    with open(nn_model_path, "rb") as f:
        params_nn = serialization.from_bytes(empty_params_nn, f.read())

    vmapped_rollout_nn = jax.vmap(
        lambda p, x, l: odeint_rollout(phnn_nn.apply, p, x, l), 
        in_axes=(None, 0, None)
    )

    print("Running full trajectory predictions for Standard PHNN...")
    xkp1_pred_nn = vmapped_rollout_nn(params_nn, x0_test, seq_len)
    mse_nn = jnp.mean((xkp1_pred_nn[:, :, :2] - target_traj[:, :, :2]) ** 2)

    print("Initializing ICNN PHNN...")
    phnn_icnn = PHNN(n=nx, nncells=cs.nncells, model_type="icnn")
    empty_params_icnn = phnn_icnn.init(dummy_key, jnp.ones((nx,)))["params"]

    with open(icnn_model_path, "rb") as f:
        params_icnn = serialization.from_bytes(empty_params_icnn, f.read())

    vmapped_rollout_icnn = jax.vmap(
        lambda p, x, l: odeint_rollout(phnn_icnn.apply, p, x, l), 
        in_axes=(None, 0, None)
    )

    print("Running full trajectory predictions for ICNN PHNN...")
    xkp1_pred_icnn = vmapped_rollout_icnn(params_icnn, x0_test, seq_len)
    mse_icnn = jnp.mean((xkp1_pred_icnn[:, :, :2] - target_traj[:, :, :2]) ** 2)

    print("Initializing NN sms...")
    phnn_NNsms = PHNN_NNsms(n=nx, nncells=cs.nncells)
    empty_params_sms = phnn_NNsms.init(dummy_key, jnp.ones((nx,)))["params"]

    with open(NNsms_MODEL_FILE, "rb") as f:
        params_sms = serialization.from_bytes(empty_params_sms, f.read())

    vmapped_rollout_sms = jax.vmap(
        lambda p, x, l: odeint_rollout(phnn_NNsms.apply, p, x, l), 
        in_axes=(None, 0, None)
    )

    print("Running full trajectory predictions for NN sms...")
    xkp1_pred_NNsms = vmapped_rollout_sms(params_sms, x0_test, seq_len)
    mse_sms = jnp.mean((xkp1_pred_NNsms[:, :, :2] - target_traj[:, :, :2]) ** 2)    

    print(f"\n======================================")
    print(f"FULL TRAJECTORY TEST MSE LOSS (PHNN): {mse_nn:.6e}")
    print(f"FULL TRAJECTORY TEST MSE LOSS (ICNN): {mse_icnn:.6e}")
    print(f"FULL TRAJECTORY TEST MSE LOSS (The proposed method): {mse_sms:.6e}")
    print(f"======================================\n")

    traj_idx = 0
    
    t_steps_full = np.round(np.arange(seq_len + 1)*0.05, 2)
    
    def prepend_zero(arr):
        return jnp.concatenate([jnp.array([0.0]), arr])
    
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    
    for spine in axs[0].spines.values():
        spine.set_linewidth(0.5)

    for spine in axs[1].spines.values():
        spine.set_linewidth(0.5)

    axs[0].plot(t_steps_full, prepend_zero(target_traj[traj_idx, :, 0]), label="Actual PHS", color="black", linewidth=0.5)
    axs[0].plot(t_steps_full, prepend_zero(xkp1_pred_icnn[traj_idx, :, 0]), label="PH-ICNN", color="blue", linestyle="--", alpha=1.0, linewidth=0.5)
    axs[0].plot(t_steps_full, prepend_zero(xkp1_pred_NNsms[traj_idx, :, 0]), label="Proposed method", color="orange", linestyle="-.", alpha=1.0, linewidth=1.)
    axs[0].set_ylabel(r"$\theta_1$", labelpad=-5)
    
    axs[0].grid(True)
    axs[0].set_yticks([-np.pi, 0, np.pi, 2*np.pi])
    axs[0].set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
    axs[0].tick_params(axis='y')
    axs[0].tick_params(axis='x')
    axs[0].legend(loc="right", ncol=1, frameon=False)
    
    # Plot 2
    axs[1].plot(t_steps_full, prepend_zero(target_traj[traj_idx, :, 1]), label="Actual PHS", color="black", linewidth=0.50)
    axs[1].plot(t_steps_full, prepend_zero(xkp1_pred_icnn[traj_idx, :, 1]), label="PH-ICNN", color="blue", linestyle="--", alpha=1.0, linewidth=0.50)
    axs[1].plot(t_steps_full, prepend_zero(xkp1_pred_NNsms[traj_idx, :, 1]), label="Proposed method", color="orange", linestyle="-.", alpha=1.0, linewidth=1.0)
    axs[1].set_ylabel(r"$\theta_2$", labelpad=-5)
    axs[1].grid(True)
    axs[1].set_yticks([ -np.pi, 0, np.pi, 2*np.pi])
    axs[1].set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
    axs[1].tick_params(axis='y')
    
    axs[1].tick_params(axis='x')

    def prepend_p1(arr):
        return jnp.concatenate([jnp.array([10.0]), arr])
    
    def prepend_p2(arr):
        return jnp.concatenate([jnp.array([-2.0]), arr])

    p1_truth = prepend_p1(target_traj[traj_idx, :, 2])
    p1_icnn = prepend_p1(xkp1_pred_icnn[traj_idx, :, 2])
    p1_proposed = prepend_p1(xkp1_pred_NNsms[traj_idx, :, 2])

    p2_truth = prepend_p2(target_traj[traj_idx, :, 3])
    p2_icnn = prepend_p2(xkp1_pred_icnn[traj_idx, :, 3])
    p2_proposed = prepend_p2(xkp1_pred_NNsms[traj_idx, :, 3])

    
    axs[2].plot(t_steps_full, p1_truth, label=r"Actual PHS-$p_1$", color="black", linewidth=1.0)
    axs[2].plot(t_steps_full, p1_icnn, label=r"PH-ICNN-$p_1$", color="green", linestyle="--", alpha=1.0, linewidth=0.50)
    axs[2].plot(t_steps_full, p1_proposed, label=r"Proposed method-$p_1$", color="red", linestyle="-.", alpha=1.0, linewidth=0.5)
    axs[2].plot(t_steps_full, p2_truth, label=r"Actual PHS-$p_2$", color="brown", linewidth=1.0)
    axs[2].plot(t_steps_full, p2_icnn, label=r"PH-ICNN-$p_2$", color="blue", linestyle="--", alpha=1.0, linewidth=0.50)
    axs[2].plot(t_steps_full, p2_proposed, label=r"Proposed method-$p_2$", color="orange", linestyle="-.", alpha=1.0, linewidth=1.0)
    axs[2].set_ylabel(r"$p_1-p_2$", labelpad=0)
    axs[2].set_xlabel(r"Time (seconds)", fontsize=18)
    axs[2].grid(True)
    axs[2].legend(loc="upper right", ncol=2, frameon=False, fontsize=12)
    

    plt.tight_layout()
    plt.savefig("./Draw/traj_DPen_X0_1.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    
if __name__ == "__main__":
    nn_MODEL_FILE = os.path.join("model","double_pendulum","NN_DPen_FullTraj_dt0.05.msgpack")
    ICNN_MODEL_FILE = os.path.join("model","double_pendulum", "ICNN_DPen_FullTraj_dt0.05.msgpack")
    NNsms_MODEL_FILE = os.path.join("model","double_pendulum", "NN_DPen_sms_dt0.05_0.5.msgpack")

    # TEST_DATA_FILE = os.path.join("data", "double_pendulum", "double_pendulum_testX02_dt0.05.npz")
    # TEST_DATA_FILE = os.path.join("data", "double_pendulum", "double_pendulum_testX03_dt0.05.npz")
    TEST_DATA_FILE = os.path.join("data", "double_pendulum", "double_pendulum_testX01_dt0.05.npz")
    test_models(nn_MODEL_FILE, ICNN_MODEL_FILE, NNsms_MODEL_FILE, TEST_DATA_FILE)