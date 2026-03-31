import glob
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import constant as cs
from flax import serialization
from matplotlib import rcParams
from scipy.integrate import solve_ivp
from NeuralToda_ICNN import PHNN as PHNN_ICNN
from NeuralToda_NN_dual_sms import PHNN as PHNN_SMS_dual


# JAX setup
key = jax.random.PRNGKey(0)

# Matplotlib settings
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "Times New Roman"


def load_latest_msgpack_params(template, pattern, label):
    ckpt_paths = glob.glob(pattern)
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")

    latest_ckpt = max(ckpt_paths, key=os.path.getmtime)
    with open(latest_ckpt, "rb") as f:
        loaded = serialization.from_bytes(template, f.read())

    print(f"Loaded {label} model checkpoint: {latest_ckpt}")
    return loaded


def build_icnn_params(rng_key, nx):
    nncells = cs.nncells
    icnn_depth = cs.nndepth
    model = PHNN_ICNN(n=nx, nncells=nncells, depth=icnn_depth)
    params_template = model.init(rng_key, jnp.ones((nx,)), jnp.ones((1,)))['params']
    params = load_latest_msgpack_params(
        params_template,
        os.path.join("model", "NeuralToda_ICNN_params.msgpack"),
        "ICNN",
    )
    return model, params



def build_sms_params_dual(rng_key, nx):
    nncells = cs.nncells
    model = PHNN_SMS_dual(n=nx, nncells=nncells)
    params_template = model.init(rng_key, jnp.ones((nx,)), jnp.ones((1,)))['params']
    params = load_latest_msgpack_params(
        params_template,
        os.path.join("model", "NeuralToda_NN_dual_sms_params1-1.msgpack"),
        "NN-dual-SMS",
    )
    return model, params


def make_u_of_t(t_samples, u_samples):
    t0 = float(t_samples[0])
    if len(t_samples) > 1:
        dt = float(t_samples[1] - t_samples[0])
    else:
        dt = float(cs.stime)

    def u_of_t(t):
        idx = int(np.floor((t - t0) / dt + 1e-12))
        if idx < 0:
            idx = 0
        elif idx >= len(u_samples):
            idx = len(u_samples) - 1
        return u_samples[idx]

    return u_of_t


def simulate_with_ode(model, params, x0, t_samples, u_samples):
    u_of_t = make_u_of_t(t_samples, u_samples)

    def rhs(t, x):
        u = u_of_t(t)
        x_j = jnp.array(x, dtype=jnp.float64)
        u_j = jnp.array(u, dtype=jnp.float64)
        dx, _, _ = model.apply({"params": params}, x_j, u_j)
        return np.array(dx)

    sol = solve_ivp(
        rhs,
        t_span=(float(t_samples[0]), float(t_samples[-1])),
        y0=np.array(x0, dtype=np.float64),
        t_eval=np.array(t_samples, dtype=np.float64),
        method="RK23",
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.y.T


for name in ["pulse", "sin", "step"]:
    test_data_name = f"{name}_TodaLat_data_test"
    test_signal_name = test_data_name.split("_", 1)[0]
    data_raw_test = np.load(os.path.join("data", f"{test_data_name}.npz"))
    print(f"Test signal: {test_signal_name}")

    test_DX_raw = data_raw_test["Xs"]
    test_DU_raw = data_raw_test["Us"]
    test_DY = data_raw_test["Ys"]
    test_H = data_raw_test["Hs"]
    test_Ts = data_raw_test["Ts"]

    test_X_full = jnp.array(test_DX_raw.T, dtype=jnp.float64)
    test_DU = np.array(test_DU_raw.reshape(-1, 1), dtype=np.float64)
    test_Ts = np.array(test_Ts, dtype=np.float64)

    x0 = np.array(test_X_full[0], dtype=np.float64)

    rng_key, key = jax.random.split(key)
    icnn_model, icnn_params = build_icnn_params(rng_key, cs.nx)

    rng_key, key = jax.random.split(key)
    # sms_model, sms_params = build_sms_params(rng_key, test_X_full.shape[1])

    rng_key, key = jax.random.split(key)
    sms_model_dual, sms_params_dual = build_sms_params_dual(rng_key, test_X_full.shape[1])

    icnn_traj = simulate_with_ode(icnn_model, icnn_params, x0, test_Ts, test_DU)
    sms_dual_traj = simulate_with_ode(sms_model_dual, sms_params_dual, x0, test_Ts, test_DU)

    def eval_outputs(model, params, x_traj, u_samples, shift = False):
        x_j = jnp.array(x_traj, dtype=jnp.float64)
        u_j = jnp.array(u_samples, dtype=jnp.float64)
        batch_apply = jax.jit(jax.vmap(lambda x, u: model.apply({"params": params}, x, u), in_axes=(0, 0)))
        _, y_est, h_est = batch_apply(x_j, u_j)

        if shift == True:
            return np.array(y_est), np.array(h_est) - h_est[0]

        return np.array(y_est), np.array(h_est)

    icnn_y, icnn_h = eval_outputs(icnn_model, icnn_params, icnn_traj, test_DU)
    sms_dual_y, sms_dual_h = eval_outputs(sms_model_dual, sms_params_dual, sms_dual_traj, test_DU, shift = True)

    true_state_norm     = np.linalg.norm(np.array(test_X_full), axis=1)
    icnn_state_norm     = np.linalg.norm(icnn_traj, axis=1)
    sms_dual_state_norm = np.linalg.norm(sms_dual_traj, axis=1)

    plt.rcParams.update({"font.size": 20})
    output_dir = os.path.join("figs")
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(test_Ts, true_state_norm, label="Actual PHS", color="black")
    plt.plot(test_Ts, icnn_state_norm, label="PH-ICNN", color="blue", linestyle="--")
    plt.plot(test_Ts, sms_dual_state_norm, label="Proposed method", color="orange", linestyle="-.")
    plt.xlabel("Time (seconds)")
    plt.xlim(0, float(test_Ts[-1]))
    plt.grid(True)
    if name == "pulse":
        plt.legend(loc="best", ncol=1)
    elif name == "sin":
        plt.legend(loc="best", ncol=2)
    else:
        plt.legend(loc="best", ncol=2)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{test_signal_name}_Toda_stateMagnitude_ODE.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(8, 5))
    plt.plot(test_Ts, np.array(test_DY), label="Actual PHS", color="black")
    plt.plot(test_Ts, icnn_y, label="PH-ICNN", color="blue", linestyle="--")
    plt.plot(test_Ts, sms_dual_y, label="Proposed method", color="orange", linestyle="-.")
    plt.xlabel("Time (seconds)")
    plt.xlim(0, float(test_Ts[-1]))
    if test_signal_name == "sin":
        plt.ylim(jnp.min(test_DY) - 0.05, jnp.max(test_DY) + 0.2)
    plt.grid(True)
    plt.legend(loc="best", ncol=2)
    if name == "sin":
        plt.ylim(-0.3, 0.5)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{test_signal_name}_Toda_output_ODE.pdf"),
        format="pdf",
        bbox_inches="tight",
    )



    fig = plt.figure(figsize=(8, 5))
    plt.plot(test_Ts, np.array(test_H), label="Actual PHS", color="black")
    plt.plot(test_Ts, icnn_h, label="PH-ICNN", color="blue", linestyle="--")
    plt.plot(test_Ts, sms_dual_h, label="Proposed method", color="orange", linestyle="-.")
    plt.xlabel("Time (seconds)")
    plt.xlim(0, float(test_Ts[-1]))
    plt.grid(True)

    if name == "pulse":
        plt.legend(loc="best", ncol=1)
    elif name == "sin":
        plt.legend(loc="best", ncol=2)
    else:
        plt.legend(loc="best", ncol=2)


    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{test_signal_name}_Toda_Hamiltonian_ODE.pdf"),
        format="pdf",
        bbox_inches="tight",
    )


    print(f"Plots saved in '{output_dir}' directory.")
