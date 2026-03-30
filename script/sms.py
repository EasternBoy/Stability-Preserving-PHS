import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from NeuralToda_NN_sms import d_smstep, sig, smstep
import constant as cs


jax.config.update("jax_enable_x64", True)


def main():
    b = float(sig(cs.sms_cm))
    sigma_max = max(2.0 * b, 1.5 * cs.sms_cm)
    sigma = jnp.linspace(0.0, sigma_max, 1000)

    smstep_vals = jax.vmap(smstep)(sigma)
    grad_vals = jax.vmap(jax.grad(smstep))(sigma)
    manual_grad_vals = jax.vmap(d_smstep)(sigma)
    grad_error = grad_vals - manual_grad_vals

    output_dir = os.path.join("figs")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(np.array(sigma), np.array(smstep_vals), color="black", label="smstep")
    axes[0].axvline(b, color="gray", linestyle="--", label=f"b = {b:.5f}")
    axes[0].set_ylabel("smstep")
    axes[0].grid(True)
    axes[0].legend(frameon=False)

    axes[1].plot(np.array(sigma), np.array(grad_vals), color="tab:blue", label="jax.grad(smstep)")
    axes[1].plot(
        np.array(sigma),
        np.array(manual_grad_vals),
        color="tab:red",
        linestyle="--",
        label="d_smstep",
    )
    axes[1].set_ylabel("Derivative")
    axes[1].grid(True)
    axes[1].legend(frameon=False)

    axes[2].plot(np.array(sigma), np.array(grad_error), color="tab:green", label="grad - d_smstep")
    axes[2].axhline(0.0, color="gray", linestyle="--")
    axes[2].set_xlabel("sigma")
    axes[2].set_ylabel("Error")
    axes[2].grid(True)
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "sms_comparison.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    max_abs_error = float(jnp.max(jnp.abs(grad_error)))
    rms_error = float(jnp.sqrt(jnp.mean(grad_error ** 2)))

    print(f"Transition point b = {b:.10f}")
    print(f"Max abs error between jax.grad(smstep) and d_smstep: {max_abs_error:.10e}")
    print(f"RMS error between jax.grad(smstep) and d_smstep: {rms_error:.10e}")
    print(f"Saved plot to {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
