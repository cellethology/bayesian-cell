import numpy as np
import matplotlib.pyplot as plt
from comparative_run import run_comparative_simulation
from bayes_navigation import BayesianNavigation

# === Configuration ===
seed = 1
noise_std = 0.6
max_steps = 200000

base_config = {
    "true_motion_sigma": 0.6,
    "process_sigma_estimate": 0.6,
    "adaptive_rate": 0.4,
    "signal_max": 5,
    "signal_decay": 0.05,
    "step_size": 0.02,
    "kernel_size": 5,
    "noise_model": "gaussian",
    "noise_std": noise_std,
    "noise_estimate": noise_std,
}

config_labels = [
    "Fixed process variance",
    "Adaptive process variance",
]

# === Run the simulations ===
configs, results = run_comparative_simulation(seed, base_config, max_steps)

# Extract trajectories
trajectory_fixed = np.array(results[0])
trajectory_adaptive = np.array(results[1])

# Print the number of steps to target
print(f"Number of steps to target for fixed process: {len(trajectory_fixed)}")
print(f"Number of steps to target for adaptive process: {len(trajectory_adaptive)}")

# === Generate the signal map from a dummy agent ===
dummy_env = BayesianNavigation(config=base_config)
signal_map = dummy_env.compute_all_expected_signal(dummy_env.true_target_pos)

target_x, target_y = dummy_env.true_target_pos

# === Plotting ===
plt.figure(figsize=(4.5, 3.6))
plt.imshow(signal_map, cmap="Greens", origin="lower")
cbar = plt.colorbar()
cbar.set_label("Expected Signal Strength")

# Plot fixed trajectory
plt.plot(
    trajectory_fixed[:, 1],
    trajectory_fixed[:, 0],
    "-",
    color="tab:blue",
    label="Standard KF",
    linewidth=2,
)

# Plot adaptive trajectory
plt.plot(
    trajectory_adaptive[:, 1],
    trajectory_adaptive[:, 0],
    "-",
    color="tab:orange",
    label="Signal-aware KF",
    linewidth=2,
)

# Add green star for the target
plt.plot(target_y, target_x, "*", color="tab:pink", markersize=12, label="Target")

# Add white square at the start
start_x, start_y = trajectory_fixed[0]
plt.plot(start_y, start_x, "s", color="black", markersize=6, label="Start")

# Formatting
plt.legend(loc="upper left", fontsize=11, frameon=False)
plt.tight_layout()
plt.savefig(
    "/Users/jerrywang/Thomson Lab Dropbox/Jerry Wang/Apps/Overleaf/bayesian_filter_2024/figure/assets/example_trajectory_overlay.pdf",
    dpi=300,
)
plt.show()
