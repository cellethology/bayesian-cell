from bayes_navigation import BayesianNavigation
import matplotlib.pyplot as plt

# Get trajectory from json file
import json
import numpy as np

with open("tmp/results_seed_7.json", "r") as f:
    result = json.load(f)
    fixed_trajectory = np.array(result["fixed_trajectory"])
    adaptive_trajectory = np.array(result["adaptive_trajectory"])

config = {
    "grid_size": 100,
    "measurement_noise_factor": 0.06,
    "signal_strength_max": 0.2,
    "signal_decay_exp": 0.3,
}

env = BayesianNavigation(config)
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# Plot the belief map and trajectory
signal_map = env.compute_all_expected_signal(env.true_target_pos)
ax.imshow(signal_map, cmap="hot", interpolation="nearest")
# add colorbar to the right of the plot with proper size
cbar = plt.colorbar(ax.imshow(signal_map, cmap="hot", interpolation="nearest"))
cbar.ax.tick_params(labelsize=8)
# add title to the colorbar, rotate it 180 degrees, add space between title and colorbar
cbar.set_label("Expected Signal Strength", rotation=270, labelpad=19, fontsize=12)


ax.plot(
    fixed_trajectory[:, 1],
    fixed_trajectory[:, 0],
    "tab:orange",
    label="fixed variance",
    linewidth=1.5,
)
ax.plot(
    adaptive_trajectory[:, 1],
    adaptive_trajectory[:, 0],
    "tab:blue",
    label="adaptive variance",
    linewidth=1.5,
)
ax.plot(
    env.true_target_pos[1],
    env.true_target_pos[0],
    "g*",
    markersize=15,
    label="Target",
)
ax.plot(fixed_trajectory[0, 1], fixed_trajectory[0, 0], "ws", label="Start")
# legend with no frame and no background, white text
ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor="white")


plt.tight_layout()
plt.savefig("bayes_navigation.pdf", format="pdf", dpi=300)
plt.show()
