import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON files
with open("tmp/adaptive_results.json", "r") as f:
    adaptive_results = json.load(f)

with open("tmp/fixed_results.json", "r") as f:
    fixed_results = json.load(f)

import numpy as np


def compute_average_signal(signal_strength_max, signal_decay_exp, d_min, d_max):
    # Exact solution using integration
    if signal_decay_exp == 1:
        # Special case: log integral
        exact_avg_signal = (
            signal_strength_max * (np.log(d_max) - np.log(d_min)) / (d_max - d_min)
        )
    else:
        # General case: power integral
        exact_avg_signal = (signal_strength_max / (d_max - d_min)) * (
            (d_max ** (1 - signal_decay_exp) - d_min ** (1 - signal_decay_exp))
            / (1 - signal_decay_exp)
        )

    return exact_avg_signal


signal_strength_max = 0.2
signal_decay_exp = 0.3
d_min = 5
d_max = 100

exact_avg = compute_average_signal(signal_strength_max, signal_decay_exp, d_min, d_max)
print(f"Exact Average Signal: {exact_avg}")

# Extract data for plotting
param_adaptive = []
mean_steps_adaptive = []
std_steps_adaptive = []

for param, data in adaptive_results.items():
    param_adaptive.append(exact_avg / float(param))
    mean_steps_adaptive.append(np.mean(data))
    std_steps_adaptive.append(np.std(data))

param_fixed = []
mean_steps_fixed = []
std_steps_fixed = []

for param, data in fixed_results.items():
    param_fixed.append(exact_avg / float(param))
    mean_steps_fixed.append(np.mean(data))
    std_steps_fixed.append(np.std(data))

# Plot the data
plt.figure(figsize=(4, 3))
plt.errorbar(
    param_adaptive,
    mean_steps_adaptive,
    yerr=std_steps_adaptive,
    label="adaptive variance",
    fmt="-o",
)
plt.errorbar(
    param_fixed,
    mean_steps_fixed,
    yerr=std_steps_fixed,
    label="fixed variance",
    fmt="-o",
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("signal noise ratio")
plt.ylabel("Steps to target")
plt.legend(frameon=False)  # Remove legend border
plt.grid(False)  # Remove grid
plt.gca().spines["top"].set_visible(False)  # Remove top border
plt.gca().spines["right"].set_visible(False)  # Remove right border

plt.savefig("adaptive_sigma_comparison.pdf", format="pdf", bbox_inches="tight")

plt.show()
