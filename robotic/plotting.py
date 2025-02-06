import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON files
with open("tmp/adaptive_results.json", "r") as f:
    adaptive_results = json.load(f)

with open("tmp/fixed_results.json", "r") as f:
    fixed_results = json.load(f)

# Extract data for plotting
param_adaptive = []
mean_steps_adaptive = []
std_steps_adaptive = []

for param, data in adaptive_results.items():
    param_adaptive.append(0.2/float(param))
    mean_steps_adaptive.append(np.mean(data))
    std_steps_adaptive.append(np.std(data))

param_fixed = []
mean_steps_fixed = []
std_steps_fixed = []

for param, data in fixed_results.items():
    param_fixed.append(0.2/float(param))
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
