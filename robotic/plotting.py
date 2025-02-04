import json
import matplotlib.pyplot as plt

# Load the JSON files
with open("results_min_motion_sigma_0.1.json", "r") as f:
    results_0_1 = json.load(f)

with open("results_min_motion_sigma_0.5.json", "r") as f:
    results_0_5 = json.load(f)

# Extract data for plotting
step_sizes_0_1 = []
mean_steps_0_1 = []
std_steps_0_1 = []

for step_size, data in results_0_1["results"].items():
    step_sizes_0_1.append(1 / float(step_size) * 2)
    mean_steps_0_1.append(data["mean_steps"])
    std_steps_0_1.append(data["std_steps"])

step_sizes_0_5 = []
mean_steps_0_5 = []
std_steps_0_5 = []

for step_size, data in results_0_5["results"].items():
    step_sizes_0_5.append(1 / float(step_size) * 2)
    mean_steps_0_5.append(data["mean_steps"])
    std_steps_0_5.append(data["std_steps"])

# Plot the data
plt.figure(figsize=(4, 3))

plt.errorbar(
    step_sizes_0_1,
    mean_steps_0_1,
    yerr=std_steps_0_1,
    label="adaptive variance",
    fmt="-o",
)
plt.errorbar(
    step_sizes_0_5,
    mean_steps_0_5,
    yerr=std_steps_0_5,
    label="fixed variance",
    fmt="-o",
)

# plt.xscale("log")
plt.yscale("log")
plt.xlabel("Effective arena size")
plt.ylabel("Steps to target")
plt.legend(frameon=False)  # Remove legend border
plt.grid(False)  # Remove grid
plt.gca().spines["top"].set_visible(False)  # Remove top border
plt.gca().spines["right"].set_visible(False)  # Remove right border

plt.savefig("adaptive_variance_comparison.svg", format="svg", bbox_inches="tight")

plt.show()
