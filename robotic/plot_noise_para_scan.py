import numpy as np
import matplotlib.pyplot as plt

# Load the saved data
data = np.load("output/noise_scan_results.npz", allow_pickle=True)
snr_values = data["snr_values"]
all_steps = data["all_steps"].item()  # This was saved as a dict

# Plotting
plt.figure(figsize=(6, 5))
for label, step_lists in all_steps.items():
    steps_array = np.array(step_lists)
    medians = np.nanmedian(steps_array, axis=1)
    q25 = np.nanpercentile(steps_array, 25, axis=1)
    q75 = np.nanpercentile(steps_array, 75, axis=1)

    plt.errorbar(
        snr_values,
        medians,
        yerr=[medians - q25, q75 - medians],
        fmt="o-",
        label=label,
        capsize=4,
    )

# Beautify
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xscale("log")
plt.xlabel("Signal-to-Noise Ratio (SNR)", fontsize=16)
plt.ylabel("Mean Steps to Target", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=16, frameon=False)
plt.tight_layout()
plt.savefig(
    "/Users/jerrywang/Thomson Lab Dropbox/Jerry Wang/Apps/Overleaf/bayesian_filter_2024/figure/assets/step_vs_noise_plot.pdf",
    dpi=300,
)
plt.show()
