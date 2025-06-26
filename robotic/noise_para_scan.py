import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from comparative_run import run_parallel_comparative_runs

MEAN_SIGNAL_STRENGTH = 0.8  # TODO: hard coded for now, need to change

if __name__ == "__main__":
    base_config = {
        "process_sigma": 1,
        "process_sigma_estimate": 0.0001,
        "adaptive_rate": 0.8,
        "signal_max": 1,
        "signal_decay": 0.2,
        "step_size": 1,
        "kernel_size": 5,
        "noise_model": "gaussian",
    }

    noise_values = np.linspace(0.5, 0.5, 1)
    n_repeats = 12
    max_steps = 1000000

    config_labels = [
        "Standard KF",
        "Signal-aware KF",
    ]

    all_steps = {label: [[] for _ in noise_values] for label in config_labels}

    for noise_idx, noise_std in enumerate(tqdm(noise_values, desc="Running SNR sweep")):
        config = base_config.copy()
        config["noise_std"] = noise_std
        config["measurement_sigma_estimate"] = noise_std
        results = run_parallel_comparative_runs(
            config, n_pairs=n_repeats, max_steps=max_steps, verbose=False
        )
        for configs, run_results in results:
            for i, traj in run_results.items():
                label = config_labels[i]
                steps = len(traj)
                all_steps[label][noise_idx].append(steps)

    for label in config_labels:
        print(f"\nSteps for {label}:")
        for noise_idx, noise_std in enumerate(noise_values):
            print(f"Noise std {noise_std:.2f}: {all_steps[label][noise_idx]}")

    # Plot median steps with quantile error bars
    snr_values = MEAN_SIGNAL_STRENGTH**2 / noise_values**2
    plt.figure(figsize=(6, 5))
    for label in config_labels:
        steps_array = np.array(all_steps[label])  # shape: (n_noise, n_repeats)
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

    # Clean up axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log")
    plt.xlabel("Signal-to-Noise Ratio (SNR)", fontsize=13)
    plt.ylabel("Median Steps to Target", fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig("output/step_vs_noise_plot.pdf", dpi=300)
    plt.show()

    # Save experiment data
    np.savez(
        "output/noise_scan_results.npz",
        max_steps=max_steps,
        snr_values=snr_values,
        all_steps=all_steps,
    )
