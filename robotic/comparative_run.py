import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from bayes_navigation import run_navigation_simulation


def run_comparative_simulation(seed, base_config, max_steps, verbose=False):
    np.random.seed(seed + 1000)
    all_configs_to_try = [
        {"initial_process_sigma": 0.001, "adaptive_process_variance": "none"},
        {"initial_process_sigma": 2, "adaptive_process_variance": "none"},
    ]
    results = {}
    for i, config in enumerate(all_configs_to_try):
        config_true = base_config.copy()
        config_true.update(config)
        trajectory, _, _, _, _ = run_navigation_simulation(
            config=config_true, steps=max_steps, verbose=verbose
        )
        results[i] = trajectory
    return all_configs_to_try, results


def _run_seed(args):
    seed, base_config, max_steps, verbose = args
    return run_comparative_simulation(seed, base_config, max_steps, verbose)


def run_parallel_comparative_runs(
    base_config, n_pairs=20, max_steps=20000, verbose=False
):
    seeds = list(range(n_pairs))
    results = []
    with Pool() as pool:
        args = [(seed, base_config, max_steps, verbose) for seed in seeds]
        for result in tqdm(pool.imap_unordered(_run_seed, args), total=n_pairs):
            results.append(result)
    return results


if __name__ == "__main__":
    base_config = {
        "true_process_sigma": 2,
        "initial_process_sigma": 0.001,  # true_process_sigma * step_size
        "motion_decay_rate": 0.8,  # Irrelevant when min == max
        "signal_strength_max": 2,
        "signal_decay_exp": 0.5,
        "step_size": 1,
        "kernel_size": 5,
        "adaptive_filtering": False,
        "adaptive_process_variance": "none",
        "noise_model": "gaussian",
        "noise_std": 0.5,
        "initial_measurement_sigma": 0.5,
    }
    # Prepare labels
    config_labels = [
        "Standard KF",
        "Signal-aware KF",
    ]

    max_steps = 1000000
    results = run_parallel_comparative_runs(
        base_config, n_pairs=10, max_steps=max_steps, verbose=False
    )

    all_configs, all_trajectories = zip(*results)

    # Initialize containers for step counts
    n_configs = len(all_configs[0])
    steps_per_config = [[] for _ in range(n_configs)]

    # Gather steps for each configuration across all runs
    for run_trajectories in all_trajectories:
        for idx, traj in run_trajectories.items():
            steps_per_config[idx].append(len(traj))

    steps_per_config = [np.array(steps) for steps in steps_per_config]
    success_per_config = [steps < max_steps for steps in steps_per_config]

    # Print success rates and mean steps
    for i, label in enumerate(config_labels):
        print(
            f"{label}: Success Rate = {success_per_config[i].mean():.2f},"
            f"Median steps to target = {np.mean(steps_per_config[i][success_per_config[i]]):.1f}"
        )

    # Plot success rate and steps in the same figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot success rate
    colors = ["gray", "orange", "green"]
    ax1.bar(
        config_labels,
        [s.mean() for s in success_per_config],
        color=colors,
    )
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate Comparison")
    for i, v in enumerate([s.mean() for s in success_per_config]):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Plot steps to target
    bp = ax2.boxplot(
        [
            steps[success]
            for steps, success in zip(steps_per_config, success_per_config)
        ],
        positions=range(1, n_configs + 1),
        widths=0.6,
        patch_artist=True,
        labels=config_labels,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax2.set_ylabel("Steps to Target")
    ax2.set_title("Steps to Target (Successful Runs Only)")

    plt.tight_layout()
    plt.show()
