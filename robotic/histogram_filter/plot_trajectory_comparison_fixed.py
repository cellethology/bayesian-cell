def plot_trajectory_comparison(
    self,
    trajectories_data: dict,
    n_plots: int = 4,
    figsize: tuple = (16, 4),
    save_path: str = None,
):
    """
    Plot trajectory comparisons with both methods on same subplot (same seed).

    Args:
        trajectories_data: Output from run_trajectory_comparison
        n_plots: Number of subplots to show (one per seed)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib figure
    """
    config_names = list(trajectories_data.keys())
    if len(config_names) != 2:
        raise ValueError("Exactly two configurations required")

    config1, config2 = config_names
    n_runs = min(len(trajectories_data[config1]), len(trajectories_data[config2]))
    n_plots = min(n_plots, n_runs)

    # Create subplots - single row with n_plots columns
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    for i in range(n_plots):
        ax = axes[i]
        
        # Get data for both configurations for the same seed
        traj_data1 = trajectories_data[config1][i]
        traj_data2 = trajectories_data[config2][i]
        
        trajectory1 = np.array(traj_data1["trajectory"])
        trajectory2 = np.array(traj_data2["trajectory"])
        env1 = traj_data1["env"]
        seed = traj_data1["seed"]
        
        # Use the first environment for the signal map
        signal_map = env1.signal_model.compute_all_expected_signal(env1.true_target_pos)
        
        # Plot signal map
        im = ax.imshow(signal_map, cmap="Greens", interpolation="nearest")
        
        # Plot both trajectories with different colors
        ax.plot(trajectory1[:, 1], trajectory1[:, 0], "b-", linewidth=2, 
               label=f"{config1} ({len(trajectory1)} steps)", alpha=0.8)
        ax.plot(trajectory2[:, 1], trajectory2[:, 0], "r-", linewidth=2, 
               label=f"{config2} ({len(trajectory2)} steps)", alpha=0.8)
        
        # Plot start position (should be the same for both)
        ax.plot(trajectory1[0, 1], trajectory1[0, 0], "bs", markersize=8, label="Start")
        
        # Plot end positions with different markers
        ax.plot(trajectory1[-1, 1], trajectory1[-1, 0], "bo", markersize=6, alpha=0.8)
        ax.plot(trajectory2[-1, 1], trajectory2[-1, 0], "ro", markersize=6, alpha=0.8)
        
        # Plot target
        ax.plot(env1.true_target_pos[1], env1.true_target_pos[0], "g*", 
               markersize=15, label="Target")
        
        # Formatting
        ax.set_title(f"Seed {seed}")
        ax.legend(fontsize=9)
        
        # Add colorbar only to the last subplot
        if i == n_plots - 1:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Signal Strength")

    plt.suptitle(f"Trajectory Comparison: {config1} vs {config2}", fontsize=16)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Trajectory plot saved to {save_path}")

    return fig