#!/usr/bin/env python3
"""
Main entry point for target tracking simulation with FilterPy-based filters.
Supports both EKF and UKF with comprehensive visualization and CLI interface.

Usage:
    python main.py --filter EKF --adaptive-process
    python main.py --filter UKF --adaptive-measurement --signal-max 25
    python main.py --config custom_config.json --save-plots --verbose
"""

import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from core import EKFEnvironment, EKFVisualizer, get_base_config
from shared import (
    ensure_output_directory, 
    print_config_summary, 
    generate_default_output_path,
    format_time_delta,
    load_config_from_file,
    save_config_to_file,
    validate_filter_config
)


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Target tracking simulation with FilterPy-based filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --filter EKF --adaptive-process
  python main.py --filter UKF --adaptive-measurement --signal-max 25
  python main.py --config my_config.json --save-plots --verbose
  python main.py --filter EKF --max-steps 50000 --output-dir results/
        """
    )

    # Filter configuration
    parser.add_argument(
        "--filter", 
        choices=["EKF", "UKF"], 
        default="EKF",
        help="Filter type to use (default: EKF)"
    )

    # Adaptation options
    parser.add_argument(
        "--adaptive-process", 
        action="store_true",
        help="Enable adaptive process noise"
    )
    parser.add_argument(
        "--adaptive-measurement", 
        action="store_true",
        help="Enable adaptive measurement noise"
    )

    # Signal parameters
    parser.add_argument(
        "--signal-max", 
        type=float,
        help="Maximum signal strength (c0)"
    )
    parser.add_argument(
        "--signal-decay", 
        type=float,
        help="Signal decay rate (lambda)"
    )

    # Simulation parameters
    parser.add_argument(
        "--max-steps", 
        type=int,
        help="Maximum simulation steps"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        help="Random seed for reproducibility"
    )

    # Robot parameters
    parser.add_argument(
        "--robot-step-size", 
        type=float,
        help="Robot step size per time step"
    )
    parser.add_argument(
        "--actuator-noise", 
        type=float,
        help="Robot actuator noise standard deviation"
    )

    # Target parameters
    parser.add_argument(
        "--target-motion-sigma", 
        type=float,
        help="Target random walk noise standard deviation"
    )

    # Configuration file
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to JSON configuration file (overrides other options)"
    )

    # Output options
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Output directory for plots and results (default: output)"
    )
    parser.add_argument(
        "--save-plots", 
        action="store_true",
        help="Save plots to files"
    )
    parser.add_argument(
        "--no-show-plots", 
        action="store_true",
        help="Don't display plots interactively"
    )
    parser.add_argument(
        "--save-config", 
        type=str,
        help="Save final configuration to JSON file"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true",
        help="Suppress non-essential output"
    )

    # UKF-specific parameters
    parser.add_argument(
        "--ukf-alpha", 
        type=float,
        help="UKF alpha parameter (sigma point spread)"
    )
    parser.add_argument(
        "--ukf-beta", 
        type=float,
        help="UKF beta parameter (distribution shape)"
    )
    parser.add_argument(
        "--ukf-kappa", 
        type=float,
        help="UKF kappa parameter (secondary scaling)"
    )

    return parser


def build_config_from_args(args):
    """Build configuration dictionary from command line arguments."""
    
    # Start with base configuration
    if args.config:
        # Load from file if specified
        config = load_config_from_file(args.config)
        if not args.quiet:
            print(f"Loaded configuration from: {args.config}")
    else:
        # Use default base configuration
        config = get_base_config()

    # Override with command line arguments
    if args.filter:
        config["filter_type"] = args.filter

    if args.adaptive_process:
        config["adaptive_process_noise"] = True

    if args.adaptive_measurement:
        config["adaptive_measurement_noise"] = True

    if args.signal_max is not None:
        config["signal_max"] = args.signal_max

    if args.signal_decay is not None:
        config["signal_decay"] = args.signal_decay

    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    if args.seed is not None:
        config["random_seed"] = args.seed

    if args.robot_step_size is not None:
        config["robot_step_size"] = args.robot_step_size

    if args.actuator_noise is not None:
        config["actuator_noise"] = args.actuator_noise

    if args.target_motion_sigma is not None:
        config["target_motion_sigma"] = args.target_motion_sigma

    # UKF-specific parameters
    if args.ukf_alpha is not None:
        config["ukf_alpha"] = args.ukf_alpha

    if args.ukf_beta is not None:
        config["ukf_beta"] = args.ukf_beta

    if args.ukf_kappa is not None:
        config["ukf_kappa"] = args.ukf_kappa

    return config


def run_simulation(config, verbose=False):
    """Run a single simulation with the given configuration."""
    
    if verbose:
        print_config_summary(config, "Simulation Configuration")

    # Set random seed if specified
    if "random_seed" in config:
        np.random.seed(config["random_seed"])
        if verbose:
            print(f"Set random seed to: {config['random_seed']}")

    # Create environment
    if verbose:
        print("Creating simulation environment...")
    
    env = EKFEnvironment(config, verbose=verbose)

    # Run simulation
    if verbose:
        print("Starting simulation...")
    
    start_time = time.time()
    results = env.run_simulation()
    end_time = time.time()
    simulation_time = end_time - start_time

    # Print results summary
    steps_completed = results["steps_completed"]
    target_reached = results["target_reached"]
    
    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Filter type: {config['filter_type']}")
    print(f"Steps completed: {steps_completed:,}")
    print(f"Target reached: {'✓' if target_reached else '✗'}")
    print(f"Simulation time: {format_time_delta(simulation_time)}")
    
    if target_reached:
        print(f"Final robot position: [{results['final_robot_pos'][0]:.1f}, {results['final_robot_pos'][1]:.1f}]")
        print(f"Final target position: [{results['final_target_pos'][0]:.1f}, {results['final_target_pos'][1]:.1f}]")
    
    # Filter-specific statistics
    sigma_history = results.get("sigma_history", [])
    if sigma_history:
        print(f"Process noise σ_Q - mean: {np.mean(sigma_history):.3f}, final: {sigma_history[-1]:.3f}")
    
    R_est_history = results.get("R_est_history", [])
    if R_est_history and config.get("adaptive_measurement_noise", False):
        print(f"Measurement noise R - mean: {np.mean(R_est_history):.3f}, final: {R_est_history[-1]:.3f}")

    print(f"{'='*60}")

    return results


def create_visualizations(results, config, args):
    """Create and optionally save visualization plots."""
    
    if not args.quiet:
        print("Creating visualizations...")

    # Create visualizer
    visualizer = EKFVisualizer(results)

    # Ensure output directory exists
    if args.save_plots:
        ensure_output_directory(args.output_dir)

    # 1. Main trajectory plot with signal field
    fig1 = plt.figure(figsize=(12, 10))
    visualizer.plot_trajectory_with_signal_field(
        fig=fig1,
        show_belief_evolution=True,
        show_measurements=True
    )
    fig1.suptitle(f"{config['filter_type']} Target Tracking Simulation", fontsize=16)
    
    if args.save_plots:
        filepath = generate_default_output_path(
            f"trajectory_{config['filter_type']}", 
            include_timestamp=True
        )
        filepath = filepath.replace("output/", f"{args.output_dir}/")
        fig1.savefig(filepath, dpi=300, bbox_inches='tight')
        if not args.quiet:
            print(f"Saved trajectory plot: {filepath}")

    # 2. Filter statistics plot
    if results.get("sigma_history") or results.get("R_est_history"):
        fig2 = plt.figure(figsize=(15, 5))
        visualizer.plot_filter_statistics(fig=fig2)
        fig2.suptitle(f"{config['filter_type']} Filter Statistics", fontsize=16)
        
        if args.save_plots:
            filepath = generate_default_output_path(
                f"filter_stats_{config['filter_type']}", 
                include_timestamp=True
            )
            filepath = filepath.replace("output/", f"{args.output_dir}/")
            fig2.savefig(filepath, dpi=300, bbox_inches='tight')
            if not args.quiet:
                print(f"Saved filter statistics plot: {filepath}")

    # 3. Belief evolution plot (if target was reached quickly enough)
    if results["target_reached"] and results["steps_completed"] < 10000:
        fig3 = plt.figure(figsize=(10, 8))
        visualizer.plot_belief_evolution(
            fig=fig3,
            n_ellipses=min(10, results["steps_completed"] // 100)
        )
        fig3.suptitle(f"{config['filter_type']} Belief Evolution", fontsize=16)
        
        if args.save_plots:
            filepath = generate_default_output_path(
                f"belief_evolution_{config['filter_type']}", 
                include_timestamp=True
            )
            filepath = filepath.replace("output/", f"{args.output_dir}/")
            fig3.savefig(filepath, dpi=300, bbox_inches='tight')
            if not args.quiet:
                print(f"Saved belief evolution plot: {filepath}")

    # Show plots if not suppressed
    if not args.no_show_plots:
        plt.show()


def main():
    """Main entry point."""
    
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Build configuration
    try:
        config = build_config_from_args(args)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate configuration
    try:
        validate_filter_config(config)
    except Exception as e:
        print(f"Configuration validation error: {e}", file=sys.stderr)
        sys.exit(1)

    # Save configuration if requested
    if args.save_config:
        try:
            save_config_to_file(config, args.save_config)
            if not args.quiet:
                print(f"Saved configuration to: {args.save_config}")
        except Exception as e:
            print(f"Error saving configuration: {e}", file=sys.stderr)

    # Run simulation
    try:
        results = run_simulation(config, verbose=args.verbose)
    except Exception as e:
        print(f"Simulation error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create visualizations
    try:
        create_visualizations(results, config, args)
    except Exception as e:
        print(f"Visualization error: {e}", file=sys.stderr)
        # Don't exit on visualization errors

    if not args.quiet:
        print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()