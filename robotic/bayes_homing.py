import numpy as np
from numpy.random import normal
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class BayesianHoming:
    def __init__(self, config=None):
        """
        Initialize the Bayesian filter for homing behavior

        Args:
            config (dict): Configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            "grid_size": 100,
            "target_pos": None,
            "true_motion_sigma": 0.5,  # Actual noise in robot motion
            "min_motion_sigma": 0.2,  # Minimum uncertainty in motion model (D_base)
            "max_motion_sigma": 2.0,  # Maximum uncertainty in motion model (D_max)
            "motion_decay_rate": 0.05,  # How quickly uncertainty decreases with signal (k)
            "measurement_noise_factor": 1,
            "signal_strength_max": 100,
            "signal_decay_exp": 2,
            "movement_step_size": 1.0,
            "kernel_size": 5,
            "target_reach_threshold": 2.0,
        }

        # Update with user configuration if provided
        if config is not None:
            self.config.update(config)

        # Initialize belief grid (uniform prior)
        self.grid_size = self.config["grid_size"]
        self.belief = np.ones((self.grid_size, self.grid_size)) / (
            self.grid_size * self.grid_size
        )

        # Set target position
        if self.config["target_pos"] is None:
            # set to 4/5 of grid size if not provided
            self.target_pos = (self.grid_size * 4 // 5, self.grid_size * 4 // 5)
            self.config["target_pos"] = self.target_pos
        else:
            self.target_pos = self.config["target_pos"]

        # Calculate expected signal strength for all positions
        self.expected_signal_strength = self.compute_signal_strength_everywhere(
            self.target_pos
        )

    def compute_signal_strength_everywhere(self, pos):
        """Compute expected signal strength for all positions relative to the target."""
        expected_signal_strength = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Use the target position here (pos should be self.target_pos)
                distance = max(0.1, np.sqrt((i - pos[0]) ** 2 + (j - pos[1]) ** 2))
                expected_signal_strength[i, j] = self.config["signal_strength_max"] / (
                    distance ** self.config["signal_decay_exp"]
                )
        return expected_signal_strength

    def get_adaptive_motion_sigma(self, signal_strength):
        """
        Calculate motion model uncertainty based on signal strength
        D(s) = D_base + (D_max - D_base) * exp(-k * s)
        where:
        - D_base is minimum uncertainty (when signal is strong)
        - D_max is maximum uncertainty (when signal is weak)
        - k controls how quickly uncertainty decreases with signal strength
        - s is the signal strength

        Args:
            signal_strength (float): Current signal strength measurement

        Returns:
            float: Adaptive motion sigma value
        """
        D_base = self.config["min_motion_sigma"]
        D_max = self.config["max_motion_sigma"]
        k = self.config["motion_decay_rate"]

        return D_base + (D_max - D_base) * np.exp(-k * signal_strength)

    def calculate_expected_signal(self, pos):
        """Calculate the signal strength at a given position"""
        distance = np.sqrt(
            (pos[0] - self.target_pos[0]) ** 2 + (pos[1] - self.target_pos[1]) ** 2
        )

        # Avoid division by zero
        distance = max(distance, 0.1)

        # Signal strength decreases with distance according to configured decay
        signal = self.config["signal_strength_max"] / (
            distance ** self.config["signal_decay_exp"]
        )
        return signal

    def get_signal_strength(self, pos):
        """Calculate the signal strength at a given position"""
        signal = self.calculate_expected_signal(pos)

        # Add noise proportional to signal strength
        noise = normal(0, np.sqrt(self.config["measurement_noise_factor"]))
        return max(0, signal + noise)

    def motion_update(self, action, current_signal):
        """
        Update the belief based solely on the control action and adaptive motion uncertainty.
        This should propagate the belief without reference to the true position.
        """
        # Instead of using old_pos, shift the belief according to the intended action.
        # For example, you can approximate the shift (ensure action is scaled to grid indices).
        dx, dy = action
        # For simplicity, round the shifts to nearest integer grid cells.
        shift_x = int(round(dx))
        shift_y = int(round(dy))

        # Shift the belief grid using np.roll:
        shifted_belief = np.roll(self.belief, shift_x, axis=0)
        shifted_belief = np.roll(shifted_belief, shift_y, axis=1)

        # Get adaptive motion sigma based on the measurement (current_signal)
        adaptive_sigma = self.get_adaptive_motion_sigma(current_signal)

        # Create the convolution kernel (as before)
        kernel_size = self.config["kernel_size"]
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                kernel[i, j] = np.exp(-(dist**2) / (2 * adaptive_sigma**2))
        kernel = kernel / kernel.sum()

        # Convolve the shifted belief to account for motion uncertainty
        self.belief = convolve2d(shifted_belief, kernel, mode="same")
        self.belief = self.belief / np.sum(self.belief)  # Normalize

        return adaptive_sigma

    def measurement_update(self, measurement):
        """
        Update belief based on the sensor measurement.
        Uses a vectorized computation: compares the measurement with the precomputed expected signal.
        """
        # self.expected_signal_strength is a grid computed once based on the target position.
        likelihood = np.exp(
            -((measurement - self.expected_signal_strength) ** 2)
            / (2 * self.config["measurement_noise_factor"])
        )

        # Bayes update: multiply prior by likelihood and normalize.
        self.belief = self.belief * likelihood
        self.belief = self.belief / np.sum(self.belief)

    def get_next_action(self):
        """Determine next movement based solely on the belief estimate (without true position info)"""
        # Use the belief’s maximum likelihood position as the estimated state
        estimated_pos = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        # Compute the vector from the estimated position to the known target
        dx = self.target_pos[0] - estimated_pos[0]
        dy = self.target_pos[1] - estimated_pos[1]
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx = (dx / magnitude) * self.config["movement_step_size"]
            dy = (dy / magnitude) * self.config["movement_step_size"]
        return (dx, dy)

    def update_true_state(self, true_pos, action):
        """
        Update the true state (position) of the robot based on the intended action.
        The update uses a noisy motion model where the noise is added to the angle.

        Args:
            true_pos (tuple or np.array): The current true position as (x, y).
            action (tuple or np.array): The intended movement vector (dx, dy).

        Returns:
            tuple: The new true position (new_x, new_y) after applying the noisy action.
        """

        # Compute the intended angle and step size from the action
        angle = np.arctan2(action[1], action[0])

        # Add noise to the angle
        noisy_angle = angle + normal(0, self.config["true_motion_sigma"])

        # Compute the actual displacement using the noisy angle (step size remains the same)
        dx = self.config["movement_step_size"] * np.cos(noisy_angle)
        dy = self.config["movement_step_size"] * np.sin(noisy_angle)

        # Update the true position and clip to stay within grid boundaries
        new_x = np.clip(true_pos[0] + dx, 0, self.grid_size - 1)
        new_y = np.clip(true_pos[1] + dy, 0, self.grid_size - 1)

        return (new_x, new_y)


def run_simulation(config=None, steps=50):
    """
    Run a simulation with the given configuration

    Args:
        config (dict): Configuration dictionary for BayesianHoming
        steps (int): Maximum number of steps to simulate
    """
    # Initialize environment and robot
    env = BayesianHoming(config)

    print("Target position:", env.target_pos)
    # Set start position (20% of grid size)
    start_pos = (env.grid_size // 5, env.grid_size // 5)
    true_pos = start_pos

    # Store trajectory and adaptive sigmas for visualization
    trajectory = [(true_pos[0], true_pos[1])]
    adaptive_sigmas = []

    for step in range(steps):
        # Compute sensor measurement using the hidden true state
        signal = env.get_signal_strength(true_pos)

        # Update the belief with the measurement
        env.measurement_update(signal)

        # Get next action based solely on the filter’s internal belief
        action = env.get_next_action()

        # Update the filter’s belief with the control action
        sigma = env.motion_update(action, signal)

        # Get the adaptive sigma value for visualization
        adaptive_sigmas.append(sigma)

        # Update the true position based on the noisy action
        true_pos = env.update_true_state(true_pos, action)

        # Store position
        trajectory.append((true_pos[0], true_pos[1]))

        # Check if target reached
        if (
            np.sqrt(
                (true_pos[0] - env.target_pos[0]) ** 2
                + (true_pos[1] - env.target_pos[1]) ** 2
            )
            < env.config["target_reach_threshold"]
        ):
            print(f"Target reached in {len(trajectory)} steps!")
            break
        elif step % 100 == 0:
            print(f"Step {step} of {steps}")
            print(f"True position: {true_pos}")

    if step == steps - 1:
        print(f"Simulation ended after maximum {steps} steps")
        pass

    return trajectory, adaptive_sigmas, env


if __name__ == "__main__":
    # set random seed for reproducibility
    np.random.seed(82)

    # Example usage with custom configuration
    example_config = {
        "grid_size": 4000,
        "true_motion_sigma": 0.5,
        "max_motion_sigma": 0.5,
        "min_motion_sigma": 1e-5,
        "motion_decay_rate": 2,
        "measurement_noise_factor": 1e-4,
        "signal_strength_max": 0.5,
        "signal_decay_exp": 0.2,
        "movement_step_size": 1,
        "kernel_size": 5,
        "target_reach_threshold": 1.5,
    }

    # Run and visualize simulation
    trajectory, adaptive_sigmas, env = run_simulation(config=example_config, steps=10)
    trajectory = np.array(trajectory)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot signal map and trajectory
    im = ax1.imshow(env.expected_signal_strength, cmap="hot", interpolation="nearest")
    ax1.plot(trajectory[:, 1], trajectory[:, 0], "b-", label="Robot Path")
    ax1.plot(env.target_pos[1], env.target_pos[0], "g*", markersize=15, label="Target")
    ax1.plot(trajectory[0, 1], trajectory[0, 0], "bs", label="Start")
    plt.colorbar(im, ax=ax1, label="Expected Signal Strength")
    ax1.legend()

    # Plot adaptive sigma over time
    ax2.plot(adaptive_sigmas, "r-")
    ax2.set_title("Adaptive Motion Sigma over Time")
    ax2.set_xlabel("Step Number")
    ax2.set_ylabel("Motion Sigma")
    ax2.grid(True)

    plt.tight_layout()

    # # Save plot
    # plt.savefig(
    #     "bayes_homing_example_nonadaptive.svg",
    #     format="svg",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    plt.show()

    # Print final configuration used
    print("\nConfiguration used:")
    for key, value in env.config.items():
        print(f"{key}: {value}")
