import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


class BayesianNavigation:
    def __init__(self, config=None):
        self.config = {
            "grid_size": 100,
            "initial_belief": None,
            "true_target_pos": None,
            "true_motion_sigma": 0.5,  # Actual noise in robot motion
            "min_motion_sigma": 0.1,  # Minimum uncertainty in motion model (D_base)
            "max_motion_sigma": 0.5,  # Maximum uncertainty in motion model (D_max)
            "adaptive_rate": 8,  # How quickly uncertainty decreases with signal (k)
            "measurement_noise_factor": 1e-4,
            "signal_max": 1,
            "signal_decay": 0.3,
            "movement_step_size": 1.0,
            "kernel_size": 5,
            "target_reach_threshold": 2.0,
        }
        if config is not None:
            self.config.update(config)

        self.grid_size = self.config["grid_size"]
        if self.config["initial_belief"] is None:
            self.belief = np.ones((self.grid_size, self.grid_size)) / (
                self.grid_size * self.grid_size
            )
        else:
            self.belief = self.config["initial_belief"]

        if self.config["true_target_pos"] is None:
            self.true_target_pos = (self.grid_size * 4 // 5, self.grid_size * 4 // 5)
            self.config["true_target_pos"] = self.true_target_pos
        else:
            self.true_target_pos = self.config["true_target_pos"]

        # Precompute kernel for motion update
        kernel_size = self.config["kernel_size"]
        center = kernel_size // 2
        y, x = np.ogrid[:kernel_size, :kernel_size]
        self.kernel_mat = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    def get_expected_signal(self, distance):
        return self.config["signal_max"] * np.exp(
            -self.config["signal_decay"] * distance
        )

    def compute_all_expected_signal(self, target_pos):
        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        distance = np.sqrt((xx - target_pos[0]) ** 2 + (yy - target_pos[1]) ** 2)
        distance = np.maximum(distance, 1e-8)  # Avoid division by zero
        signal = self.get_expected_signal(distance)
        return signal

    def get_noisy_measurement(self, pos, target_pos):
        distance = np.sqrt(
            (pos[0] - target_pos[0]) ** 2 + (pos[1] - target_pos[1]) ** 2
        )
        expected_signal = self.get_expected_signal(distance)
        noise = max(0, normal(0, self.config["measurement_noise_factor"]))
        return expected_signal + noise

    def measurement_update(self, measurement, robot_pos):
        # For each possible target position, compute expected signal at robot position
        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        # Distance from each possible target position to robot position
        distance = np.sqrt((xx - robot_pos[0]) ** 2 + (yy - robot_pos[1]) ** 2)
        distance = np.maximum(distance, 0.1)  # Avoid division by zero
        expected_measurement = self.get_expected_signal(distance)

        likelihood = np.exp(
            -((measurement - expected_measurement) ** 2)
            / (2 * self.config["measurement_noise_factor"] ** 2)
        )
        self.belief *= likelihood
        self.belief /= np.sum(self.belief)

    def get_adaptive_motion_sigma(self, signal_strength):
        """
        Calculate motion model uncertainty based on signal strength
        D(s) = D_base + (D_max - D_base) * exp(-k * s)
        """
        D_base = self.config["min_motion_sigma"]
        D_max = self.config["max_motion_sigma"]
        k = self.config["adaptive_rate"]

        return D_base + (D_max - D_base) * np.exp(-k * signal_strength)

    def motion_update(self, current_signal):
        adaptive_sigma = self.get_adaptive_motion_sigma(current_signal)
        kernel = np.exp(-(self.kernel_mat**2) / (2 * adaptive_sigma**2))
        kernel /= kernel.sum()
        self.belief = convolve2d(
            self.belief, kernel, mode="same", boundary="fill", fillvalue=0
        )
        self.belief /= np.sum(self.belief)
        return adaptive_sigma

    def get_next_intended_action(self, robot_pos):
        """get unnormalized step vector"""
        estimated_target_pos = np.unravel_index(
            np.argmax(self.belief), self.belief.shape
        )
        dx = estimated_target_pos[0] - robot_pos[0]
        dy = estimated_target_pos[1] - robot_pos[1]
        return dx, dy

    def update_position(self, true_pos, action):
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


def run_navigation_simulation(config=None, steps=100):
    env = BayesianNavigation(config)
    print("True target position:", env.true_target_pos)

    robot_pos = (env.grid_size // 5, env.grid_size // 5)
    trajectory = [robot_pos]
    sigmas = [env.config["max_motion_sigma"]]

    for step in range(steps):
        measurement = env.get_noisy_measurement(robot_pos, env.true_target_pos)
        env.measurement_update(measurement, robot_pos)
        sigma = env.motion_update(measurement)
        action = env.get_next_intended_action(robot_pos)
        robot_pos = env.update_position(robot_pos, action)
        trajectory.append(robot_pos)
        sigmas.append(sigma)

        distance_to_target = np.linalg.norm(
            np.array(robot_pos) - np.array(env.true_target_pos)
        )
        if distance_to_target < env.config["target_reach_threshold"]:
            print(f"Target reached in {len(trajectory)} steps!")
            break

    return trajectory, env, sigmas


if __name__ == "__main__":
    np.random.seed(2)
    example_config = {
        "grid_size": 100,
        "true_motion_sigma": 0.5,
        "min_motion_sigma": 0.5,
        "max_motion_sigma": 0.5,
        "adaptive_rate": 0.8,  # Irrelevant when min == max
        "measurement_noise_factor": 0.06,
        "signal_max": 0.3,
        "signal_decay": 0.2,
        "movement_step_size": 0.2,
        "kernel_size": 5,
        "target_reach_threshold": 5.0,
    }
    trajectory, env, sigmas = run_navigation_simulation(
        config=example_config, steps=500000
    )

    trajectory = np.array(trajectory)
    sigmas = np.array(sigmas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Plot the belief map and trajectory
    signal_map = env.compute_all_expected_signal(env.true_target_pos)
    ax1.imshow(signal_map, cmap="hot", interpolation="nearest")

    ax1.plot(trajectory[:, 1], trajectory[:, 0], "b-", label="Robot Path")
    ax1.plot(
        env.true_target_pos[1],
        env.true_target_pos[0],
        "g*",
        markersize=15,
        label="Target",
    )
    ax1.plot(trajectory[0, 1], trajectory[0, 0], "bs", label="Start")
    ax1.legend()
    ax1.set_title("Belief Map and Trajectory")

    # Plot the sigmas over time
    ax2.plot(sigmas, "r-", label="Sigma")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Sigma")
    ax2.legend()
    ax2.set_title("Sigmas Over Time")

    plt.tight_layout()
    plt.show()
