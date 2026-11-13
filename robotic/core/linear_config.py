"""
Additive config getters for the linear-Gaussian variant.  get_base_config() is reused
verbatim and then overlaid, so every existing key keeps its meaning.
"""

from .base_config import get_base_config


def get_linear_base_config(preset="robotic"):
    """Base config for the linear-Gaussian variant.

    preset="robotic"  -> robotic's kinematics (step 0.1, actuator 1.0, tol 3.0)
    preset="source_kf" -> source_kf's kinematics (step 1.0, actuator 0.0, tol 6.0)
    """
    cfg = get_base_config()
    cfg.update(
        {
            # --- what makes it linear-Gaussian --------------------------------
            "measurement_model": "linear_gradient",  # documentation only
            "linear_n_receptors": 8,
            "linear_cell_radius": 5.0,
            "linear_sigma_receptor": 1.0,
            "linear_shot_noise": True,
            "linear_kappa": None,  # None -> auto: signal_max / span^2
            "linear_signal_channel": "mean",  # scalar channel the Coupled arm uses
            "linear_R_mode": "frozen",  # "frozen" (as EKF) | "oracle"
            "linear_measurement_mode": "vector",  # "vector" (H=I) | "radial" (rank-1)
            "linear_range_decay": 0.0,  # ablation: exp-decay of position-space info
            "adaptive_R_use_prefit": False,  # mirror the EKF's post-fit residual
            # --- the log-normal field has no analogue here; off by default -----
            "spatial_correlation_length": 0.0,
            # --- keep runs cheap; linear runs are ~1e3 steps, not ~1e5 ---------
            "max_steps": 200000,
        }
    )
    if preset == "source_kf":
        cfg.update(
            {
                "robot_step_size": 1.0,
                "actuator_noise": 0.0,
                "distance_tolerance": 6.0,
                "target_motion_sigma": 0.5,
                "signal_max": 20.0,
                "initial_belief_mean": [100.0, 100.0],
                "initial_belief_variance": 10000.0,
                "baseline_process_noise": 0.5,  # source_kf uses Q = sigma_source^2
                "max_steps": 20000,
            }
        )
    return cfg


def get_linear_method_configs():
    """Same three arms as the nonlinear study, so the plots line up."""
    return {
        "Standard EKF": {
            "adaptive_process_noise": False,
            "adaptive_measurement_noise": False,
        },
        "Coupled EKF": {
            "adaptive_process_noise": True,
            "adaptive_measurement_noise": False,
        },
        "Adaptive EKF": {
            "adaptive_process_noise": False,
            "adaptive_measurement_noise": True,
        },
    }


def get_linear_study_config():
    return {
        "n_runs": 200,
        "signal_max_values": [10, 15, 20, 25, 30, 35, 40, 45, 50],
    }
