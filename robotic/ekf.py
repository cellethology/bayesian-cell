import numpy as np

# ------------ main simulation loop (sketch) -------------
T = 1000000  # number of steps
mu, Sigma = np.array([100.0, 100.0]), 100 * np.eye(2)  # broad prior
r_pos = np.array([80.0, 80.0])  # robot start
tgt_pos = np.array([120.0, 120.0])  # true target
r_pos_trace = []
tgt_pos_trace = []

# ------------ parameters -------------
arena_lo, arena_hi = 0.0, 200.0
dist_tol = 2.0

c0 = 40.0  # max signal
lam = 0.03  # decay exponent
sigma_Q = 0.5  # baseline target step std
step_sz = 0.5  # robot step per tick
sigma_u = 0.2  # actuator noise std
alpha_R = 0.01  # innovation-based measurement noise update rate
is_adaptive = False

# set seed
np.random.seed(55)


# ------------ helper functions -------------
def h(mu, r):
    """expected signal at robot pose r for target mean mu"""
    d = np.linalg.norm(mu - r)
    return c0 * np.exp(-lam * d)


def jacobian_h(mu, r):
    d = np.linalg.norm(mu - r) + 1e-9  # avoid /0
    h_val = h(mu, r)
    return (-lam * h_val * (mu - r) / d).reshape(1, 2)  # 1×2


# constant measurement variance chosen from initial geometry
R_est = h(mu, r_pos)


def ekf_step(mu, Sigma, z, r, R_est, adaptive=False):
    """one EKF iteration, returns updated (mu, Sigma)"""
    # --- 1. adaptive or fixed process noise ---
    Q = (sigma_Q**2 / (1 + z) if adaptive else sigma_Q**2) * np.eye(2)

    # --- 2. prediction ---
    mu_pred = mu
    Sigma_pred = Sigma + Q

    # --- 3. linearise measurement ---
    h_pred = h(mu_pred, r)
    H = jacobian_h(mu_pred, r)
    R = R_est

    # --- 4. innovation & Kalman gain ---
    y = z - h_pred
    S = (H @ Sigma_pred @ H.T + R).item() + 1e-12  # scalar with numerical stability
    K = (Sigma_pred @ H.T / S).reshape(
        2,
    )

    # --- 5. update ---
    mu_upd = mu_pred + K * y
    Sigma_upd = (np.eye(2) - np.outer(K, H)).dot(Sigma_pred)

    # numerical hygiene
    Sigma_upd = 0.5 * (Sigma_upd + Sigma_upd.T)

    # --- 6. adaptive noise updates (innovation‑based) ---
    # measurement‑noise: update with exponential moving average of innovation²
    R_est_new = (1 - alpha_R) * R_est + alpha_R * (y**2)

    return mu_upd, Sigma_upd, R_est_new


R_now = R_est

for t in range(T):
    # 1. robot senses
    lam_true = h(tgt_pos, r_pos)
    z_t = np.random.poisson(lam_true)

    # 2. EKF update
    mu, Sigma, R_now = ekf_step(mu, Sigma, z_t, r_pos, R_now, adaptive=is_adaptive)

    # 3. robot moves toward target estimate
    vec = mu - r_pos
    if np.linalg.norm(vec) > 1e-6:
        r_pos += step_sz * vec / np.linalg.norm(vec)
    r_pos += np.random.randn(2) * sigma_u
    r_pos = np.clip(r_pos, arena_lo, arena_hi)

    # 4. target random walk
    tgt_pos += np.random.randn(2) * sigma_Q
    tgt_pos = np.clip(tgt_pos, arena_lo, arena_hi)

    # track r_pos and tgt_pos
    r_pos_trace.append(r_pos.copy())
    tgt_pos_trace.append(tgt_pos.copy())

    # 5. check if r_pos is close to tgt_pos
    if np.linalg.norm(r_pos - tgt_pos) < dist_tol:
        print(f"Target found at {tgt_pos} at time {t}")
        break
