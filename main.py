from __future__ import annotations
import os
import numpy as np
from PIL import Image
from typing import Tuple
from numba import njit


def load_and_prepare_image(path: str, width, height) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = Image.open(path).convert('L')
    arr = np.array(img)
    rows = np.where(arr.mean(axis=1) < 250)[0]
    cols = np.where(arr.mean(axis=0) < 250)[0]
    if len(rows) > 0 and len(cols) > 0:
        y0, y1 = rows[0], rows[-1]
        x0, x1 = cols[0], cols[-1]
        img = img.crop((x0, y0, x1 + 1, y1 + 1))
    img_resized = img.resize((width, height), Image.Resampling.BILINEAR)
    data = 1.0 - np.array(img_resized, dtype=np.float32) / 255.0
    mask = (data > 0.05).astype(np.float32)  # порог — подберите
    return data * mask

def create_velocity_field(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    xc, yc = 0.5, 0.5
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    x_full, y_full = np.meshgrid(x, y)
    u = -(y_full - yc)
    v = (x_full - xc)
    return u, v # соленоидальное поле

def compute_dt_from_courant(courant: float, u: np.ndarray, v: np.ndarray, dx: float, dy: float, safety: float = 0.999) -> float:
    # Проверка валидности ключа
    if not (0.0 < courant <= 1.0):
        raise ValueError("Courant (ключ) должен быть в интервале (0, 1].")
    max_u = np.max(np.abs(u))
    max_v = np.max(np.abs(v))
    denominator = (max_u / dx) + (max_v / dy)
    if denominator == 0.0:
        return 0.0
    dt = safety * courant / denominator
    return float(dt)

@njit(fastmath=True, cache=True)
def cabaret_init_fluxes_numba(q):
    ny, nx = q.shape
    phi_x = q.copy()
    phi_y = q.copy()
    for j in range(ny):
        jp = (j + 1) % ny
        for i in range(nx):
            ip = (i + 1) % nx
            phi_x[j, i] = 0.5 * (q[j, i] + q[j, ip])
            phi_y[j, i] = 0.5 * (q[j, i] + q[jp, i])
    return phi_x, phi_y



@njit(fastmath=True, cache=True)
def cabaret_step_full_numba(q, phi_x, phi_y, dt, dx, dy, u, v):
    ny, nx = q.shape
    q_half = q.copy()
    f = u * q
    g = v * q
    for j in range(ny):
        jm = (j - 1) % ny
        jp = (j + 1) % ny
        for i in range(nx):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            df = (f[j, ip] - f[j, im]) / (2.0 * dx)
            dg = (g[jp, i] - g[jm, i]) / (2.0 * dy)
            q_half[j, i] -= 0.5 * dt * (df + dg)
    phi_x_half = phi_x.copy()
    phi_y_half = phi_y.copy()
    for j in range(ny):
        jp = (j + 1) % ny
        for i in range(nx):
            ip = (i + 1) % nx
            phi_x_half[j, i] = 0.5 * (q_half[j, i] + q_half[j, ip])
            phi_y_half[j, i] = 0.5 * (q_half[j, i] + q_half[jp, i])
    phi_x_new = 2.0 * phi_x_half - phi_x
    phi_y_new = 2.0 * phi_y_half - phi_y
    u_face = u.copy()
    v_face = v.copy()
    for j in range(ny):
        for i in range(nx):
            ip = (i + 1) % nx
            u_face[j, i] = 0.5 * (u[j, i] + u[j, ip])
    for j in range(ny):
        jp = (j + 1) % ny
        for i in range(nx):
            v_face[j, i] = 0.5 * (v[j, i] + v[jp, i])

    F = u_face * phi_x_new
    G = v_face * phi_y_new

    q_new = q_half.copy()
    for j in range(ny):
        jm = (j - 1) % ny
        for i in range(nx):
            im = (i - 1) % nx
            q_new[j, i] -= 0.5 * dt * (
                (F[j, i] - F[j, im]) / dx +
                (G[j, i] - G[jm, i]) / dy
            )

    return q_new, phi_x_new, phi_y_new


def to_frame_video(arr: np.ndarray) -> np.ndarray:
    return (255.0 * (1.0 - np.clip(arr, 0.0, 1.0))).astype(np.uint8)

def generate_video(psi0, u, v, dt, dx, dy,
                   steps_per_pass, passes,
                   output_path, backward=False, perturb_fn=None, fps=100):

    psi = psi0.copy().astype(np.float32)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    phi_x, phi_y = cabaret_init_fluxes_numba(psi)
    phi_x = phi_x.astype(np.float32)
    phi_y = phi_y.astype(np.float32)

    frames = []

    pass_indices = range(passes) if not backward else reversed(range(passes))

    for pass_idx in pass_indices:
        u_mod, v_mod = perturb_fn(u, v, pass_idx) if perturb_fn else (u, v)
        u_mod = u_mod.astype(np.float32)
        v_mod = v_mod.astype(np.float32)

        for _ in range(steps_per_pass):
            step_dt = -dt if backward else dt
            # >>> правильный вызов <<<
            psi, phi_x, phi_y = cabaret_step_full_numba(
                psi, phi_x, phi_y, step_dt, dx, dy, u_mod, v_mod
            )
            frames.append(to_frame_video(psi))

    frames.append(to_frame_video(psi))

    import imageio
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def normalize_courant(courant: float) -> float:
    courant = float(courant)
    if courant <= 0.33:
        return courant
    if courant < 0.0 or courant > 1.0:
        raise ValueError("Courant (ключ) должен быть в интервале (0, 1].")
    # детерминированное "случайное" малое усиление на основе ключа
    rng = np.random.default_rng(seed=int(courant * 1e6))
    eps = 0.005 + 0.01 * rng.random()    # лежит примерно от 0.005 до 0.015
    return 0.32 + eps

