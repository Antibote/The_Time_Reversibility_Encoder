import argparse
import numpy as np
from PIL import Image
from main import create_velocity_field, cabaret_step_full_numba, compute_dt_from_courant, generate_video, normalize_courant


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--infile', required=True, help='Файл .npz, созданный encrypt.py')
    p.add_argument('--courant', type=float, required=True, help='ключ (0 < C ≤ 1)')
    p.add_argument('--gif', default='decrypted_image.gif')
    p.add_argument('--outpng', default='decrypted_image.png')
    return p.parse_args()

def perturb_velocity(u: np.ndarray, v: np.ndarray, pass_idx: int, master_key: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=int(master_key*1e6 + pass_idx))
    factor = 0.05
    u_mod = u + factor * (rng.random(u.shape) - 0.5)
    v_mod = v + factor * (rng.random(v.shape) - 0.5)
    return u_mod, v_mod

def decrypt():
    args = parse_args()
    data = np.load(args.infile, allow_pickle=False)
    psi_enc = data['psi']
    frames_fwd = int(data['frames_fwd'])
    steps_per_frame = int(data['steps_per_frame'])
    factor = int(data['factor'])
    passes = int(data['passes'])
    width = int(data['width'])
    height = int(data['height'])

    ny, nx = psi_enc.shape
    u, v = create_velocity_field(nx, ny)
    dx = 1.0 / nx; dy = 1.0 / ny
    courant = normalize_courant(args.courant)
    total_steps = frames_fwd * factor * steps_per_frame
    steps_per_pass = total_steps // passes
    dt = compute_dt_from_courant(courant, u, v, dx, dy)

    psi = psi_enc.copy().astype(np.float32)
    phi_x = data['phi_x'].astype(np.float32)
    phi_y = data['phi_y'].astype(np.float32)

    # Обратный порядок
    for pass_idx in reversed(range(passes)):
        u_mod, v_mod = perturb_velocity(u, v, pass_idx, courant)
        u_mod = u_mod.astype(np.float32)
        v_mod = v_mod.astype(np.float32)
        for _ in range(steps_per_pass):
            psi, phi_x, phi_y = cabaret_step_full_numba(
                psi, phi_x, phi_y, -dt, dx, dy, u_mod, v_mod)

    # PNG визуализация
    img_dec = (255.0 * (1.0 - np.clip(psi, 0.0, 1.0))).astype(np.uint8)
    Image.fromarray(img_dec).save(args.outpng)

    # GIF (обратный процесс)

    generate_video(
        psi_enc, u, v, dt, dx, dy,
        steps_per_pass=steps_per_pass,
        passes=passes,
        output_path="decrypted_video.mp4",
        backward=True,
        perturb_fn=lambda u, v, idx: perturb_velocity(u, v, idx, courant),
        fps=100
    )

    print("Расшифрованная картинка называется:", args.outpng)
    print("Видео расшифровки называется:", args.gif)
    return psi

if __name__ == '__main__':
    decrypt()
