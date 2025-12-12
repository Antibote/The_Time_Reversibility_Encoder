import argparse
import numpy as np
from PIL import Image
from main import load_and_prepare_image, create_velocity_field, cabaret_step_full_numba, compute_dt_from_courant, generate_video, \
    normalize_courant, cabaret_init_fluxes_numba



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='Файл, который будем шифровать')
    p.add_argument('--courant', type=float, required=True, help='Ключ (0 < C ≤ 1)')
    p.add_argument('--frames_fwd', type=int, default=10, help='Количество кадров')
    p.add_argument('--steps_per_frame', type=int, default=5, help='Количество шагов на один кадр')
    p.add_argument('--factor', type=int, default=10, help='Увеличивает общее количество шагов схемы без изменения количества кадров в анимации')
    p.add_argument('--passes', type=int, default=5, help='Число проходов для усиления шифрования')
    p.add_argument('--out', default='encrypted_image.npz')
    p.add_argument('--png', default='encrypted_image.png')
    return p.parse_args()

def auto_size(input_path: str) -> tuple[int, int]:
    img = Image.open(input_path).convert("RGB")
    weight, height = img.size

    if abs(weight - height) < 5:
        return 128, 128

    if weight > height:
        return 256, 128
    else:
        return 128, 256

def perturb_velocity(u: np.ndarray, v: np.ndarray, pass_idx: int, master_key: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=int(master_key*1e6 + pass_idx))
    factor = 0.05  # 5% perturbation
    u_mod = u + factor * (rng.random(u.shape) - 0.5)
    v_mod = v + factor * (rng.random(v.shape) - 0.5)
    return u_mod, v_mod

def encrypt():
    args = parse_args()
    width, height = auto_size(args.image)
    psi0 = load_and_prepare_image(args.image, width=width, height=height)
    ny, nx = psi0.shape
    u, v = create_velocity_field(nx, ny)
    dx = 1.0 / nx; dy = 1.0 / ny
    total_steps = args.frames_fwd * args.factor * args.steps_per_frame
    steps_per_pass = total_steps // args.passes
    courant = normalize_courant(args.courant)
    psi = psi0.copy()
    dt = compute_dt_from_courant(courant, u, v, dx, dy)  # фиксированный dt для всех проходов

    for pass_idx in range(args.passes):
        u_mod, v_mod = perturb_velocity(u, v, pass_idx, courant)
        phi_x, phi_y = cabaret_init_fluxes_numba(psi)

        # Главный цикл
        for _ in range(steps_per_pass):
            psi, phi_x, phi_y = cabaret_step_full_numba(
                psi, phi_x, phi_y, dt, dx, dy, u_mod, v_mod)

    # Сохраняем точный массив и метаданные
    np.savez(args.out,
             psi=psi,
             phi_x=phi_x,
             phi_y=phi_y,
             width=width,
             height=height,
             courant=courant,
             frames_fwd=args.frames_fwd,
             steps_per_frame=args.steps_per_frame,
             factor=args.factor,
             passes=args.passes)

    # PNG визуализация
    img_enc = (255.0 * (1.0 - np.clip(psi, 0.0, 1.0))).astype(np.uint8)
    Image.fromarray(img_enc).save(args.png)

    # Генерация видео
    generate_video(
        psi0, u, v, dt, dx, dy,
        steps_per_pass=steps_per_pass,
        passes=args.passes,
        output_path="encrypted_video.mp4",
        backward=False,
        perturb_fn=lambda u, v, idx: perturb_velocity(u, v, idx, args.courant),
        fps=100
    )

    print("Закодированная картинка создана и называется:", args.png)
    print("Видео с кодирование создано и называется:", "encrypted_video.mp4")


if __name__ == '__main__':
    encrypt()
