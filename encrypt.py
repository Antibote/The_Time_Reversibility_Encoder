import argparse
import numpy as np
from PIL import Image
from main import load_and_prepare_image, create_velocity_field, rk4_step, generate_gif


def parse_args():
    parser = argparse.ArgumentParser(description="Шифратор изображения с помощью схемы КАБАРЭ.")
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению для шифрования.')
    parser.add_argument('--size', type=int, default=120, help='Размер сетки по каждой координате.')
    parser.add_argument('--courant', type=float, default=0.5, help='Число Куранта.')
    parser.add_argument('--frames_fwd', type=int, default=20, help='Количество кадров закручивания.')
    parser.add_argument('--steps_per_frame', type=int, default=5, help='Количество шагов на каждый кадр.')
    parser.add_argument('--factor', type=int, default=10, help='Фактор для увеличения количества кадров.')
    parser.add_argument('--output_gif', type=str, default='encrypted_image.gif', help='Путь для сохранения GIF.')
    parser.add_argument('--output_image', type=str, default='encrypted_image.png', help='Путь для сохранения зашифрованного изображения.')
    return parser.parse_args()


def encrypt_image():
    args = parse_args()
    psi0 = load_and_prepare_image(args.image, size=args.size)
    ny, nx = psi0.shape
    u, v = create_velocity_field(nx, ny)

    # Генерация GIF с закручиванием (до полного шифрования)
    generate_gif(psi0, u, v, args.courant, args.frames_fwd * args.factor, 0,  # Без обратного раскручивания
                 args.steps_per_frame, args.output_gif)

    # Шифрование изображения с использованием схемы КАБАРЭ
    dt = args.courant / max(nx, ny)  # Вычисляем шаг по времени
    encrypted_image = psi0.copy()
    for _ in range(args.frames_fwd * args.factor):  # Процесс закручивания
        for _ in range(args.steps_per_frame):
            encrypted_image = rk4_step(encrypted_image, dt, 1.0 / nx, 1.0 / ny, u, v)

    # Сохраняем зашифрованное изображение
    img = Image.fromarray(((1 - np.clip(encrypted_image, 0, 1)) * 255).astype(np.uint8))
    img.save(args.output_image)
    print(f"Зашифрованное изображение сохранено в файл: {args.output_image}")


if __name__ == "__main__":
    encrypt_image()
