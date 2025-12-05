import argparse
import numpy as np
from PIL import Image
from main import load_and_prepare_image, create_velocity_field, rk4_step, generate_gif


def parse_args():
    parser = argparse.ArgumentParser(description="Дешифратор изображения по схеме КАБАРЭ.")
    parser.add_argument('--image', type=str, required=True, help='Путь к зашифрованному изображению.')
    parser.add_argument('--size', type=int, default=120, help='Размер сетки по каждой координате.')
    parser.add_argument('--courant', type=float, default=0.5, help='Число Куранта.')
    parser.add_argument('--frames_bwd', type=int, default=20, help='Количество кадров раскручивания.')
    parser.add_argument('--steps_per_frame', type=int, default=5, help='Количество шагов на каждый кадр.')
    parser.add_argument('--factor', type=int, default=10, help='Фактор для увеличения количества кадров.')
    parser.add_argument('--output_gif', type=str, default='decrypted_image.gif', help='Путь для сохранения GIF.')
    parser.add_argument('--output_image', type=str, default='decrypted_image.png', help='Путь для сохранения расшифрованного изображения.')
    return parser.parse_args()


def decrypt_image():
    args = parse_args()
    psi0 = load_and_prepare_image(args.image, size=args.size)
    ny, nx = psi0.shape
    u, v = create_velocity_field(nx, ny)

    # Генерация GIF с раскручиванием (до полного расшифрования)
    generate_gif(psi0, u, v, args.courant, 0, args.frames_bwd * args.factor,  # Без шифрования вперед
                 args.steps_per_frame, args.output_gif)

    # Расшифровка изображения
    decrypted_image = psi0.copy()
    dt = args.courant / max(nx, ny)  # Используем тот же шаг времени
    for _ in range(args.frames_bwd * args.factor):  # Процесс раскручивания
        for _ in range(args.steps_per_frame):
            decrypted_image = rk4_step(decrypted_image, -dt, 1.0 / nx, 1.0 / ny, u, v)

    # Сохраняем расшифрованное изображение
    img = Image.fromarray(((1 - np.clip(decrypted_image, 0, 1)) * 255).astype(np.uint8))
    img.save(args.output_image)
    print(f"Расшифрованное изображение сохранено в файл: {args.output_image}")


if __name__ == "__main__":
    decrypt_image()
