"""
Минимальный пример генерации GIF‑анимации закручивания и
раскручивания изображения. Используется бездиффузионная численная
схема на центральных разностях с интегрированием методом
Рунге–Кутты 4‑го порядка (вариант улучшенной схемы КАБАРЭ). Число
Куранта C определяет шаг по времени, а параметр ``factor``
масштабирует количество кадров для более длительного закручивания.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
from PIL import Image


def load_and_prepare_image(path: str, size: int = 120) -> np.ndarray:
    """
    Загружает изображение, обрезает белые поля, масштабирует до
    квадрата ``size×size`` и инвертирует яркость в диапазон [0, 1] (белое→1,
    чёрное→0). Этот предварительный шаг не имеет отношения к схеме
    КАБАРЭ и служит только для подготовки данных.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл '{path}' не найден")
    img = Image.open(path).convert('L')
    arr = np.array(img)
    rows = np.where(arr.mean(axis=1) < 250)[0]
    cols = np.where(arr.mean(axis=0) < 250)[0]
    if len(rows) > 0 and len(cols) > 0:
        y0, y1 = rows[0], rows[-1]
        x0, x1 = cols[0], cols[-1]
        img = img.crop((x0, y0, x1 + 1, y1 + 1))
    img_resized = img.resize((size, size), Image.BILINEAR)
    data = 1.0 - np.array(img_resized, dtype=np.float64) / 255.0
    return data


def create_velocity_field(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает поле скоростей ``(u, v)``, описывающее вращение вокруг
    центра области (0.5, 0.5). Этот вихрь определяет характеристики в
    уравнении переноса и используется в схеме КАБАРЭ.
    """
    xc, yc = 0.5, 0.5
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    u = -(Y - yc)
    v = (X - xc)
    return u, v


def compute_rhs(psi: np.ndarray, u: np.ndarray, v: np.ndarray,
                dx: float, dy: float) -> np.ndarray:
    """
    Вычисляет правую часть уравнения переноса
    \(\partial\_t\psi + u\,\partial\_x\psi + v\,\partial\_y\psi = 0\).
    Здесь используется центральная разность второго порядка, как в
    схеме КАБАРЭ, для аппроксимации пространственных производных.
    """
    dpsi_dx = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * dx)
    dpsi_dy = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * dy)
    return -(u * dpsi_dx + v * dpsi_dy)


def rk4_step(psi: np.ndarray, dt: float, dx: float, dy: float,
              u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Выполняет один шаг метода Рунге–Кутты 4‑го порядка. Такая схема
    служит антидиссипативным аналогом КАБАРЭ: RK4 минимизирует
    численные искажения при интегрировании во времени.
    """
    k1 = compute_rhs(psi, u, v, dx, dy)
    k2 = compute_rhs(psi + 0.5 * dt * k1, u, v, dx, dy)
    k3 = compute_rhs(psi + 0.5 * dt * k2, u, v, dx, dy)
    k4 = compute_rhs(psi + dt * k3, u, v, dx, dy)
    return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def to_frame(arr: np.ndarray, scale: int = 256) -> Image.Image:
    """
    Преобразует массив ψ в изображение для использования в анимации.
    Масштабирование и инверсия используются только для удобства просмотра.
    """
    clipped = np.clip(arr, 0.0, 1.0)
    inv = 1.0 - clipped
    img = Image.fromarray((inv * 255.0).astype(np.uint8), mode='L')
    return img.resize((scale, scale), Image.NEAREST)


def generate_gif(
    psi_init: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    courant: float,
    frames_fwd: int,
    frames_bwd: int,
    steps_per_frame: int,
    output_path: str,
) -> None:
    """
    Интегрирует изображение вперёд и назад по времени и сохраняет
    последовательность кадров в виде GIF. Используются центральные
    разности и интегратор Рунге–Кутты 4‑го порядка, то есть схема
    КАБАРЭ без диссипации. Параметры ``frames_fwd`` и ``frames_bwd``
    определяют длину зашифрования и расшифрования; ``steps_per_frame`` —
    количество шагов между кадрами. Шаг по времени вычисляется как
    ``dt = C / n``, где ``n`` — число узлов.
    """
    # размеры сетки и шаги
    ny, nx = psi_init.shape
    dx = 1.0 / nx
    dy = 1.0 / ny
    # шаг времени из числа Куранта (вариант КАБАРЭ)
    dt = courant / nx
    # копия данных и контейнер для кадров
    psi = psi_init.copy()
    frames: List[Image.Image] = [to_frame(psi)]
    # прямой ход (шифрование)
    for _ in range(frames_fwd):
        for _ in range(steps_per_frame):
            psi = rk4_step(psi, dt, dx, dy, u, v)
        frames.append(to_frame(psi))
    # обратный ход (дешифрование)
    for _ in range(frames_bwd):
        for _ in range(steps_per_frame):
            psi = rk4_step(psi, -dt, dx, dy, u, v)
        frames.append(to_frame(psi))
    frames.append(to_frame(psi_init))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )


def parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки. Параметр ``--factor``
    увеличивает число кадров в прямом и обратном направлении,
    позволяя продлить анимацию (вариант настройки схемы КАБАРЭ).
    """
    parser = argparse.ArgumentParser(
        description=(
            'Создание GIF по схеме КАБАРЭ (без диффузии) с заданным числом Куранта.\n'
            'Используйте параметр --factor для увеличения длины закручивания и '
            'раскручивания. При значении 10 анимация будет в 10 раз длиннее.'
        )
    )
    parser.add_argument(
        '--image', type=str,
        default='2.png',
        help='Путь к исходному изображению'
    )
    parser.add_argument(
        '--size', type=int, default=120,
        help='Размер сетки по каждой координате'
    )
    parser.add_argument(
        '--courant', type=float, default=0.5,
        help='Число Куранта C (0 < C ≤ 1). Используется для вычисления dt = C/n.'
    )
    parser.add_argument(
        '--frames_fwd', type=int, default=20,
        help='Базовое количество кадров в прямом направлении'
    )
    parser.add_argument(
        '--frames_bwd', type=int, default=20,
        help='Базовое количество кадров в обратном направлении'
    )
    parser.add_argument(
        '--steps_per_frame', type=int, default=5,
        help='Количество временных шагов между кадрами GIF'
    )
    parser.add_argument(
        '--factor', type=int, default=10,
        help=(
            'Множитель для числа кадров. По умолчанию 10 — изображение '
            'закручивается в десять раз дольше. Для другого значения укажите '
            'свой множитель.'
        )
    )
    parser.add_argument(
        '--output_path', type=str, default='swirl_cabare2.gif',
        help='Путь сохранения GIF'
    )
    return parser.parse_args()


def main() -> None:
    """
    Основная функция: загружает изображение, строит поле скоростей и
    формирует GIF с помощью варианта схемы КАБАРЭ. Параметр ``factor``
    увеличивает количество кадров для более длительного закручивания.
    """
    args = parse_args()
    # Загрузка и подготовка изображения
    psi0 = load_and_prepare_image(args.image, size=args.size)
    ny, nx = psi0.shape
    # Построение поля скоростей вращения
    u, v = create_velocity_field(nx, ny)
    # Учитываем коэффициент удлинения. Если factor>1, увеличиваем число
    # кадров в прямом и обратном направлении пропорционально.
    frames_fwd = args.frames_fwd * args.factor
    frames_bwd = args.frames_bwd * args.factor
    # Генерируем анимацию
    generate_gif(
        psi0, u, v, args.courant,
        frames_fwd=frames_fwd,
        frames_bwd=frames_bwd,
        steps_per_frame=args.steps_per_frame,
        output_path=args.output_path,
    )
    print(f'GIF сохранён: {args.output_path}')


if __name__ == '__main__':
    main()