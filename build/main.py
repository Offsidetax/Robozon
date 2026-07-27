import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

# Проверяем наличие файлов
if not os.path.exists("cloud1.bin") or not os.path.exists("cloud2.bin"):
    print("Ошибка: Файлы cloud1.bin или cloud2.bin не найдены!")
    print("Сначала запустите скомпилированный C++ файл (.exe).")
    exit()

# 1. Читаем облака точек
cloud1 = np.fromfile("cloud1.bin", dtype=np.float32).reshape(-1, 3)
cloud2 = np.fromfile("cloud2.bin", dtype=np.float32).reshape(-1, 3)

# 2. Настраиваем 3D-график
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 3. Отрисовываем облака точек
if len(cloud1) > 0:
    ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], 
               c='#0078D7', s=1, alpha=0.5, label='Камера 1 (Спереди)')
if len(cloud2) > 0:
    ax.scatter(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2], 
               c='#D13438', s=1, alpha=0.5, label='Камера 2 (Сзади)')

# --- ДОБАВЛЕННЫЙ БЛОК: ЧТЕНИЕ И ОТРИСОВКА OBB РАМКИ ---
if os.path.exists("obb_data.txt"):
    with open("obb_data.txt", "r") as f:
        lines = f.readlines()
        center = np.array(list(map(float, lines[0].split())))
        axes = np.array([list(map(float, lines[i].split())) for i in range(1, 4)])
        min_ext = np.array(list(map(float, lines[4].split())))
        max_ext = np.array(list(map(float, lines[5].split())))

    # Вычисляем координаты 8 вершин куба (рамки)
    vertices = []
    for i, j, k in itertools.product([0, 1], repeat=3):
        ext0 = max_ext[0] if i else min_ext[0]
        ext1 = max_ext[1] if j else min_ext[1]
        ext2 = max_ext[2] if k else min_ext[2]
        # Векторное сложение: от центра шагаем вдоль осей на нужные дистанции
        v = center + ext0 * axes[0] + ext1 * axes[1] + ext2 * axes[2]
        vertices.append(v)
    vertices = np.array(vertices)

    # Список из 12 ребер, соединяющих 8 вершин
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Рисуем линии рамки неоново-зеленым цветом (lime)
    for idx, edge in enumerate(edges):
        ax.plot([vertices[edge[0], 0], vertices[edge[1], 0]],
                [vertices[edge[0], 1], vertices[edge[1], 1]],
                [vertices[edge[0], 2], vertices[edge[1], 2]], 
                color='lime', linewidth=2.5, label='Габаритная рамка (OBB)' if idx == 0 else "")
# -----------------------------------------------------

# 4. Настройка осей и легенды
ax.set_title("Симуляция работы машинного зрения: Облако точек + OBB", fontsize=14, fontweight='bold')
ax.set_xlabel("Ось X (Ширина конвейера, мм)", labelpad=10)
ax.set_ylabel("Ось Y (Вдоль конвейера, мм)", labelpad=10)
ax.set_zlabel("Ось Z (Высота объекта, мм)", labelpad=10)
ax.legend(markerscale=5, loc='upper right')

# 5. Выравнивание пропорций осей
all_points = np.vstack((cloud1, cloud2)) if len(cloud1) > 0 and len(cloud2) > 0 else cloud1
if len(all_points) > 0:
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.show()