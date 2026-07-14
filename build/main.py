import numpy as np
import matplotlib.pyplot as plt
import os

# Проверяем наличие файлов, чтобы скрипт не падал, если C++ еще не отработал
if not os.path.exists("cloud1.bin") or not os.path.exists("cloud2.bin"):
    print("Ошибка: Файлы cloud1.bin или cloud2.bin не найдены!")
    print("Сначала запустите скомпилированный C++ файл (.exe).")
    exit()

# 1. Читаем бинарные файлы. 
# dtype=np.float32 совпадает с типом float в C++.
# .reshape(-1, 3) группирует сырой массив чисел в матрицу [Количество точек, 3 координаты (X,Y,Z)]
cloud1 = np.fromfile("cloud1.bin", dtype=np.float32).reshape(-1, 3)
cloud2 = np.fromfile("cloud2.bin", dtype=np.float32).reshape(-1, 3)

print(f"Загружено с Камеры 1: {len(cloud1)} точек")
print(f"Загружено с Камеры 2: {len(cloud2)} точек")

# 2. Настраиваем 3D-график
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 3. Отрисовываем облака точек
# Ограничиваем размер точек (s=1) и делаем их полупрозрачными (alpha=0.5) для эффекта сканера
# Камера 1 (Спереди) — синим цветом
if len(cloud1) > 0:
    ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], 
               c='#0078D7', s=1, alpha=0.5, label='Камера 1 (Спереди)')

# Камера 2 (Сзади) — красным цветом
if len(cloud2) > 0:
    ax.scatter(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2], 
               c='#D13438', s=1, alpha=0.5, label='Камера 2 (Сзади)')

# 4. Настройка осей и легенды
ax.set_title("Симуляция работы двух 3D-камер (Point Cloud)", fontsize=14, fontweight='bold')
ax.set_xlabel("Ось X (Ширина конвейера, мм)", labelpad=10)
ax.set_ylabel("Ось Y (Вдоль конвейера, мм)", labelpad=10)
ax.set_zlabel("Ось Z (Высота объекта, мм)", labelpad=10)
ax.legend(markerscale=10, loc='upper right') # markerscale увеличивает точки в легенде

# 5. Выравнивание пропорций осей (чтобы коробки не выглядели сплющенными)
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