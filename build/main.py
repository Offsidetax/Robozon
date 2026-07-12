import numpy as np
import matplotlib.pyplot as plt

width, height = 800, 600

# ИСПРАВЛЕНИЕ 1: Читаем новый файл с высотами (Z-координатами)
depth_data = np.fromfile("height_output.bin", dtype=np.float32)
depth_map = depth_data.reshape((height, width))

# ИСПРАВЛЕНИЕ 2: Теперь фон ленты равен строго 0.0, убираем его
depth_map[depth_map == 0.0] = np.nan

plt.figure(figsize=(10, 8))
plt.imshow(depth_map, cmap='viridis')
plt.colorbar(label='Абсолютная высота (мм)')
plt.title("Визуализация матрицы высот (Наклонная камера)")
plt.tight_layout()
plt.show()