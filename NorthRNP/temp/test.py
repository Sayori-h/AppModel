import numpy as np
import ILStools.ILStools as tl
import matplotlib.pyplot as plt

file_path = '../../doc/mktNorth.xlsx'
# 调用封装的函数加载坐标数据
x_values, y_values = tl.load_coordinates(file_path)

coordinates = [
    (111.59349999999999, 36.351555555555556),
    (111.77055555555556, 36.37672222222222),
    (111.81224999999999, 36.43702777777777),
    (111.70322222222222, 36.367194444444444)
]

# 墨卡托投影的转换
mercator_coordinates = [tl.latlon_to_mercator(lon, lat) for lon, lat in coordinates]
sorted_mercator_coordinates = sorted(mercator_coordinates, key=lambda coord: coord[0])
# 输出转换后的墨卡托坐标
for mercator in sorted_mercator_coordinates:
    print(mercator)

p = np.array(sorted_mercator_coordinates)


# 创建一个统一的绘图对象
fig, ax = plt.subplots(figsize=(12, 10))  # 增大图像尺寸以更好展示

# 绘制实际航迹的散点图，设置点的大小
ax.scatter(x_values, y_values, color='red', s=15, label='实际航迹', alpha=0.5)

# 绘制原始蓝色平滑航迹线
original_path = tl.plot_smoothed_path(ax, p, offset=0, color='blue', label='定义航迹')

# 绘制向上平移1000米（垂直距离）的蓝色平滑航迹线
tl.plot_smoothed_path(ax, p, offset=1000, color='green', label='平移+1000m')

# 绘制向下平移1000米（垂直距离）的蓝色平滑航迹线
tl.plot_smoothed_path(ax, p, offset=-1000, color='orange', label='平移-1000m')

# 添加标题和标签，并设置较大的字号
ax.set_xlabel('X/m', fontsize=18)
ax.set_ylabel('Y/m', fontsize=18)

# 调整坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=16)

# 调整偏移文本（如1e6, 1e7）的字体大小
ax.xaxis.get_offset_text().set_fontsize(16)
ax.yaxis.get_offset_text().set_fontsize(16)

# 添加图例，并设置较大的字号
ax.legend(fontsize=14)

# 显示图像
plt.show()
