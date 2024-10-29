import pandas as pd
import matplotlib.pyplot as plt
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体来支持中文字符
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
file_path = '../doc/mktSouth.xlsx'
df = pd.read_excel(file_path)

# 提取X, Y, HEIGHT坐标
x_values = df['X']
y_values = df['Y']
z_values = df['HEIGHT']
z_values *= 0.3048

# 创建三维散点图
fig = plt.figure(figsize=(10, 8))  # 调整图形大小
ax = fig.add_subplot(111, projection='3d')

# 绘制实际航迹的三维散点图
ax.scatter(x_values, y_values, z_values, color='red', s=3, label='实际航迹', alpha=0.2)

# 经纬度数据，用于绘制定义航迹的折线图
coordinates = [
    (111.7459444, 36.16991667, 2700),
    (111.7032222, 36.36719444, 2000),
    (111.5935, 36.35155556, 1500)
]

# 墨卡托投影的转换函数
R = 6378137


def latlon_to_mercator(longitude, latitude):
    lon_rad = math.radians(longitude)
    lat_rad = math.radians(latitude)
    x = R * lon_rad
    y = R * math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return x, y


# 转换经纬度数据为墨卡托投影坐标
mercator_coordinates = [latlon_to_mercator(lon, lat) + (height,) for lon, lat, height in coordinates]
line_x_values = [coord[0] for coord in mercator_coordinates]
line_y_values = [coord[1] for coord in mercator_coordinates]
line_z_values = [coord[2] for coord in mercator_coordinates]

# 绘制定义航迹的折线图
ax.plot(line_x_values, line_y_values, line_z_values, color='blue', linewidth=3, label='定义航迹')

# 设置轴标签
ax.set_xlabel('X/m')
ax.set_ylabel('Y/m')
ax.set_zlabel('Height/m')

# 设置轴的缩放
ax.ticklabel_format(style='sci', axis='x', scilimits=(7, 7))
ax.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))

# 调整视角和比例，使其更接近参考图
ax.view_init(elev=25, azim=140)  # elev 控制俯仰角度，azim 控制方位角度
ax.set_box_aspect([1.5, 3, 1.2])  # 调整 X, Y, Z 轴的比例

# 设置轴的范围，使图形更贴近参考图


# 添加图例
# Move the legend inside the plot closer to the upper right corner
ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9), frameon=False)
# 显示图像
plt.show()
