import pandas as pd
import matplotlib.pyplot as plt
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体来支持中文字符
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
file_path = '../doc/mktNorth.xlsx'  # 请确保文件路径正确
df = pd.read_excel(file_path)

# 提取X, Y, HEIGHT坐标
x_values = df['X']
y_values = df['Y']
z_values = df['HEIGHT']

# 创建三维散点图
fig = plt.figure(figsize=(10, 8))  # 调整图形大小
ax = fig.add_subplot(111, projection='3d')

# 绘制实际航迹的三维散点图
ax.scatter(x_values, y_values, z_values, color='red', s=5, label='实际航迹', alpha=0.5)

# 经纬度数据，用于绘制定义航迹的折线图
coordinates = [
    (111.59349999999999, 36.351555555555556, 1500),
    (111.77055555555556, 36.37672222222222, 2300),
    (111.81224999999999, 36.43702777777777, 2700),
    (111.70322222222222, 36.367194444444444, 2000)
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

# 对墨卡托坐标按照 X 值进行排序，确保线是依次连接而没有环
sorted_mercator_coordinates = sorted(mercator_coordinates, key=lambda coord: coord[0])

# 提取排序后的 X, Y, Z 坐标
sorted_line_x_values = [coord[0] for coord in sorted_mercator_coordinates]
sorted_line_y_values = [coord[1] for coord in sorted_mercator_coordinates]
sorted_line_z_values = [coord[2] for coord in sorted_mercator_coordinates]

# 绘制定义航迹的折线图
ax.plot(sorted_line_x_values, sorted_line_y_values, sorted_line_z_values,
        color='blue', linewidth=3, label='定义航迹')

# 设置轴标签
ax.set_xlabel('X/m')
ax.set_ylabel('Y/m')
ax.set_zlabel('Height/m')

# 设置轴的缩放
ax.ticklabel_format(style='sci', axis='x', scilimits=(7, 7))
ax.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))

# 调整视角和比例，使其更接近参考图
ax.view_init(elev=25, azim=140)  # elev 控制俯仰角度，azim 控制方位角度
ax.set_box_aspect([2.5, 1.5, 1.2])  # 调整 X, Y, Z 轴的比例

# 添加图例
# Move the legend inside the plot closer to the upper right corner
ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9), frameon=False)

# 显示图像
plt.show()
