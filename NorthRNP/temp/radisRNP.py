import math

import matplotlib.pyplot as plt
import numpy as np

# 输入的坐标点和墨卡托投影转换函数
R = 6378137  # 地球半径（墨卡托投影使用的常数）

# 经纬度数据
coordinates = [
    (111.7459444, 36.16991667),
    (111.7032222, 36.36719444),
    (111.5935, 36.35155556)
]


def latlon_to_mercator(longitude, latitude):
    lon_rad = math.radians(longitude)
    lat_rad = math.radians(latitude)
    x = R * lon_rad
    y = R * math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return x, y


# 墨卡托投影的转换
mercator_coordinates = [latlon_to_mercator(lon, lat) for lon, lat in coordinates]
sorted_mercator_coordinates = sorted(mercator_coordinates, key=lambda coord: coord[0])

# 输出转换后的墨卡托坐标
for mercator in sorted_mercator_coordinates:
    print(mercator)

# 提取排序后的X和Y坐标
sorted_line_x_values = [coord[0] for coord in sorted_mercator_coordinates]
sorted_line_y_values = [coord[1] for coord in sorted_mercator_coordinates]


# 计算两向量的夹角
def calculate_angle(coord1, coord2, coord3):
    # 向量 A 为 coord1 -> coord2, 向量 B 为 coord2 -> coord3
    vector_a = (coord2[0] - coord1[0], coord2[1] - coord1[1])
    vector_b = (coord3[0] - coord2[0], coord3[1] - coord2[1])

    # 计算向量点积
    dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]

    # 计算向量的模（长度）
    magnitude_a = math.sqrt(vector_a[0] ** 2 + vector_a[1] ** 2)
    magnitude_b = math.sqrt(vector_b[0] ** 2 + vector_b[1] ** 2)

    # 通过点积公式计算夹角（弧度）
    cos_angle = dot_product / (magnitude_a * magnitude_b)

    # 防止cos值略微超过1或低于-1，导致math.acos计算出错
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # 返回夹角（弧度转角度）
    return math.degrees(math.acos(cos_angle))


# 计算转弯半径的函数
def calculate_turn_radius(V_kt, L1_L2_NM, c_seconds, theta_deg):
    V = V_kt * 0.514444  # 将速度从节 (kt) 转换为米每秒 (m/s)
    L1_L2 = L1_L2_NM * 1852  # 将距离从海里 (NM) 转换为米 (m)

    # 计算转弯半径 r
    r = (L1_L2 - c_seconds * V / 3600) / math.tan(math.radians(theta_deg / 2))
    return r


# north
# y = 0.17699x + 2150439.45656
# y = 1.99243x - 20438619.016
# south
# y = 0.17699x + 2150439.45718
# y = -5.72735x + 75569406.55625
# 参数
V_kt = 180  # 速度，单位为节 (kt)
L1_L2_NM = 0.8  # 距离，单位为海里 (NM)
c_seconds = 5  # 建立坡度时间，单位为秒
A_coords = (12442241.33012574, 4352582.352807988)

# 计算转弯角度
theta_deg = calculate_angle(coordinates[0], coordinates[1], coordinates[2])
print(theta_deg)

# 计算转弯半径
turn_radius = calculate_turn_radius(V_kt, L1_L2_NM, c_seconds, theta_deg)
print(turn_radius)


# 计算圆心坐标的函数
def calculate_circle_center(A_coords, turn_radius, theta_deg):
    x_A, y_A = A_coords  # 解包A点的坐标
    theta_rad = math.radians(theta_deg)  # 将角度从度转换为弧度

    # 计算圆心 O 的坐标
    x_O = x_A + turn_radius * math.cos(theta_rad / 2)
    y_O = y_A + turn_radius * math.sin(theta_rad / 2)

    return x_O, y_O


# 拐点参数
# 计算圆心坐标
# turn_center = calculate_circle_center(A_coords, turn_radius, theta_deg)


# 绘制图形
plt.figure(figsize=(10, 8))

# 绘制原始折线
plt.plot(sorted_line_x_values, sorted_line_y_values, 'bo-', label="Original Path", linewidth=2)

# 图例和标题
plt.xlabel("X (Mercator)")
plt.ylabel("Y (Mercator)")
plt.title("Path with Rounded Turning Point")
plt.legend()
plt.grid(True)

plt.show()
