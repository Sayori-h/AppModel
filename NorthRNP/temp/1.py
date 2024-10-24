import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体来支持中文字符
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
file_path = '../../doc/mktNorth.xlsx'  # 请确保文件路径正确
df = pd.read_excel(file_path)

# 从Excel中提取X和Y坐标
x_values = df['X']
y_values = df['Y']

coordinates = [
    (111.59349999999999, 36.351555555555556),
    (111.77055555555556, 36.37672222222222),
    (111.81224999999999, 36.43702777777777),
    (111.70322222222222, 36.367194444444444)
]

R = 6378137  # 地球半径（墨卡托投影使用的常数）

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

p = np.array(sorted_mercator_coordinates)

def calculate_angle(vector1, vector2):
    """
    Calculate the angle from vector1 to vector2.
    Counterclockwise is positive, clockwise is negative.

    Parameters:
    vector1 (array-like): The first vector.
    vector2 (array-like): The second vector.

    Returns:
    float: The angle in degrees.
    """
    # Product of the magnitudes of the two vectors
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    # Cross product
    cross_product = np.rad2deg(np.arcsin(np.cross(vector1, vector2) / norm_product))
    # Dot product
    dot_product = np.rad2deg(np.arccos(np.dot(vector1, vector2) / norm_product))

    if cross_product < 0:
        return -dot_product
    else:
        return dot_product

def calculate_point_on_line(A, B, l):
    """
    Calculate the point on the line segment AB that is at a distance l from point B.

    Parameters:
    A (array-like): The first point of the line segment.
    B (array-like): The second point of the line segment.
    l (float): The distance from point B.

    Returns:
    array-like: The coordinates of the point at distance l from B.
    """
    # Vector from A to B
    AB = np.array(A) - np.array(B)
    # Unit vector in the direction of AB
    AB_unit = AB / np.linalg.norm(AB)
    # Point at distance l from B
    point = np.array(B) + l * AB_unit
    return point

def calculate_point_on_perpendicular(A, B, P, d=1):
    """
    Calculate a point on the perpendicular line from point P to the line AB, at a distance d from P.

    Parameters:
    A (array-like): The first point of the line segment.
    B (array-like): The second point of the line segment.
    P (array-like): The point from which the perpendicular is dropped.
    d (float): The distance from point P to the desired point on the perpendicular line.

    Returns:
    array-like: The coordinates of the point on the perpendicular line at distance d from P.
    """
    A = np.array(A)
    B = np.array(B)
    P = np.array(P)

    # Vector from A to B
    AB = B - A
    # Unit vector in the direction of AB
    AB_unit = AB / np.linalg.norm(AB)
    # Perpendicular vector to AB
    perpendicular_vector = np.array([-AB_unit[1], AB_unit[0]])

    # Calculate the point at distance d from P along the perpendicular vector
    point_on_perpendicular = P + d * perpendicular_vector

    return point_on_perpendicular

def calculate_intersection(A, B, C, D):
    """
    Calculate the intersection point of lines AB and CD.
    If the lines are parallel or coincident, raise an error.

    Parameters:
    A (array-like): The first point of the first line segment.
    B (array-like): The second point of the first line segment.
    C (array-like): The first point of the second line segment.
    D (array-like): The second point of the second line segment.

    Returns:
    array-like: The coordinates of the intersection point.
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * A[0] + b1 * A[1]

    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * C[0] + b2 * C[1]

    determinant = a1 * b2 - a2 * b1

    if np.abs(determinant) <= 1e-5:
        return None

    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    return np.array([x, y])

def calculate_tangent_circle(A, B, C, d):
    """
    Calculate the center and radius of the tangent circle at point B on the polyline ABC,
    such that the distance from B to the tangent point on the circle is d, and the arc range.

    Parameters:
    A (array-like): The first point of the polyline.
    B (array-like): The second point of the polyline (the vertex where the circle is tangent).
    C (array-like): The third point of the polyline.
    d (float): The distance from point B to the tangent point on the circle.

    Returns:
    tuple: The coordinates of the center of the circle, the radius of the circle.
    """

    # Adjust the center to ensure the circle is tangent to the polyline at distance d from B
    tangent_point_A = calculate_point_on_line(A, B, d)
    tangent_point_A2 = calculate_point_on_perpendicular(A, B, tangent_point_A)
    tangent_point_C = calculate_point_on_line(C, B, d)
    tangent_point_C2 = calculate_point_on_perpendicular(C, B, tangent_point_C)
    center = calculate_intersection(tangent_point_A, tangent_point_A2, tangent_point_C, tangent_point_C2)
    if center is not None:
        radius = np.linalg.norm(tangent_point_A - center)
        return center, radius
    else:
        return None, d

def calculate_L2(V_kt=180, c_seconds=5):
    c_seconds = 5  # 建立坡度时间，单位为秒
    V = V_kt * 0.514444  # 将速度从节 (kt) 转换为米每秒 (m/s)
    return V * c_seconds

# 创建一个统一的绘图对象
fig, ax = plt.subplots(figsize=(12, 10))  # 增大图像尺寸以更好展示

# 绘制实际航迹的散点图，设置点的大小
scatter = ax.scatter(x_values, y_values, color='red', s=15, label='实际航迹', alpha=0.5)  # s=5 表示点的大小

# 绘制 smoothed polyline using tangent circles
for i in range(1, len(p) - 1):
    A = np.array(p[i - 1])
    B = np.array(p[i])
    C = np.array(p[i + 1])

    # Distance from the vertex to the tangent point on the circle
    L2 = calculate_L2()
    # TODO: d is calculator by user, np.linalg.norm(A - B) - L2
    d = 1852  # 请根据需要调整此值

    center, radius = calculate_tangent_circle(A, B, C, d)
    print(f"A: {A}, B: {B}, C: {C}, Center: {center}, Radius: {radius}")
    if center is None:
        start_point = B
        end_point = B
    else:
        # Draw the arc
        start_point = calculate_point_on_line(A, B, d)
        end_point = calculate_point_on_line(C, B, d)
        start_angle = calculate_angle((1, 0), start_point - center)
        end_angle = calculate_angle((1, 0), end_point - center)
        theta = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), 100)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        ax.plot(x, y, color="blue", linewidth=4, label='定义航迹' if i == 1 else "")  # 加粗蓝色线条

    # Draw lines from the previous point to the start of the arc and from the end of the arc to the next point
    if i == 1:
        ax.plot([A[0], start_point[0]], [A[1], start_point[1]], color="blue", linewidth=4)
    else:
        ax.plot([last_point[0], start_point[0]], [last_point[1], start_point[1]], color="blue", linewidth=4)
    if i == len(p) - 2:
        ax.plot([end_point[0], C[0]], [end_point[1], C[1]], color="blue", linewidth=4)
    last_point = end_point

# 添加标题和标签，并设置较大的字号
ax.set_xlabel('X/m', fontsize=18)
ax.set_ylabel('Y/m', fontsize=18)

# 调整坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=16)

# 调整偏移文本（如1e6, 1e7）的字体大小
ax.xaxis.get_offset_text().set_fontsize(16)
ax.yaxis.get_offset_text().set_fontsize(16)

# 添加图例，并设置较大的字号
ax.legend(fontsize=18)

# 显示图像
plt.show()
