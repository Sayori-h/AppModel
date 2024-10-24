import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ILStools.constant import FIG_SIZE, SCATTER_COLOR, SCATTER_SIZE, SCATTER_ALPHA, SCATTER_LABEL, LINE_WIDTH, \
    LINE_COLORS, \
    LINE_LABELS, FONT_SIZE, TICK_LABEL_SIZE, OFFSET_TEXT_SIZE, LEGEND_FONT_SIZE, OFFSET_UP, OFFSET_DOWN, d, R


def load_coordinates(file_path):
    """
    读取Excel文件，提取X和Y坐标。

    参数：
    file_path (str): Excel文件的路径。

    返回：
    tuple: 包含X和Y坐标的元组 (x_values, y_values)。
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体来支持中文字符
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 从Excel中提取X和Y坐标
    x_values = df['X']
    y_values = df['Y']

    return x_values, y_values


def latlon_to_mercator(longitude, latitude):
    lon_rad = math.radians(longitude)
    lat_rad = math.radians(latitude)
    x = R * lon_rad
    y = R * math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return x, y


def convert_coordinates(coordinates):
    """
    将经纬度坐标列表转换为墨卡托投影坐标，并按X值排序。

    参数：
    coordinates (list of tuples): 经纬度坐标列表，例如 [(lon1, lat1), (lon2, lat2), ...]。

    返回：
    numpy.ndarray: 排序后的墨卡托投影坐标数组。
    """

    # 墨卡托投影的转换
    mercator_coordinates = [latlon_to_mercator(lon, lat) for lon, lat in coordinates]
    # 按X值排序
    sorted_mercator_coordinates = sorted(mercator_coordinates, key=lambda coord: coord[0])

    # 输出转换后的墨卡托坐标
    for mercator in sorted_mercator_coordinates:
        print(mercator)

    # 转换为numpy数组
    p = np.array(sorted_mercator_coordinates)
    return p


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
    cross_product = np.cross(vector1, vector2)
    # Dot product
    dot_product = np.dot(vector1, vector2)
    angle = np.arctan2(cross_product, dot_product)
    angle_deg = np.degrees(angle)
    return angle_deg


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
    # Vector from B to A
    BA = np.array(A) - np.array(B)
    # Unit vector in the direction of BA
    BA_unit = BA / np.linalg.norm(BA)
    # Point at distance l from B towards A
    point = np.array(B) + l * BA_unit
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
    If the lines are parallel or coincident, return None.

    Parameters:
    A (array-like): The first point of the first line.
    B (array-like): The second point of the first line.
    C (array-like): The first point of the second line.
    D (array-like): The second point of the second line.

    Returns:
    array-like: The coordinates of the intersection point, or None if no intersection.
    """
    xdiff = (A[0] - B[0], C[0] - D[0])
    ydiff = (A[1] - B[1], C[1] - D[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if np.abs(div) < 1e-10:
        return None

    d = (det(A, B), det(C, D))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])


def calculate_tangent_circle(A, B, C, d):
    """
    Calculate the center and radius of the tangent circle at point B on the polyline ABC,
    such that the distance from B to the tangent point on the circle is d.

    Parameters:
    A (array-like): The first point of the polyline.
    B (array-like): The second point of the polyline (the vertex where the circle is tangent).
    C (array-like): The third point of the polyline.
    d (float): The distance from point B to the tangent point on the circle.

    Returns:
    tuple: The coordinates of the center of the circle, the radius of the circle.
    """
    # Points at distance d from B towards A and C
    tangent_point_A = calculate_point_on_line(A, B, d)
    tangent_point_C = calculate_point_on_line(C, B, d)

    # Perpendicular lines at tangent points
    tangent_point_A2 = calculate_point_on_perpendicular(A, B, tangent_point_A)
    tangent_point_C2 = calculate_point_on_perpendicular(C, B, tangent_point_C)

    center = calculate_intersection(tangent_point_A, tangent_point_A2, tangent_point_C, tangent_point_C2)
    if center is not None:
        radius = np.linalg.norm(tangent_point_A - center)
        return center, radius
    else:
        return None, d


def calculate_L2(V_kt=180, c_seconds=5):
    V = V_kt * 0.514444  # 将速度从节 (kt) 转换为米每秒 (m/s)
    return V * c_seconds


# 封装绘制平滑航迹的函数
def plot_smoothed_path(ax, p, offset=0, color='blue', label='定义航迹'):
    last_point = None
    offset_points = []
    for i in range(1, len(p) - 1):
        A = np.array(p[i - 1])
        B = np.array(p[i])
        C = np.array(p[i + 1])

        # Distance from the vertex to the tangent point on the circle
        L2 = calculate_L2()

        center, radius = calculate_tangent_circle(A, B, C, d)
        if center is None:
            start_point = B
            end_point = B
        else:
            # Draw the arc
            start_point = calculate_point_on_line(A, B, d)
            end_point = calculate_point_on_line(C, B, d)
            start_angle = calculate_angle((1, 0), start_point - center)
            end_angle = calculate_angle((1, 0), end_point - center)

            # 确保角度在0-360度之间
            start_angle = start_angle % 360
            end_angle = end_angle % 360

            # 处理角度跨越0度的情况
            if end_angle - start_angle > 180:
                end_angle -= 360
            elif end_angle - start_angle < -180:
                end_angle += 360

            theta = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), 100)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            path_coords = np.vstack((x, y)).T

            # 如果需要偏移
            if offset != 0:
                # 计算每个点的法向量
                dx = np.gradient(x)
                dy = np.gradient(y)
                lengths = np.hypot(dx, dy)
                normals_x = -dy / lengths
                normals_y = dx / lengths
                x_offset = x + offset * normals_x
                y_offset = y + offset * normals_y
                ax.plot(x_offset, y_offset, color=color, linewidth=LINE_WIDTH, label=label if i == 1 else "")
                offset_points.extend(list(zip(x_offset, y_offset)))
            else:
                ax.plot(x, y, color=color, linewidth=LINE_WIDTH, label=label if i == 1 else "")
                offset_points.extend(list(zip(x, y)))

        # Draw lines from the previous point to the start of the arc and from the end of the arc to the next point
        if i == 1:
            # Line before the arc
            line_start = A
            line_end = start_point
            if offset != 0:
                # 计算线段的法向量
                line_vec = line_end - line_start
                line_length = np.hypot(line_vec[0], line_vec[1])
                normal = np.array([-line_vec[1], line_vec[0]]) / line_length
                line_start_offset = line_start + offset * normal
                line_end_offset = line_end + offset * normal
                ax.plot([line_start_offset[0], line_end_offset[0]], [line_start_offset[1], line_end_offset[1]],
                        color=color, linewidth=LINE_WIDTH)
            else:
                ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color=color, linewidth=LINE_WIDTH)
        else:
            # Line between arcs
            line_start = last_point
            line_end = start_point
            if offset != 0:
                # 计算线段的法向量
                line_vec = line_end - line_start
                line_length = np.hypot(line_vec[0], line_vec[1])
                normal = np.array([-line_vec[1], line_vec[0]]) / line_length
                line_start_offset = line_start + offset * normal
                line_end_offset = line_end + offset * normal
                ax.plot([line_start_offset[0], line_end_offset[0]], [line_start_offset[1], line_end_offset[1]],
                        color=color, linewidth=LINE_WIDTH)
            else:
                ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color=color, linewidth=LINE_WIDTH)
        if i == len(p) - 2:
            # Line after the arc
            line_start = end_point
            line_end = C
            if offset != 0:
                # 计算线段的法向量
                line_vec = line_end - line_start
                line_length = np.hypot(line_vec[0], line_vec[1])
                normal = np.array([-line_vec[1], line_vec[0]]) / line_length
                line_start_offset = line_start + offset * normal
                line_end_offset = line_end + offset * normal
                ax.plot([line_start_offset[0], line_end_offset[0]], [line_start_offset[1], line_end_offset[1]],
                        color=color, linewidth=LINE_WIDTH)
            else:
                ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color=color, linewidth=LINE_WIDTH)
        last_point = end_point

    return offset_points


def plot_paths(x_values, y_values, p):
    """
    绘制实际航迹的散点图和平滑航迹线，包括上下偏移的航迹线。

    参数：
    x_values (array-like): 实际航迹的X坐标。
    y_values (array-like): 实际航迹的Y坐标。
    p (numpy.ndarray): 墨卡托投影并排序后的坐标数组。
    """

    # 创建一个统一的绘图对象
    fig, ax = plt.subplots(figsize=FIG_SIZE)  # 使用常量 FIG_SIZE

    # 绘制实际航迹的散点图，使用常量配置
    ax.scatter(x_values, y_values, color=SCATTER_COLOR, s=SCATTER_SIZE, label=SCATTER_LABEL, alpha=SCATTER_ALPHA)

    # 绘制原始蓝色平滑航迹线
    plot_smoothed_path(ax, p, offset=0, color=LINE_COLORS['original'], label=LINE_LABELS['original'])

    # 使用常量 OFFSET_UP 绘制向上平移的平滑航迹线
    plot_smoothed_path(ax, p, offset=OFFSET_UP, color=LINE_COLORS['offset_up'], label=LINE_LABELS['offset_up'])

    # 使用常量 OFFSET_DOWN 绘制向下平移的平滑航迹线
    plot_smoothed_path(ax, p, offset=OFFSET_DOWN, color=LINE_COLORS['offset_down'], label=LINE_LABELS['offset_down'])

    # 添加标题和标签，并使用常量设置字体大小
    ax.set_xlabel('X/m', fontsize=FONT_SIZE)
    ax.set_ylabel('Y/m', fontsize=FONT_SIZE)

    # 调整坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

    # 调整偏移文本的字体大小
    ax.xaxis.get_offset_text().set_fontsize(OFFSET_TEXT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(OFFSET_TEXT_SIZE)

    # 添加图例，并设置字体大小
    ax.legend(fontsize=LEGEND_FONT_SIZE)

    # 显示图像
    plt.show()

