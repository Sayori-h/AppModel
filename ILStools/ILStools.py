import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .constant import *


# 控制散点图和包络线
def plot_all_paths(ax, x_values, y_values, p):
    """
    绘制实际航迹的散点图和平滑航迹线，包括上下偏移的航迹线。

    参数：
    ax (matplotlib.axes.Axes): 绘图的轴对象。
    x_values (array-like): 实际航迹的X坐标。
    y_values (array-like): 实际航迹的Y坐标。
    p (numpy.ndarray): 墨卡托投影并排序后的坐标数组。
    """
    # 绘制实际航迹的散点图
    # ax.scatter(x_values, y_values, color=SCATTER_COLOR, s=SCATTER_SIZE, label=SCATTER_LABEL, alpha=SCATTER_ALPHA)

    # 绘制原始蓝色平滑航迹线
    plot_smoothed_path(ax, p, offset=0, color=LINE_COLORS["original"], label=LINE_LABELS["original"])

    # 使用常量 OFFSET_UP 绘制向上平移的平滑航迹线
    # plot_smoothed_path(ax, p, offset=OFFSET_UP, color=LINE_COLORS["offset_up"], label=LINE_LABELS["offset_up"])

    # 使用常量 OFFSET_DOWN 绘制向下平移的平滑航迹线
    # plot_smoothed_path(ax, p, offset=OFFSET_DOWN, color=LINE_COLORS["offset_down"], label=LINE_LABELS["offset_down"])


def load_coordinates(file_path):
    """
    读取Excel文件，提取X和Y坐标。

    参数：
    file_path (str): Excel文件的路径。

    返回：
    tuple: 包含X和Y坐标的元组 (x_values, y_values)。
    """
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体字体来支持中文字符
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 从Excel中提取X和Y坐标
    x_values = df["X"]
    y_values = df["Y"]

    return x_values, y_values


def latlon_to_mercator(longitude, latitude):
    lon_rad = math.radians(longitude)
    lat_rad = math.radians(latitude)
    x = R * lon_rad
    y = R * math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return x, y


# 定义函数接收三维坐标输入
def convert_to_mercator_with_height(np_array):
    mercator_with_height = [
        (*latlon_to_mercator(lon, lat), height)
        for lon, lat, height in np_array
    ]
    return np.array(mercator_with_height)


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
    # 在 BA_unit 计算前确保 BA 向量的长度不为零
    BA_length = np.linalg.norm(BA)
    if BA_length != 0:
        BA_unit = BA / BA_length
    else:
        BA_unit = np.zeros_like(BA)  # 当 BA 向量长度为零时，用零向量替代
    # Unit vector in the direction of BA
    BA_unit = BA / np.linalg.norm(BA)
    # Point at distance l from B towards A
    point = np.array(B) + l * BA_unit
    return point


# def calculate_point_on_perpendicular(A, B, P, d=1):
#     """
#     Calculate a point on the perpendicular line from point P to the line AB, at a distance d from P.

#     Parameters:
#     A (array-like): The first point of the line segment.
#     B (array-like): The second point of the line segment.
#     P (array-like): The point from which the perpendicular is dropped.
#     d (float): The distance from point P to the desired point on the perpendicular line.

#     Returns:
#     array-like: The coordinates of the point on the perpendicular line at distance d from P.
#     """
#     A = np.array(A)
#     B = np.array(B)
#     P = np.array(P)

#     # Vector from A to B
#     AB = B - A

#     # 在 AB_unit 计算前确保 AB 向量的长度不为零
#     AB_length = np.linalg.norm(AB)
#     if AB_length != 0:
#         AB_unit = AB / AB_length
#     else:
#         AB_unit = np.zeros_like(AB)  # 当 AB 向量长度为零时，用零向量替代

#     # Unit vector in the direction of AB
#     AB_unit = AB / np.linalg.norm(AB)
#     # Perpendicular vector to AB
#     perpendicular_vector = np.array([-AB_unit[1], AB_unit[0]])

#     # Calculate the point at distance d from P along the perpendicular vector
#     point_on_perpendicular = P + d * perpendicular_vector

#     return point_on_perpendicular


# def calculate_intersection(A, B, C, D):
#     """
#     Calculate the intersection point of lines AB and CD.
#     If the lines are parallel or coincident, return None.

#     Parameters:
#     A (array-like): The first point of the first line.
#     B (array-like): The second point of the first line.
#     C (array-like): The first point of the second line.
#     D (array-like): The second point of the second line.

#     Returns:
#     array-like: The coordinates of the intersection point, or None if no intersection.
#     """
#     xdiff = (A[0] - B[0], C[0] - D[0])
#     ydiff = (A[1] - B[1], C[1] - D[1])

#     def det(a, b):
#         return a[0] * b[1] - a[1] * b[0]

#     div = det(xdiff, ydiff)
#     if np.abs(div) < 1e-10:
#         return None

#     d = (det(A, B), det(C, D))
#     x = det(d, xdiff) / div
#     y = det(d, ydiff) / div
#     return np.array([x, y])


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

    # # Perpendicular lines at tangent points
    # tangent_point_A2 = calculate_point_on_perpendicular(A, B, tangent_point_A)
    # tangent_point_C2 = calculate_point_on_perpendicular(C, B, tangent_point_C)

    # center = calculate_intersection(
    #     tangent_point_A, tangent_point_A2, tangent_point_C, tangent_point_C2
    # )
    # if center is not None:
    #     radius = np.linalg.norm(tangent_point_A - center)
    #     return center, radius
    # else:
    #     return None, d
    mid = (tangent_point_A + tangent_point_C) / 2
    # 1e-6~1e-8
    # print(np.sum(np.abs(mid - B)))
    if np.sum(np.abs(mid - B)) / np.linalg.norm(mid) < 1e-6:
        return None, d
    direction = mid - B
    direction = direction / np.linalg.norm(direction) * d * 2
    ltimes, rtimes = 0, 1
    while rtimes - ltimes > 1e-10:
        x1 = (ltimes * 2 + rtimes) / 3
        x2 = (ltimes + rtimes * 2) / 3
        p1 = B + direction * x1
        p2 = B + direction * x2

        def func(p):
            return np.abs(np.dot(p - tangent_point_A, A - B)) + np.abs(
                np.dot(p - tangent_point_C, C - B)
            )

        if func(p1) < func(p2):
            rtimes = x2
        else:
            ltimes = x1
    times = (ltimes + rtimes) / 2
    center = B + direction * times
    radius = np.linalg.norm(tangent_point_A - center)
    return center, radius


def calculate_L2(V_kt=180, c_seconds=5):
    V = V_kt * 0.514444  # 将速度从节 (kt) 转换为米每秒 (m/s)
    return V * c_seconds


# 封装绘制平滑航迹的函数
def plot_smoothed_path(ax, p, offset=0, color="blue", label="定义航迹"):
    last_point = None
    # 计算偏移量
    if offset != 0:
        pxx = p[:, 0]
        pyy = p[:, 1]
        ddx = np.gradient(pxx)
        ddy = np.gradient(pyy)
        normals_ddx = -ddy / np.abs(ddy)
        normals_ddy = ddx / np.abs(ddx)
        ddx_offset = pxx + offset * normals_ddx
        ddy_offset = pyy + offset * normals_ddy
        p = np.stack([ddx_offset, ddy_offset], axis=1)
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
            p1list = [start_point, end_point]
            for _ in range(10):
                p2list = []
                for idx in range(len(p1list) - 1):
                    mid = (p1list[idx] + p1list[idx + 1]) / 2
                    directional = mid - center
                    directional /= np.linalg.norm(directional)
                    directional *= radius
                    mid = center + directional
                    p2list.append(p1list[idx])
                    p2list.append(mid)
                p2list.append(p1list[-1])
                p1list = p2list
            p1list = np.stack(p1list, axis=0)
            ax.plot(
                *[p1list[:, id] for id in range(p1list[0].size)],
                color=color,
                linewidth=LINE_WIDTH,
            )
            offset_points.extend(list(zip(*[p1list[:, id] for id in range(p1list[0].size)])))

        # Draw lines from the previous point to the start of the arc and from the end of the arc to the next point
        if i == 1:
            # Line before the arc
            line_start = A
            line_end = start_point
            ax.plot(
                *[x for x in zip(line_start, line_end)],
                color=color,
                linewidth=LINE_WIDTH,
                label=label,
            )
        else:
            # Line between arcs
            line_start = last_point
            line_end = start_point
            ax.plot(
                *[x for x in zip(line_start, line_end)],
                color=color,
                linewidth=LINE_WIDTH,
            )
        if i == len(p) - 2:
            # Line after the arc
            line_start = end_point
            line_end = C
            ax.plot(
                *[x for x in zip(line_start, line_end)],
                color=color,
                linewidth=LINE_WIDTH,
            )
        last_point = end_point

    return offset_points


def plot_3d_path(input_array, x_values, y_values, z_values, scale=None, offset=0, color="blue", label="定义航迹"):
    """
    绘制三维航迹图

    参数:
    - input_array: 输入的数据数组
    - scale: 轴的缩放比例，默认值为 [2.5, 1.5, 1.2]
    - offset: 航迹的偏移量，默认为 0
    - color: 航迹的颜色，默认为蓝色
    - label: 图例标签，默认为 "三维航迹"
    """

    # 转换为 Mercator 坐标系并计算高度
    if scale is None:
        scale = [2.5, 1.5, 1.2]
    p = convert_to_mercator_with_height(input_array)

    # 创建 3D 图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes((0, 0, 1, 1), projection='3d')

    # 绘制实际航迹的三维散点图
    ax.scatter(x_values, y_values, z_values, color='red', s=5, label='实际航迹', alpha=0.5)

    # 绘制平滑的路径
    plot_smoothed_path(ax, p, offset=offset, color=color, label=label)

    # 设置轴标签
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Height/m')

    # 设置轴的缩放
    ax.ticklabel_format(style='sci', axis='x', scilimits=(7, 7))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))

    # 调整视角和比例
    ax.view_init(elev=25, azim=140)  # elev 控制俯仰角度，azim 控制方位角度
    ax.set_box_aspect(scale)  # 使用传入的缩放比例

    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9), frameon=False)

    # 显示图像
    plt.show()


def setup_plot(ax, x_label, y_label):
    """
    配置绘图对象的基本设置，包括标题、标签、字体大小和坐标轴格式。

    参数：
    ax (matplotlib.axes.Axes): 绘图的轴对象。
    x_label (str): X轴标签。
    y_label (str): Y轴标签。
    """
    # 添加标题和标签
    ax.set_xlabel(x_label, fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE)

    # 调整坐标轴刻度字体大小
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)

    # 调整偏移文本的字体大小
    ax.xaxis.get_offset_text().set_fontsize(OFFSET_TEXT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(OFFSET_TEXT_SIZE)

    # 添加图例，并设置字体大小
    ax.legend(fontsize=LEGEND_FONT_SIZE)


def Splot_paths_scaled(x_values, y_values, p):
    """
    绘制实际航迹的散点图和平滑航迹线，纵坐标放大两倍。
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_all_paths(ax, x_values, y_values, p)
    ax.set_aspect(0.91)  # 将纵坐标放大0.9倍
    setup_plot(ax, "X/m", "Y/m")
    plt.show()


def Nplot_paths_scaled(x_values, y_values, p):
    """
    绘制实际航迹的散点图和平滑航迹线，横坐标加宽。
    """
    fig, ax = plt.subplots(figsize=(FIG_SIZE[0] * 1.26, FIG_SIZE[1]))  # 将宽度加倍
    plot_all_paths(ax, x_values, y_values, p)
    setup_plot(ax, "X/m", "Y/m")
    plt.show()


def plot_paths(x_values, y_values, p):
    """
    绘制实际航迹的散点图和平滑航迹线，包括上下偏移的航迹线。
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_all_paths(ax, x_values, y_values, p)
    setup_plot(ax, "X/m", "Y/m")
    plt.show()
