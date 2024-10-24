import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 地球半径（以米为单位，WGS 84标准）
R = 6378137

# 将经纬度（以度为单位）转换为墨卡托投影直角坐标
def latlon_to_mercator(longitude, latitude):
    # 将经纬度转换为弧度
    lon_rad = math.radians(longitude)
    lat_rad = math.radians(latitude)

    # 计算墨卡托投影的x和y坐标
    x = R * lon_rad
    y = R * math.log(math.tan(math.pi / 4 + lat_rad / 2))

    return x, y

# 经纬度数据
coordinates = [
    (111.59349999999999, 36.351555555555556),
    (111.77055555555556, 36.37672222222222),
    (111.81224999999999, 36.43702777777777),
    (111.91180555555556, 36.59644444444444),
    (111.70322222222222, 36.367194444444444)
]

# 转换为墨卡托投影坐标
mercator_coordinates = [latlon_to_mercator(lon, lat) for lon, lat in coordinates]

# 对坐标按照X值进行排序
sorted_coordinates = sorted(mercator_coordinates, key=lambda coord: coord[0])

# 分别提取排序后的X和Y坐标
sorted_x_values = [coord[0] for coord in sorted_coordinates]
sorted_y_values = [coord[1] for coord in sorted_coordinates]

# 设置插值区间
x_min, x_max = 1.2440 * 1e7, 1.2444 * 1e7

# 将数据分为需要插值的部分和不需要插值的部分
x = np.array(sorted_x_values)
y = np.array(sorted_y_values)

# 分别提取需要插值的部分和不需要插值的部分
mask = (x >= x_min) & (x <= x_max)
x_to_smooth = x[mask]
y_to_smooth = y[mask]

# 对插值范围内的数据进行插值
if len(x_to_smooth) <= 1:
    # 如果范围内只有一个点或者没有点，增加一个虚拟点进行插值
    if len(x_to_smooth) == 1:
        x_virtual = x_to_smooth[0] + 1500  # 在当前x的基础上增加一个小值
        y_virtual = y_to_smooth[0] + 1500  # 在当前y的基础上增加一个小值
        x_to_smooth = np.append(x_to_smooth, x_virtual)
        y_to_smooth = np.append(y_to_smooth, y_virtual)
    else:
        # 如果没有点，保持原样
        x_smooth = x_to_smooth
        y_smooth = y_to_smooth

# 如果有两个及以上的点则进行样条插值
if len(x_to_smooth) > 1:
    # 使用样条插值使曲线更加平滑
    x_smooth = np.linspace(x_to_smooth.min(), x_to_smooth.max(), 1000)
    spline = make_interp_spline(x_to_smooth, y_to_smooth, k=3)  # 三次样条插值
    y_smooth = spline(x_smooth)
else:
    # 如果范围内的数据点不足，保持原样
    x_smooth = x_to_smooth
    y_smooth = y_to_smooth

# 构造最终的x和y数组
x_final = np.concatenate((x[~mask], x_smooth))
y_final = np.concatenate((y[~mask], y_smooth))

# 按照x值对最终的数组进行排序
sorted_indices = np.argsort(x_final)
x_final = x_final[sorted_indices]
y_final = y_final[sorted_indices]

# 清空当前绘图对象，防止多条线混在一起
plt.clf()

# 绘制平滑后的折线图，减小数据点大小和增加线条宽度
plt.plot(x_final, y_final, marker='o', linestyle='-', color='b', markersize=2, linewidth=3)

# 添加标题和标签
plt.title('Mercator Projection Line Plot (Partially Smoothed by Cubic Spline Interpolation)')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')

# 显示图像
plt.show()
