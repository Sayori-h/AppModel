import math
import matplotlib.pyplot as plt

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
    (111.7459444, 36.16991667),
    (111.7032222, 36.36719444),
    (111.5935, 36.35155556)
]

# 转换为墨卡托投影坐标
mercator_coordinates = [latlon_to_mercator(lon, lat) for lon, lat in coordinates]

# 分别提取X和Y坐标
x_values = [coord[0] for coord in mercator_coordinates]
y_values = [coord[1] for coord in mercator_coordinates]

# 清空当前绘图对象，防止多条线混在一起
plt.clf()

# 绘制折线图
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')

# 添加标题和标签
plt.title('Mercator Projection Line Plot')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')

# 显示图像
plt.show()
