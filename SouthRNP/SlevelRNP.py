import ILStools.ILStools as tl

file_path = '../doc/mktSouth.xlsx'
# 调用封装的函数加载坐标数据
x_values, y_values = tl.load_coordinates(file_path)

# 经纬度数据
coordinates = [
    (111.7459444, 36.16991667),
    (111.7032222, 36.36719444),
    (111.5935, 36.35155556)
]

# 调用坐标转换函数
p = tl.convert_coordinates(coordinates)

# 调用绘图函数
tl.plot_paths(x_values, y_values, p)
