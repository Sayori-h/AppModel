import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import AppModel.ILStools as tl


def main():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体来支持中文字符
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 读取Excel文件
    file_path = r'L:\graduate\AppModel\AppModel\doc\mktNorth.xlsx'  # 请确保文件路径正确
    df = pd.read_excel(file_path)

    # 提取X, Y, HEIGHT坐标
    x = df['X']
    y = df['Y']
    z = df['HEIGHT']

    # 经纬度数据，用于绘制定义航迹的折线图
    input_array = np.array([
        (111.81224999999999, 36.43702777777777, 2700),
        (111.77055555555556, 36.37672222222222, 2300),
        (111.70322222222222, 36.367194444444444, 2000),
        (111.59349999999999, 36.351555555555556, 1500)
    ])

    p = tl.convert_to_mercator_with_height(input_array)

    tl.plot_3d_path(input_array, x, y, z)


if __name__ == "__main__":
    main()
