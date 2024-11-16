import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import AppModel.ILStools as tl


def main():
    # 读取Excel文件
    file_path = r'L:\graduate\AppModel\AppModel\doc\mktSouth.xlsx'
    df = pd.read_excel(file_path)

    # 提取X, Y, HEIGHT坐标
    x = df['X']
    y = df['Y']
    z = df['HEIGHT']
    z *= 0.3048

    # 经纬度数据，用于绘制定义航迹的折线图
    input_array = np.array([
        (111.7459444, 36.16991667, 2700),
        (111.7032222, 36.36719444, 2000),
        (111.5935, 36.35155556, 1500)
    ])

    tl.plot_3d_path(input_array, x, y, z, scale=[1.5, 3, 1.2])


if __name__ == "__main__":
    main()
