import AppModel.ILStools as tl


def main():
    file_path = r'L:\graduate\AppModel\AppModel\doc\mktNorth.xlsx'
    # 调用封装的函数加载坐标数据
    x_values, y_values = tl.load_coordinates(file_path)

    coordinates = [
        (111.59349999999999, 36.351555555555556),
        (111.77055555555556, 36.37672222222222),
        (111.81224999999999, 36.43702777777777),
        (111.70322222222222, 36.367194444444444)
    ]
    # 调用坐标转换函数
    p = tl.convert_coordinates(coordinates)

    # 调用绘图函数
    tl.Nplot_paths_scaled(x_values, y_values, p)


if __name__ == "__main__":
    main()
