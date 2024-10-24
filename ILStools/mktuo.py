import pandas as pd
import ILStools.ILStools as tl

# 文件路径
file_path = '../doc/mktNorth.xlsx'

# 读取Excel文件
print("读取 Excel 文件...")
df = pd.read_excel(file_path)

# 初始化新列X和Y
df['X'] = 0.0
df['Y'] = 0.0

print("开始转换经纬度为墨卡托投影坐标...")
# 处理每一行的经纬度并计算墨卡托投影坐标
for index, row in df.iterrows():
    lon = row['LON']
    lat = row['LAT']

    # 打印当前处理的经纬度
    print(f"正在处理第 {index + 1} 行: 经度={lon}, 纬度={lat}")

    x, y = tl.latlon_to_mercator(lon, lat)

    # 打印转换后的墨卡托坐标
    print(f"转换结果: X={x}, Y={y}")

    # 将X, Y坐标写入新列
    df.at[index, 'X'] = x
    df.at[index, 'Y'] = y

# 将处理后的数据保存回同一个Excel文件
print(f"将结果保存回 Excel 文件: {file_path}")
df.to_excel(file_path, index=False)

print("处理完成！")
