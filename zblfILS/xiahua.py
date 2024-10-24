import math
import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.family'] = 'Microsoft YaHei'
# 定义常量和函数
# 定义常量和函数
k = 0.117
sin_3_deg = np.sin(np.deg2rad(3))

def calculate_DDM(a):
    return 4 * k * np.cos(np.pi * np.sin(np.deg2rad(a)) / (2 * sin_3_deg))
# 生成数据
a_values = np.linspace(0, 6, 2000)  # 从0°到24°生成1000个数据点
DDM_values = calculate_DDM(a_values)

# 绘制图形
plt.plot(a_values, DDM_values,linewidth=3)
plt.xlabel('下滑角 (°)')
plt.ylabel('DDM')
plt.title('下滑角与DDM关系图')
plt.grid(True)

# 设置横轴刻度
plt.xticks(np.arange(0, 7, 1))

plt.show()
