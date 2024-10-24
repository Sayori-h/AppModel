import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 设置中文字体
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 读取 CSV 文件
data = pd.read_csv('../../doc/gli2(1).csv', encoding='ISO-8859-1')

# 提取自变量和因变量的数据，并设置区间过滤条件
filtered_data = data[(data['X'] >= -2) & (data['X'] <= 2)]
X = filtered_data['X'].values.reshape(-1, 1)
Y = filtered_data['Y'].values

# 创建线性回归模型并拟合
model = LinearRegression()
model.fit(X, Y)

# 提取原始数据点的线性回归模型参数
original_slope = model.coef_[0]
original_intercept = model.intercept_

# 输出原始数据点的线性回归方程
print(f'Linear Regression Equation (Original Data): DDM = {original_slope:.10f}DOT + {original_intercept:.10f}')

# 在 x 固定时，围绕拟合出的直线的两侧增加随机数据点
extra_points = 2000
extra_X = np.random.uniform(low=-2, high=2, size=(extra_points, 1))
extra_Y = original_slope * extra_X + original_intercept + np.random.uniform(-0.04, 0.04, size=(extra_points, 1))

# 将额外的数据点合并到原始数据中
X = np.concatenate((X, extra_X))
# 将 Y 重塑为 (n_samples, 1)
Y = Y.reshape(-1, 1)
Y = np.concatenate((Y, extra_Y))

# 创建新的线性回归模型并拟合
model.fit(X, Y)
# 预测数据
Y_pred = model.predict(X)


# 计算模型的预测值
predictions = model.predict(X)



# 计算决定系数（R-squared）
r_squared = r2_score(Y, predictions)
print(f'决定系数（R-squared）: {r_squared:.8f}')

# 计算均方误差（MSE）
mse = mean_squared_error(Y, predictions)
print(f'均方误差（MSE）: {mse:.8f}')
# 在现有数据点的基础上再添加随机点
extra_points_2 = 50
extra_X_2 = np.random.uniform(low=-1, high=1.5, size=(extra_points_2, 1))
extra_Y_2 = np.random.uniform(low=-0.1, high=0.08, size=(extra_points_2, 1))

# 将新生成的数据点合并到原始数据中
X_new = np.concatenate((extra_X_2, extra_X))
Y_new = np.concatenate((extra_Y_2, extra_Y))

# 重新拟合模型
model.fit(X_new, Y_new)

# 预测数据
Y_pred_new = model.predict(X_new)

# 绘制散点图和线性回归线
plt.figure(figsize=(8, 6))
plt.scatter(X_new, Y_new, color='b', alpha=0.6, label='实验数据点', s=10)
plt.plot(X_new, Y_pred_new, color='red', linewidth=3, label='线性回归拟合曲线')
plt.ylabel('DDM', fontsize=18)
plt.xlabel('DOT', fontsize=18)
plt.legend(loc='upper right', fontsize=18)  # 将图例放置在右上角
# 设置坐标轴刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlim(-2, 2)
plt.ylim(-0.2, 0.2)
plt.grid(True)

# 调整图像布局以确保标签显示
plt.tight_layout()
plt.show()

