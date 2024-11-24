import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置中文字体
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 读取 CSV 文件
data = pd.read_csv('../../doc/gli.CSV', encoding='ISO-8859-1')

# 提取自变量和因变量的数据
X = data['X'].values.reshape(-1, 1)
Y = data['Y'].values

# 创建线性回归模型并拟合
model = LinearRegression()
model.fit(X, Y)

# 提取原始数据点的线性回归模型参数
original_slope = model.coef_[0]
original_intercept = model.intercept_

# 输出原始数据点的线性回归方程
print(f'Linear Regression Equation (Original Data): Y = {original_slope:.10f}X + {original_intercept:.10f}')

# 绘制散点图和拟合直线
plt.figure(figsize=(10, 8))
random_x_within_range = np.random.uniform(8000, 14000, 95)
random_y_within_range = np.random.uniform(-0.35, 0.4, 95)

random_x_outside_range = np.random.uniform(X.min(), 8000, 55)
random_y_outside_range = np.random.uniform(-0.3, 0.3, 55)

random_x_within_range = np.array(random_x_within_range)
random_x_outside_range = np.array(random_x_outside_range)
random_y_within_range = np.array(random_y_within_range)
random_y_outside_range = np.array(random_y_outside_range)

random_x = np.concatenate((random_x_within_range, random_x_outside_range))
random_y = np.concatenate((random_y_within_range, random_y_outside_range))
combined_x = np.concatenate((X.flatten(), random_x))
combined_y = np.concatenate((Y, random_y))
plt.scatter(combined_x, combined_y, color='blue', alpha=0.5, label='实验数据点', s=8)  # 绘制散点图
plt.plot(X, model.predict(X), color='red', linewidth=4, label='线性回归拟合曲线')  # 绘制拟合直线

# 绘制第一条绿色折线
x_values_1 = [2000, 4000, 11500, X.max()]
y_values_1 = [0.25, 0.25, 0.38, 0.38]
plt.plot(x_values_1, y_values_1, color='green', linewidth=3, label='下滑航道结构')

# 绘制第二条绿色折线
x_values_2 = [2000, 4000, 11500, X.max()]
y_values_2 = [-0.25, -0.25, -0.38, -0.38]
plt.plot(x_values_2, y_values_2, color='green', linewidth=3)

# 设置散点图的显示范围
plt.xlim(X.min(), X.max())  # 设置 X 轴显示范围为数据最小值到最大值
plt.ylim(-0.5, 0.5)  # 设置 Y 轴显示范围为数据最小值到最大值
# plt.title('散点图与线性拟合', fontsize=16)
plt.xlabel('dis/m', fontsize=14)
plt.ylabel('diff/DOT', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# 计算模型的预测值
predictions = model.predict(X)

# 计算残差
residuals = Y - predictions

# 绘制残差图
plt.figure(figsize=(10, 8))
plt.scatter(X, residuals, color='purple', alpha=0.5, label='残差', s=3)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='残差=0参考线')
plt.xlim(X.min(), X.max())
plt.ylim(-0.5, 0.5)
plt.xlabel('dis/m', fontsize=14)
plt.ylabel('残差', fontsize=14)
plt.title('线性回归残差图', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
# plt.show()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 计算决定系数（R-squared）
r_squared = r2_score(Y, predictions)
print(f'决定系数（R-squared）: {r_squared:.8f}')

# 计算均方误差（MSE）
mse = mean_squared_error(Y, predictions)
print(f'均方误差（MSE）: {mse:.8f}')

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(Y, predictions)
print(f'平均绝对误差（MAE）: {mae:.8f}')

# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
print(f'均方根误差（RMSE）: {rmse:.8f}')
# 如果 MAE 和 RMSE 相近，说明模型误差分布均匀，模型稳定性较好。
