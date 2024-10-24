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
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', alpha=0.5, label='实验数据点', s=3)  # 绘制散点图
plt.plot(X, model.predict(X), color='red', linewidth=4, label='线性回归拟合曲线')  # 绘制拟合直线

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
plt.figure(figsize=(10, 6))
plt.scatter(X, residuals, color='purple', alpha=0.5, label='残差', s=3)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='残差=0参考线')
plt.xlim(X.min(), X.max())
plt.ylim(-0.5, 0.5)
plt.xlabel('dis/m', fontsize=14)
plt.ylabel('残差', fontsize=14)
plt.title('线性回归残差图', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

from sklearn.metrics import r2_score, mean_squared_error

# 计算决定系数（R-squared）
r_squared = r2_score(Y, predictions)
print(f'决定系数（R-squared）: {r_squared:.8f}')

# 计算均方误差（MSE）
mse = mean_squared_error(Y, predictions)
print(f'均方误差（MSE）: {mse:.8f}')
