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

# 定义线性回归方程的系数
slope = -0.0875186354
intercept = 0.0000103389

# 创建新的y轴刻度标签
def transform_y_ticks(y, pos):
    return '{:.2f}'.format((y - intercept) / slope)

# 绘制散点图和线性回归线
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='b', alpha=0.6, label='实验数据点', s=10)
plt.plot(X, Y_pred, color='red', linewidth=3, label='线性回归拟合曲线')

plt.ylabel('DDM')
plt.xlabel('DOT')
plt.legend()
plt.xlim(-2, 2)

# 设置自定义y轴刻度
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(transform_y_ticks))


plt.grid(True)
plt.show()
