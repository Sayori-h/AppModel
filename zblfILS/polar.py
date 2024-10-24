import numpy as np
import matplotlib.pyplot as plt


# 定义函数
def f(a):
    return 2 * np.abs(np.sin((np.pi * np.cos(a)) / 2))


# 生成角度值
theta = np.linspace(0, 2 * np.pi, 1000)

# 计算函数值
r = f(theta)

# 绘制极坐标图
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r, label=r'${f_{a} (\theta ,\varphi )=2\sin\left(\frac{\pi d}{\lambda }\cos\theta\right)}$')

# 显示极坐标刻度
ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
ax.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                    r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$'],
                   fontsize=15)

# 添加网格
ax.grid(True)

# 显示图例
ax.legend(loc='upper left')

# 显示图形
plt.show()
