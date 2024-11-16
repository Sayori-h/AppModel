# constant.py

# 图形尺寸
FIG_SIZE = (12, 10)

# 实际航迹的散点图配置
SCATTER_COLOR = 'red'
SCATTER_SIZE = 20
SCATTER_ALPHA = 0.5
SCATTER_LABEL = '实际航迹'

# 平滑航迹线的配置
LINE_WIDTH = 4
LINE_COLORS = {
    'original': 'blue',
    'offset_up': 'green',
    'offset_down': 'green'
}
LINE_LABELS = {
    'original': '定义航迹',
    'offset_up': '+1nm包络线',
    'offset_down': '-1nm包络线'
}

OFFSET_UP = 1470
OFFSET_DOWN = -1470

# 字体大小
FONT_SIZE = 18

# 坐标轴配置
TICK_LABEL_SIZE = 16
OFFSET_TEXT_SIZE = 16

# 图例字体大小
LEGEND_FONT_SIZE = 14

# 转弯半径
d = 1852

R = 6378137  # 地球半径，墨卡托投影使用的常数
