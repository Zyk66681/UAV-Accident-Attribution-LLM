import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体，确保图表显示正常
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 适用，Mac 可改用 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 事故类型分布 
# ==========================================
print("生成事故类型分布图...")
labels = ['人为因素主导型', '设备故障型', '环境干扰型', '管理缺失型', '复合型']
counts = [139, 402, 150, 23, 499] # 论文表3数据
percentages = [11.5, 33.1, 12.4, 1.9, 41.1] # 论文表3数据

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, counts, color=sns.color_palette("pastel"))
plt.title('事故类型分布统计 (N=1213)', fontsize=14)
plt.ylabel('频次', fontsize=12)

# 在柱状图上添加百分比标签
for bar, pct in zip(bars, percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{pct}%', ha='center', va='bottom')
plt.savefig('accident_type_distribution.png', dpi=300)
plt.show()

# ==========================================
# 2. 地理分布 (对应论文1.2.3节)
# ==========================================
print("生成地理分布饼图...")
geo_labels = ['北美', '欧洲', '东亚', '其他']
geo_sizes = [38.2, 29.5, 15.1, 100 - (38.2 + 29.5 + 15.1)] # 论文数据
explode = (0.05, 0, 0, 0)

plt.figure(figsize=(8, 8))
plt.pie(geo_sizes, explode=explode, labels=geo_labels, autopct='%1.1f%%', 
        shadow=True, startangle=140, colors=sns.color_palette("Set2"))
plt.title('事故地理分布 (共覆盖62个国家和地区)', fontsize=14) # 论文数据
plt.savefig('geographic_distribution.png', dpi=300)
plt.show()
