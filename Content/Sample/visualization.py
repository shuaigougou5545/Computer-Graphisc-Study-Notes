import random
import math
import numpy as np
import matplotlib.pyplot as plt

# 设置随机样本的数量
r = 1  # 圆的半径
num_samples = 10000

def generate_random_samples_in_circle(r, num_samples):
    x_list = []
    y_list = []
    for _ in range(num_samples):
        random_theta = random.random() * math.pi * 2
        random_r = random.random() * r
        x = math.cos(random_theta) * random_r
        y = math.sin(random_theta) * random_r
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


x_list, y_list = generate_random_samples_in_circle(r, num_samples)

# 创建图形并绘制采样点
plt.figure(figsize=(8, 8))
plt.scatter(x_list, y_list, s=5, color='blue', alpha=0.6)
plt.title('Random Sampling in a 2D Circle')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')  # 保持比例一致
plt.grid(True)
plt.show()
