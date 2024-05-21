import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机样本的数量
R = 1  # 圆的半径
num_samples = 5000

def simple_circle_sampling(R, num_samples):
    # 简单圆面采样
    x_list = []
    y_list = []
    for _ in range(num_samples):
        random_theta = random.random() * math.pi * 2 # U(0,1) -> [0,2𝜋]
        random_r = random.random() * R # U(0,1) -> [0,r]
        x = math.cos(random_theta) * random_r
        y = math.sin(random_theta) * random_r
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list

def uniform_circle_sampling(R, num_samples):
    #均匀圆面采样
    x_list = []
    y_list = []
    for _ in range(num_samples):
        random_theta = random.random() * math.pi * 2 # 2𝜋θ
        random_r = math.sqrt(random.random()) * R # R·sqrt{r}
        x = math.cos(random_theta) * random_r
        y = math.sin(random_theta) * random_r
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list

def simple_sphere_sampling(R, num_samples):
    # 简单球面采样
    x_list = []
    y_list = []
    z_list = []
    for _ in range(num_samples):
        random_theta = random.random() * math.pi # U(0,1) -> [0,𝜋]
        random_phi = random.random() * math.pi * 2 # U(0,1) -> [0,2𝜋]
        x = math.sin(random_theta) * math.cos(random_phi) * R
        y = math.sin(random_theta) * math.sin(random_phi) * R
        z = math.cos(random_theta) * R
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list

def uniform_sphere_sampling(R, num_samples):
    # 均匀球面采样
    x_list = []
    y_list = []
    z_list = []
    for _ in range(num_samples):
        random_theta = math.acos(2 * random.random() - 1) # acos(2θ-1)
        random_phi = random.random() * math.pi * 2 # 2𝜋𝜑
        x = math.sin(random_theta) * math.cos(random_phi) * R
        y = math.sin(random_theta) * math.sin(random_phi) * R
        z = math.cos(random_theta) * R
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list

def cosine_sphere_sampling(R, num_samples):
    # 余弦球面采样
    x_list = []
    y_list = []
    z_list = []
    for _ in range(num_samples):
        random_theta = math.acos(1 - 2 * random.random()) / 2 # 0.5 * acos(1-2θ)
        random_phi = random.random() * math.pi * 2 # 2𝜋𝜑
        x = math.sin(random_theta) * math.cos(random_phi) * R
        y = math.sin(random_theta) * math.sin(random_phi) * R
        z = math.cos(random_theta) * R
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list

# base = 2 -> 二进制
def van_der_corput(bits : int) -> float:
    # 分治算法 - 按位反转一个无符号32位整数
    bits = (bits >> 16) | (bits << 16)
    bits = ((bits & 0xFF00FF00) >> 8) | ((bits & 0x00FF00FF) << 8)
    bits = ((bits & 0xF0F0F0F0) >> 4) | ((bits & 0x0F0F0F0F) << 4)
    bits = ((bits & 0xCCCCCCCC) >> 2) | ((bits & 0x33333333) << 2)
    bits = ((bits & 0xAAAAAAAA) >> 1) | ((bits & 0x55555555) << 1)
    return float(bits) * 2.3283064365386963e-10  # 除以2^32,将32位整数转换成0~1浮点数

def harmmersley(i : int, N : int) -> tuple[float, float]:
    x = float(i) / float(N)
    y = van_der_corput(i)
    return x, y

def harmmersley_uniform_sphere_sampling(R, num_samples):
    # 均匀球面采样 + 低差异序列
    x_list = []
    y_list = []
    z_list = []
    for index in range(num_samples):
        random_1, random_2 = harmmersley(index, num_samples)
        random_theta = math.acos(2 * random_1 - 1) # acos(2θ-1)
        random_phi = random_2 * math.pi * 2 # 2𝜋𝜑
        x = math.sin(random_theta) * math.cos(random_phi) * R
        y = math.sin(random_theta) * math.sin(random_phi) * R
        z = math.cos(random_theta) * R
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list

# 2D 
# x_list, y_list = simple_circle_sampling(R, num_samples)
# x_list, y_list = uniform_circle_sampling(R, num_samples)

# plt.figure(figsize=(8, 8))
# plt.scatter(x_list, y_list, s=5, color='blue', alpha=0.6)
# plt.title('Random Sampling in a 2D Circle')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.axis('equal')  # 保持比例一致
# plt.grid(True)
# plt.savefig('image.png', dpi=300)
# plt.show()

# 3D
# x_list, y_list, z_list = simple_sphere_sampling(R, num_samples)
# x_list, y_list, z_list = uniform_sphere_sampling(R, num_samples)
# x_list, y_list, z_list = cosine_sphere_sampling(R, num_samples)
x_list, y_list, z_list = harmmersley_uniform_sphere_sampling(R, num_samples)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_list, y_list, z_list, s=5, color='blue', alpha=0.6)
ax.set_title('Random Sampling on a 3D Sphere')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1,1,1])  # Equal aspect ratio
plt.savefig('image.png', dpi=300)
plt.show()
