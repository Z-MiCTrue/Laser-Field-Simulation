import os

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


# 判断路径是否存在若不存在就生成
def confirm_dir(aim_dir):
    if not os.path.exists(aim_dir):
        os.makedirs(aim_dir)


# 模归一化至01区间
def abs_01(u):
    u_abs = np.abs(u)
    u_min, u_max = np.min(u_abs), np.max(u_abs)
    rate = ((u_abs - u_min) / (u_max - u_min) + 1e-10) / (u_abs + 1e-10)  # 最值归一化比例系数
    u = rate * u
    return u


# 生成模板坐标矩阵
def generate_loc_matrix(kernel_size):
    relative_x, relative_y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
    loc_matrix = (np.concatenate((np.expand_dims(relative_x, axis=-1),
                                  np.expand_dims(relative_y, axis=-1)), axis=-1)) - (kernel_size - 1) / 2
    return loc_matrix


def show_img(img, img_title=''):
    plt.clf()
    plt.title(img_title)
    plt.imshow(img, origin='lower')
    plt.colorbar()
    plt.savefig(f'./dataset/{img_title}')
    # plt.show()


class Resonant_Cavity:
    def __init__(self):
        # --- Config --- #
        # 波长
        self.light_lambda = 5.12e-7  # 512 nm
        # 腔长
        self.L = 1.92e1
        # 卷积核大小
        self.kernel_size = 127
        # 初始光场 1pixel跨越0.1mm
        self.u_i = np.ones((128, 128), dtype=np.float32)  # 均匀
        # self.u_i = np.tril(np.ones((128, 128), dtype=np.double))  # 三角
        # self.u_i = np.random.rand(128, 128)  # 随机
        assert self.u_i.shape[0] == self.u_i.shape[1], 'shape of u_i must be a square'
        # 是否为平行平面腔
        self.with_f = True

        # --- Auto --- #
        # 菲涅尔衍射卷积核
        self.fresnel_kernel = self.generate_fresnel_kernel(self.kernel_size)
        # 掩膜
        self.mask = np.zeros(self.u_i.shape, dtype=np.float32)
        h, w = self.mask.shape[:2]
        self.mask[h // 4: h // 4 * 3, w // 4: w // 4 * 3] = 1
        # 等效共焦透镜的相位变化矩阵
        self.phase_change_matrix = self.generate_phase_change_matrix(self.u_i.shape[0])

    # 生成菲涅尔衍射卷积核
    def generate_fresnel_kernel(self, kernel_size):
        assert kernel_size % 2, 'size must be an odd number'
        k_2z = np.pi / (self.light_lambda * self.L)
        fresnel_kernel = generate_loc_matrix(kernel_size)
        fresnel_kernel = (np.power(fresnel_kernel[:, :, 0], 2) + np.power(fresnel_kernel[:, :, 1], 2)) * 1e-8
        fresnel_kernel = np.exp(1j * k_2z * fresnel_kernel)
        return fresnel_kernel

    # 生成等效共焦透镜的相位变化矩阵
    def generate_phase_change_matrix(self, matrix_size):
        k_2f = np.pi / (self.light_lambda * self.L)
        pc_matrix = generate_loc_matrix(matrix_size)
        pc_matrix = (np.power(pc_matrix[:, :, 0], 2) + np.power(pc_matrix[:, :, 1], 2)) * 1e-8
        pc_matrix = np.exp(-1j * k_2f * pc_matrix)
        return pc_matrix

    # 一次腔长传播并反射
    def forward(self, u):
        k_z = (2 * np.pi * self.L) / self.light_lambda
        u = (np.exp(1j * k_z) / 1j) * convolve2d(u, self.fresnel_kernel, mode='same', boundary="fill", fillvalue=0)
        u = abs_01(u)
        u = u * self.mask
        if self.with_f:
            u = u * self.phase_change_matrix
        return u


def main():
    confirm_dir('./dataset/')
    resonant_cavity = Resonant_Cavity()
    show_img(resonant_cavity.u_i, 'epoch--0')
    u = resonant_cavity.u_i
    for i in range(1000):
        u = resonant_cavity.forward(u)
        if i % 1 == 0:
            u_abs = np.abs(u)
            show_img(u_abs, f'epoch--{i + 1}')


if __name__ == '__main__':
    main()
