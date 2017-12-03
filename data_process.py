# TODO 实现 Gaussain Jordan 方法求解 Ax = b
import numpy as np
from fractions import Fraction
from work import swapRows, scaleRow, addScaledRow, get_col, augmentMatrix

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16

    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""


def find_max_element(M, col):
    temp = 0
    for i in range(col, len(M)):
        if abs(M[i][col]) > abs(temp):
            temp = M[i][col]
    return i, temp


def find_max_element_v2(col, r):
    temp = 0
    index = 0
    for i in range(r, len(col)):
        if abs(col[i]) >= abs(temp):
            temp = col[i]
            index = i
        elif abs(col[i]) == abs(temp):
            temp = col[i]
    return index, temp


def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    if len(A) != len(b):
        return None
    augmented_matrix = augmentMatrix(A, b)
    print(augmented_matrix)
    size = len(A)
    for i in range(size):
        print("开始对第{}列进行转换".format(i))
        col = []
        for j in range(size):
            col.append(augmented_matrix[j][i])
        index, max_element = find_max_element_v2(col, i)
        if max_element == 0:
            return None
        # 与对角线元素进行行交换
        if index != i:
            swapRows(augmented_matrix, i, index)
        scaleRow(augmented_matrix, i, Fraction(1, max_element))
        for m in range(size):
            if m == i:
                continue
            addScaledRow(augmented_matrix, m, i, -augmented_matrix[m][i])
    print(augmented_matrix)
    result = get_col(augmented_matrix, len(augmented_matrix[0]) - 1)
    return [[result[round(i)]] for i in range(len(result))]


if __name__ == '__main__':
    r = np.random.randint(low=3, high=10)
    A = np.random.randint(low=-10, high=10, size=(r, r))
    b = np.arange(r).reshape((r, 1))
    # print(A)
    # for i in range(len(A)):
    #     col = []
    #     for j in range(len(A)):
    #         col.append(A[j][i])
    #     index, max_element = find_max_element_v2(col, i)
    #     print(index, max_element)

    x = gj_Solve(A.tolist(), b.tolist(), epsilon=1.0e-8)
    print(x)
    Ax = np.dot(A, np.array(x))
    loss = np.mean((Ax - b) ** 2)
    print(loss)
