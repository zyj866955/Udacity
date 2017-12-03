from fractions import Fraction
import numpy as np
from helper import transpose, matxMultiply, generatePoints
from work import augmentMatrix, swapRows, scaleRow, addScaledRow

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
    size = len(A)
    for i in range(size):
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

    return [[round(i[size], decPts)] for i in augmented_matrix]

# r = np.random.randint(low=3, high=10)
# a = np.random.randint(low=-10, high=10, size=(r, r))
# b = np.arange(r).reshape((r, 1))
a = [[-5, 9, 5, 9], [-3, 4, 4, 5], [8, -10, -2, -2], [8, -8, -8, -7]]
b = [[1],[1],[1],[1]]
x = gj_Solve(a, b, epsilon=1.0e-8)
print(x)
