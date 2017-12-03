from copy import deepcopy


def matxRound(M, decPts=4):
    for i in range(len(M)):
        for j in range(len(M[i])):
            M[i][j] = round(M[i][j], decPts)


# 矩阵的转置
def transpose(M):
    return [[i[j] for i in M] for j in range(len(M[0]))]


# 交换行
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]


# 给行乘以scale,如果r为空，默认给矩阵所有元素都乘以scale
def scaleRow(M, r, scale=1):
    if 0 == scale:
        raise ValueError
    for i in range(len(M[r])):
        try:
            M[r][i] = M[r][i] * scale
        except TypeError as e:
            pass


# 给行乘以系数，加到另一行
def addScaledRow(M, r1, r2, scale):
    for i in range(len(M[r1])):
        M[r1][i] = M[r1][i] + M[r2][i] * scale


# 矩阵乘法
def matxMultiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError
    return [[sum(map(lambda x, y: x * y, i, j)) for j in transpose(B)] for i in A]


# 曾广矩阵
def augmentMatrix(A, b):
    return [A[i] + b[i] for i in range(len(A))]


# 查找序列中最大元素，返回值和索引
def find_max_element(term):
    temp = 0
    for i in range(len(term)):
        if abs(term[i]) > abs(temp):
            temp = term[i]
    return i, temp


# 计算矩阵的行列式
def calculate_determinant(M):
    size = len(M)
    if size == 1:
        return M[0][0]
    else:
        ix = jx = d = 0
        for i in M:
            while (jx + 1 <= size):
                m = subset(M, ix, jx)
                d += (-1) ** (ix - jx) * M[ix][jx] * calculate_determinant(m)
                jx += 1
            return d


def det(m):
    if len(m) <= 0:
        return None
    elif len(m) == 1:
        return m[0][0]
    else:
        s = 0
    for i in range(len(m)):
        n = [[row[a] for a in range(len(m)) if a != i] for row in m[1:]]  # 这里生成余子式
        s += m[0][i] * det(n) * (-1) ** (i % 2)
    return s


# 按照行列获取子集
def subset(M, i, j):
    M = deepcopy(M)
    del M[i]
    for m in M:
        del m[j]
    return M


# 判断是否是奇异矩阵
def is_singular_matrix(M):
    if calculate_determinant(M) == 0:
        return True
    return False


# 获取列
def get_col(M, col=None):
    return [i[col] for i in M]


# 消元法
def solve(augmented_matrix):
    size = len(augmented_matrix)
    for i in range(size):
        print("第{}次".format(i))
        if i < size:
            for j in range(i + 1, size):
                #                 index, value = find_max_element(augmented_matrix, i)
                scale = (augmented_matrix[j][i] / augmented_matrix[i][i])
                addScaledRow(augmented_matrix, j, i, -scale)

    scale = (1 / augmented_matrix[size-1][size-1])
    scaleRow(augmented_matrix, size-1, scale)


def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    if len(A) != len(b):
        return None
    if is_singular_matrix(A):
        return None
    augmented_matrix = augmentMatrix(A, b)
    solve(augmented_matrix)
    matxRound(augmented_matrix, decPts)
    print(augmented_matrix)
    result = get_col(augmented_matrix, len(augmented_matrix[0]) - 1)
    return [[result[i]] for i in range(len(result))]


def gj_SolveV2(A, b, decPts=4, epsilon=1.0e-16):
    if len(A) != len(b):
        return None
    if is_singular_matrix(A):
        return None
    augmented_matrix = augmentMatrix(A, b)
    for i in range(len(A)):
        current_col = get_col(A, i)
        index, value = find_max_element(current_col)
        if value == 0:
            continue
        for j in range(0, len(A) - i):
            scale = current_col[j] / value
            if index == 0:
                addScaledRow(augmented_matrix, j, i, -scale)
            else:
                swapRows(augmented_matrix, 0, index)
                addScaledRow(augmented_matrix, j, i, -scale)
    matxRound(augmented_matrix, decPts)
    result = get_col(augmented_matrix, len(augmented_matrix[0]) - 1)
    return [[result[i]] for i in range(len(result))]


# 获取伴随矩阵
def adjoint_matrix(M):
    size = len(M)
    return [[det(subset(M, j, i)) for j in range(size)] for i in range(size)]


# 获取逆矩阵
def inverse_matrix(m):
    deter = det(m)
    matrix = adjoint_matrix(m)
    scaleRow(matrix, scale=1 / deter)
    return matrix


def linearRegression(X, Y):
    Y = [[i] for i in Y]
    size = len(X)
    single = [[1] * size] * size
    for i in range(len(X)):
        single[i][0] = X[i]
    X = single
    tran_matrix = transpose(X)
    print(tran_matrix)
    Xt = matxMultiply(tran_matrix, X)
    print(Xt)
    inverse_mat = inverse_matrix(Xt)
    inverse_mat_mul_tran = matxMultiply(inverse_mat, tran_matrix)
    h = matxMultiply(inverse_mat_mul_tran, Y)
    print(h)
    return h[0][0], h[1][0]


import numpy as np

# m,b = linearRegression(X,Y)
# print(m,b)
if __name__ == '__main__':
    c = [[7, -9, 6, 3, -10, 3, -10, -3], [-7, 3, 9, -8, -3, -8, 7, 7], [8, 5, -6, 0, -4, 0, -1, 7],
         [-8, -4, -4, 1, -3, 3, 3, 8],
         [-4, -2, 6, -3, -1, 0, -1, -8], [0, 7, 9, 1, -1, 9, -4, 9], [-10, 7, 7, -4, 7, 5, -7, -1],
         [5, 2, 2, 7, -6, 7, -4, 0]]
    p = [[0], [1], [2], [3], [4], [5], [6], [7]]
    m = [[-10, 3, 4], [4, 5, 6], [-1, 5, -9]]

    X = [-3.0388846783056445, -2.2205669355759805, 3.0026770138937344, 3.9848177123752517, -2.1262775940438647,
         -0.9272671284026952,
         -1.7265182734055817, 4.461181568138308, 4.276551645769736, -3.848693613685227, -3.4514920215851275,
         4.52871798639497,
         2.3662177943079685, -1.5235340317529422, -4.092823249946128, 2.995537615320922, -0.026147353789510497,
         -1.575861537975741,
         3.826047967257665, -3.2641675338514284, -4.8106303113184845, 4.188585985122387, 3.578132703044828,
         -1.954819486991235,
         -3.398869533436475, 4.4572046748416145, 0.5917019501466392, -4.2261027538811335, 1.395228405895966,
         2.697580737438038,
         0.43904626045014616, -3.198064885463341, 0.3957265660571476, 3.8957627593363213, 3.9787809035737727,
         -0.027207021428687028,
         -0.34033526585809604, -2.5588298675813492, -2.476922632108133, 2.5517283953758687, 1.5993085365149255,
         -1.2766691830705215,
         -3.4645291426813696, 0.05628567501007797, -0.849978518535373, 2.6160806954360716, -2.242316891022634,
         -0.7837252156670029,
         0.7039730333299845, -2.061478941832605, 0.6111988229297074, 4.703163619428018, -2.5485658533370104,
         3.076114150330298,
         -1.6134348075574048, -1.772138688351422, -0.9863985786580276, 3.5152685444009144, 0.7777428381228022,
         -1.4579303655037479,
         -2.274182843231667, 3.406012391260326, 0.00870953105274097, -0.5114736659036865, -2.2169821069535223,
         -2.635526452927418,
         -1.2365519548748827, 3.763177532954318, -0.24779334555842247, -1.0048734372691759, -2.8387420614128986,
         -0.5115610614233601,
         -2.8658867269614574, -0.4326936658670064, -2.0548181751338546, -1.7024706243294419, -1.2276809193242078,
         3.868532985643526,
         4.907118610850738, 0.43141710361675933, 0.051285978138296606, -3.420168891598557, -0.9553061457363121,
         -4.502791837199149,
         -4.275961604772258, 0.24369522604682725, 2.2315768953356976, 1.862851041047838, -4.24389732641611,
         -0.05410734053463795,
         -1.7569349549015643, 3.4067283985075374, -2.079513445525285, -2.144346112503098, 1.7999242973787268,
         -2.029288491048934,
         0.9078072792419061, 0.12082217426729347, -3.59460778028477, 3.194224812803144]
    Y = [27.33646518646935, 22.057067767069558, -0.29252932169587154, -3.2196198569087717, 21.105388153641453,
         14.96190610011547, 21.520393080102078, -5.843035746934613, -4.325394198651437, 28.94731756195954,
         27.92510745586494, -5.90896101045712, 2.89255562723833, 17.887303990029835, 32.11945264671019,
         0.29329934347622905, 13.96459646043565, 20.213654107861714, -3.1072009118582793, 25.983414176192056,
         35.199266021942556, -3.3022783416013235, -2.041307438176686, 22.65721988121683, 25.996613264118288,
         -4.716319915383961, 9.350988207079954, 32.76402144130039, 6.426062656495944, 2.3864117608450934,
         11.251616118575773, 27.073081518411318, 11.143581299235446, -3.090858571578666, -3.13659904575397,
         14.241744314894007, 15.897292207896811, 26.080065437103414, 21.925903907629856, 2.7329537000556923,
         7.61600445911094, 18.474596972440885, 28.999787577752663, 11.578467877601122, 17.097182719458726,
         1.2307909468815428, 22.89146316390771, 16.77718599349703, 11.334827165780759, 23.624223751643928,
         11.396000069430668, -8.611890918751039, 24.72718698675966, -0.2205406938120179, 20.420187561070623,
         20.564954537180885, 16.95272729062368, -1.717467841618502, 9.270433757420408, 20.504955944045154,
         21.573002817675007, 0.10988569193477371, 12.9358264153982, 14.173473105774296, 22.25964665321144,
         25.648642372571327, 19.713064358813913, -0.46490957452261084, 13.831982344000686, 17.903101645435214,
         25.944195265026163, 13.57121493112116, 23.287345971869158, 13.41554499752009, 22.65281487446945,
         20.049600812003717, 19.88043703004002, -2.07968838469641, -8.89526142216453, 11.09155925373586,
         15.805495336207384, 27.469651606847304, 15.358148193059694, 31.748429925113392, 31.921918077127597,
         13.134957404718133, 2.654581772926984, 6.450940249221191, 31.5327135937047, 14.972984687863164,
         21.436688963276914, 1.7374058608650795, 22.47121229896289, 19.92397225026726, 5.700696217286979,
         23.73985780894829, 10.278013673716487, 11.754127067336501, 28.388017863033408, -0.571702471141752]
    x = [-3.0388846783056445, -2.2205669355759805, 3.0026770138937344, 3.9848177123752517, -2.1262775940438647,
         -0.9272671284026952, -1.7265182734055817, 4.461181568138308, 4.276551645769736, -3.848693613685227]
    y = [27.33646518646935, 22.057067767069558, -0.29252932169587154, -3.2196198569087717, 21.105388153641453,
         14.96190610011547, 21.520393080102078, -5.843035746934613, -4.325394198651437, 28.94731756195954]
    # for i in
    # b = get_col(c, 0)
    # print(b)
    # d = adjoint_matrix(m)
    # size = len(x)
    # size2 = len(y)
    # q = inverse_matrix(m)
    # z = np.linalg.inv(m)
    # print(z)
#     k, l = linearRegression(x, y)
#     print(k, l)
    r = np.random.randint(low=3, high=10)
    A = np.random.randint(low=-10, high=10, size=(r, r))
    b = np.arange(r).reshape((r, 1))

    x = gj_Solve(A.tolist(), b.tolist(), epsilon=1.0e-8)
    print(x)
    Ax = np.dot(A, np.array(x))
    loss = np.mean((Ax - b) ** 2)
    print(loss)
