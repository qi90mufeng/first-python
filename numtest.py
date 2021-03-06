import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(a)
# 结果返回一个tuple元组 (2L, 5L)
print(a.shape)
# 获得行数，返回 2
print(a.shape[0])
# 获得列数，返回 5
print(a.shape[1])
# 默认从0开始到10（不包括10），步长为1 # 返回 [0 1 2 3 4 5 6 7 8 9]
a11 = np.arange(10)
print(a11)
# 从5开始到20（不包括20），步长为2 # 返回 [ 5  7  9 11 13 15 17 19]
a12 = np.arange(5, 20, 2)
print(a12)
# 截取矩阵a中大于6的元素，范围的是一维数组  返回 [7  8  9 10]
b = a[a > 6]
print(b)


# 大于6清零后矩阵为
a[a > 6] = 0
print(a)

# ----------------------------------------------------------------------------------------------------------------------
print("----------------------")
a21 = np.array([[1, 2], [3, 4]])
a22 = np.array([[5, 6], [7, 8]])
# !注意 参数传入时要以列表list或元组tuple的形式传入
# 横向合并，返回结果如下 [[1 2 5 6][3 4 7 8]]
print(np.hstack([a21, a22]))
# 纵向合并，返回结果如下[[1 2][3 4][5 6][7 8]]
print(np.vstack((a21, a22)))


# ----------------------------------------------------------------------------------------------------------------------
print("----------------------")
# 生成首位是0，末位是10，含7个数的等差数列
a31 = np.linspace(0, 10, 7)
print(a31)
# logspace用于生成等比数列
a = np.logspace(0, 2, 5)
print(a)


# ----------------------------------------------------------------------------------------------------------------------
print("----------矩阵------------")
# ones创建全1矩阵
# zeros创建全0矩阵
# eye创建单位矩阵
# empty创建空矩阵（实际有值）

# 创建3*4的全1矩阵
a_ones = np.ones((3, 4))
print(a_ones)

a = np.array([[1,2,3],[4,5,6]])
# 对整个矩阵求累积和
print(a.cumsum())
# 对行方向求累积和
print(a.cumsum(axis=0))

# ----------------------------------------------------------------------------------------------------------------------
print("正态生成4行5列的二维数组----------------------")
# 正态生成4行5列的二维数组
arr = np.random.normal(1.75, 0.1, (4, 5))
print(arr)
abs(10)

# ----------------------------------------------------------------------------------------------------------------------
print("json串----------------------")
d = {"Start": "开始", "learning": "学习", "python": "python", "version": 3}
print(type(d))
print(d.keys())
# 用for语句循环返回所有键
for key in d.keys():
    print(key, end=' ')
print()
# 用for语句循环返回所有值
for values in d.values():
    print(values, end=' ')
print()
# 用items()返回一组一组的键值对   结果是list，只不过list里面的元素是元组
print(d.items())
# 查看dict项目个数
print(len(d))