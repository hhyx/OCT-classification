# encoding:utf-8
import numpy
import cv2
import math
from skimage import measure


def LinearFitting(x, y):
    parameter = numpy.polyfit(x, y, 1)  # 一次多项式拟合，相当于线性拟合
    function = numpy.poly1d(parameter)
    return parameter, function


def PolynomialFitting(x, y):
    parameter = numpy.polyfit(x, y, 2)
    function = numpy.poly1d(parameter)
    return parameter, function


def PictureParsing(path, filename):
    img = cv2.imread(path + "MorphologicalClosing-" + filename, cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("../datas/" + "MorphologicalClosing-" + 'DRUSEN-2.png', cv2.IMREAD_GRAYSCALE)
    # img = save_max_objects(img)
    # cv2.imwrite(path + 'save_max_objects-' + filename, img)
    img = numpy.array(img)
    height, width = img.shape
    # print(width, height)

    x = numpy.zeros((1, width))
    ymid = numpy.zeros((1, width))
    ymin = numpy.zeros((1, width))
    for i in range(width):
        x[0][i] = i
        count = 0
        sum = 0
        for j in range(height):
            if img[j][i] == 255:
                count += 1
                sum += j
            elif img[j][i] == 0 and j > 0 and img[j-1][i] == 255:
                ymin[0][i] = height - j + 1
                break
        if count == 0 and i != 0:
            ymid[0][i] = ymid[0][i-1]
        elif count == 0:
            ymid[0][i] = height/2
        else:
            ymid[0][i] = height - sum / count

    return x, ymid, ymin


def CorrelationCoefficient(X, Y, function):
    size = X.size
    residual = 0
    total = 0
    for i in range(0, size-1):
        y = function(X[i])
        residual += math.pow((y - Y[i]), 2)
        total += math.pow(y, 2)

    return (total - residual) / total


# 输入：一张二值图像，无须指定面积阈值，
# 输出：会返回保留了面积最大的连通域的图像
def save_max_objects(img):
    labels = measure.label(img)  # 返回打上标签的img数组
    jj = measure.regionprops(labels)  # 找出连通域的各种属性。  注意，这里jj找出的连通域不包括背景连通域
    if len(jj) == 1:
        out = img
    else:
        # 通过与质心之间的距离进行判断
        num = labels.max()  # 连通域的个数
        del_array = numpy.array([0] * (num + 1))  # 生成一个与连通域个数相同的空数组来记录需要删除的区域（从0开始，所以个数要加1）
        for k in range(num):  # 这里如果遇到全黑的图像的话会报错
            if k == 0:
                initial_area = jj[0].area
                save_index = 1  # 初始保留第一个连通域
            else:
                k_area = jj[k].area  # 将元组转换成array

                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1

        del_array[save_index] = 1
        del_mask = del_array[labels]
        out = img * del_mask
    return out


def Fitting(path, filename):
    X, Ymid, Ymin = PictureParsing(path, filename)
    # print(Ymid[0])
    # print(Ymin[0])
    parameter, function = PolynomialFitting(X[0], Ymid[0])
    if parameter[0] > 0:
        print(function)
        return parameter
    else:
        parameter2, function2 = PolynomialFitting(X[0], Ymin[0])
        parameter3, function3 = LinearFitting(X[0], Ymin[0])
        if parameter2[0] > 0:
            c2 = CorrelationCoefficient(X[0], Ymin[0], function2)
            c3 = CorrelationCoefficient(X[0], Ymin[0], function3)
            if c2 > c3:
                print(function2)
                return parameter2
            else:
                print(function3)
                return parameter3
        else:
            print(function3)
            return parameter3
