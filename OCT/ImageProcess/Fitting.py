# encoding:utf-8
import numpy
import cv2
import math


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


if __name__ == '__main__':
    print(Fitting("../datas/", "CNV-9.png"))
