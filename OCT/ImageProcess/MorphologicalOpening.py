# encoding:utf-8
import cv2
import numpy


def MorphologicalOpening(path, filename):
    src = cv2.imread(path + "MedianFilter-" + filename)

    # 设置卷积核
    kernel = numpy.ones((20, 20), numpy.uint8)

    # 图像开运算
    result = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(path + "MorphologicalOpening-" + filename, result)
