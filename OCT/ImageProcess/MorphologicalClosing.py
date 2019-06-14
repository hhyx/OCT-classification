# encoding:utf-8
import cv2
import numpy


def MorphologicalClosing(path, filename):
    src = cv2.imread(path + "MorphologicalOpening-" + filename)

    # 设置卷积核
    kernel = numpy.ones((25, 25), numpy.uint8)

    # 图像闭运算
    result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(path + "MorphologicalClosing-" + filename, result)
