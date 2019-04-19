# encoding:utf-8
import cv2
import numpy as np


def MorphologicalOpening(path, filename):
    src = cv2.imread(path + "MedianFilter-" + filename)

    # 设置卷积核
    kernel = np.ones((15, 15), np.uint8)

    # 图像开运算
    result = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("src", src)
    # cv2.imshow("result", result)
    cv2.imwrite("../datas/MorphologicalOpening-" + filename, result)

if __name__ == '__main__':
    MorphologicalOpening("../datas/", "CNV-6.png")
    cv2.waitKey()
    cv2.destroyAllWindows()
