# encoding:utf-8
import cv2
import numpy as np


def MorphologicalClosing(path, filename):
    src = cv2.imread(path + "MorphologicalOpening-" + filename)

    # 设置卷积核
    kernel = np.ones((30, 30), np.uint8)

    # 图像开运算
    result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("src", src)
    # cv2.imshow("result", result)
    cv2.imwrite("../datas/MorphologicalClosing-" + filename, result)


if __name__ == '__main__':
    MorphologicalClosing("../datas/", "CNV-6.png")
    cv2.waitKey()
    cv2.destroyAllWindows()
