# encoding:utf-8
import cv2
import numpy


def Binaryzation(path, filename):
    im_gray = cv2.imread(path+"BM3D-"+filename, cv2.IMREAD_GRAYSCALE)

    # retval, dst = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow("dst", dst)
    # cv2.imwrite("../datas/Binaryzation1-" + filename, dst)

    h, w = im_gray.shape[:2]
    m = numpy.reshape(im_gray, [1, w * h])
    mean = m.sum() / (w * h)
    # print("mean:", mean)
    ret, binary = cv2.threshold(im_gray, mean, 255, cv2.THRESH_BINARY)

    # cv2.imshow("outPutGray", im_gray)
    # cv2.imshow('outPutBinaay',binary)
    cv2.imwrite("../datas/Binaryzation-" + filename, binary)


if __name__ == '__main__':
    Binaryzation("../datas/", "CNV-6.png")
    cv2.waitKey()
    cv2.destroyAllWindows()