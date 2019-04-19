# encoding:utf-8
import cv2


def MedianFilter(path, filename):
    img = cv2.imread(path + "Binaryzation-" + filename)

    # 中值滤波
    result = cv2.medianBlur(img, 7)

    # 显示图像
    # cv2.imshow("source img", img)
    # cv2.imshow("medianBlur", result)
    cv2.imwrite("../datas/MedianFilter-" + filename, result)

if __name__ == '__main__':
    MedianFilter("../datas/", "CNV-1.png")
    cv2.waitKey()
    cv2.destroyAllWindows()
