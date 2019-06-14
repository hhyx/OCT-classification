# encoding:utf-8
import cv2
import numpy
from PIL import Image


def textureSquare(imgPath):
    img = Image.open(imgPath)

    width, height = img.size
    for i in range(0, width):
        for j in range(0, height):
            data = (img.getpixel((i, j)))
            if data == 255:
                img.putpixel((i, j), 0)
    img = img.convert("L")
    img.save(imgPath)


def Binaryzation(path, filename):
    textureSquare(path + "BM3D-" + filename)
    im_gray = cv2.imread(path+"BM3D-"+filename, cv2.IMREAD_GRAYSCALE)

    h, w = im_gray.shape[:2]
    m = numpy.reshape(im_gray, [1, w * h])
    mean = m.sum() / (w * h)
    ret, binary = cv2.threshold(im_gray, mean, 255, cv2.THRESH_BINARY)

    cv2.imwrite(path + "Binaryzation-" + filename, binary)
