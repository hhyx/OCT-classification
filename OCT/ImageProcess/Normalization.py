# encoding:utf-8
import numpy
import cv2


def Normalization(parameter, path, filename):
    img = cv2.imread(path + "MorphologicalClosing-" + filename, cv2.IMREAD_GRAYSCALE)
    img = numpy.array(img)
    height, width = img.shape
    function = numpy.poly1d(parameter)

    if len(parameter) == 2:
        minvalue = min(function(0), function(width-1))
        print(minvalue)
        for i in range(width):
            move = int(function(i) - minvalue)
            for j in range(height-1, move):
                img[j][i] = img[j-move][i]
            for j in range(move, -1, -1):
                img[j][i] = 0

        cv2.imwrite("../datas/Normalization-" + filename, img)
        print('Linear')

    else:
        if parameter[0] > 0:
            minvalue = min(function(0), function(width - 1), (4*parameter[0]*parameter[2]-parameter[1]*parameter[1])/(4*parameter[0]))
            print(minvalue)
            for i in range(width):
                move = int(function(i) - minvalue)
                for j in range(height - 1, move, -1):
                    img[j][i] = img[j - move][i]
                for j in range(move, -1, -1):
                    img[j][i] = 0
        else:
            minvalue = min(function(0), function(width-1))
            print(minvalue)
            for i in range(width):
                move = int(function(i) - minvalue)
                for j in range(height-1, move):
                    img[j][i] = img[j-move][i]
                for j in range(move, -1, -1):
                    img[j][i] = 0

        cv2.imwrite("../datas/Normalization-" + filename, img)
        print('Polynomial')


if __name__ == '__main__':
    Normalization([7.98554107e-04, -6.49025423e-01, 3.16419538e+02], "../datas/", "CNV-9.png")
