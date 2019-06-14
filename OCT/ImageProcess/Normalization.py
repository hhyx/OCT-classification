# encoding:utf-8
import numpy
import cv2
import os


def adjust(img, width, height, function, minvalue):
    for i in range(width):
        move = int(function(i) - minvalue)
        for j in range(height - 1, move, -1):
            img[j][i] = img[j - move][i]
        for j in range(move, -1, -1):
            img[j][i] = 0
    return img


def Tailoring(img, width, height):
    minheight = 0
    maxheight = height-1
    i = 0
    j = 0
    for j in range(height):
        for i in range(width):
            if img[j][i] == 255:
                minheight = j
                break
        if i < width-1:
            break

    for j in range(height-1, -1, -1):
        for i in range(width):
            if img[j][i] == 255:
                maxheight = j
                break
        if i < width-1:
            break

    print(minheight, maxheight)
    return minheight, maxheight


def Normalization(parameter, path, filename, functions):
    img = cv2.imread(path + 'MorphologicalClosing-' + filename, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path + 'BM3D-' + filename, cv2.IMREAD_GRAYSCALE)
    img = numpy.array(img)
    img2 = numpy.array(img2)
    height, width = img.shape
    height2, width2 = img2.shape
    function = numpy.poly1d(parameter)
    loadpath = ''
    if filename[0:3] == 'CNV':
        loadpath = 'CNV'
    elif filename[0:3] == 'DME':
        loadpath = 'DME'
    elif filename[0:3] == 'DRU':
        loadpath = 'DRUSEN'
    elif filename[0:3] == 'NOR':
        loadpath = 'NORMAL'

    if functions == 'train':
        if not os.path.exists('../datas/train_data'):
            os.makedirs('../datas/train_data')
        if not os.path.exists('../datas/train_data/CNV'):
            os.makedirs('../datas/train_data/CNV')
        if not os.path.exists('../datas/train_data/DME'):
            os.makedirs('../datas/train_data/DME')
        if not os.path.exists('../datas/train_data/DRUSEN'):
            os.makedirs('../datas/train_data/DRUSEN')
        if not os.path.exists('../datas/train_data/NORMAL'):
            os.makedirs('../datas/train_data/NORMAL')
    elif functions == 'test':
        if not os.path.exists('../datas/test_data'):
            os.makedirs('../datas/test_data')
        if not os.path.exists('../datas/test_data/CNV'):
            os.makedirs('../datas/test_data/CNV')
        if not os.path.exists('../datas/test_data/DME'):
            os.makedirs('../datas/test_data/DME')
        if not os.path.exists('../datas/test_data/DRUSEN'):
            os.makedirs('../datas/test_data/DRUSEN')
        if not os.path.exists('../datas/test_data/NORMAL'):
            os.makedirs('../datas/test_data/NORMAL')

    if len(parameter) == 2:
        minvalue = min(function(0), function(width-1))
        print(minvalue)

        img = adjust(img, width, height, function, minvalue)
        cv2.imwrite(path + 'Normalization-' + filename, img)
        minheight, maxheight = Tailoring(img, width, height)
        img = numpy.delete(img, numpy.arange(maxheight, height - 1, 1), axis=0)
        img = numpy.delete(img, numpy.arange(0, minheight - 1, 1), axis=0)
        cv2.imwrite(path + 'Tailoring-' + filename, img)

        img2 = adjust(img2, width2, height2, function, minvalue)
        cv2.imwrite(path + 'Normalization2-' + filename, img2)
        img2 = numpy.delete(img2, numpy.arange(maxheight, height2 - 1, 1), axis=0)
        img2 = numpy.delete(img2, numpy.arange(0, minheight - 1, 1), axis=0)
        if functions == 'train':
            cv2.imwrite('../datas/train_data/' + loadpath + '/Result-' + filename, img2)
        elif functions == 'test':
            cv2.imwrite('../datas/test_data/' + loadpath + '/Result-' + filename, img2)

        print('Linear')

    else:
        if parameter[0] > 0:
            if 0 < -parameter[1]/(2*parameter[0]) < width:
                minvalue = min(function(0), function(width - 1), (4*parameter[0]*parameter[2]-parameter[1]*parameter[1])/(4*parameter[0]))
            else:
                minvalue = min(function(0), function(width - 1))
            print(minvalue)

            img = adjust(img, width, height, function, minvalue)
            cv2.imwrite(path + 'Normalization-' + filename, img)
            minheight, maxheight = Tailoring(img, width, height)
            img = numpy.delete(img, numpy.arange(maxheight, height - 1, 1), axis=0)
            img = numpy.delete(img, numpy.arange(0, minheight - 1, 1), axis=0)
            cv2.imwrite(path + 'Tailoring-' + filename, img)

            img2 = adjust(img2, width2, height2, function, minvalue)
            cv2.imwrite(path + 'Normalization2-' + filename, img2)
            img2 = numpy.delete(img2, numpy.arange(maxheight, height2 - 1, 1), axis=0)
            img2 = numpy.delete(img2, numpy.arange(0, minheight - 1, 1), axis=0)
            if functions == 'train':
                cv2.imwrite('../datas/train_data/' + loadpath + '/Result-' + filename, img2)
            elif functions == 'test':
                cv2.imwrite('../datas/test_data/' + loadpath + '/Result-' + filename, img2)

        else:
            minvalue = min(function(0), function(width-1))
            print(minvalue)
            img = adjust(img, width, height, function, minvalue)
            cv2.imwrite(path + 'Normalization-' + filename, img)
            minheight, maxheight = Tailoring(img, width, height)
            img = numpy.delete(img, numpy.arange(maxheight, height - 1, 1), axis=0)
            img = numpy.delete(img, numpy.arange(0, minheight - 1, 1), axis=0)
            cv2.imwrite(path + 'Tailoring-' + filename, img)

            img2 = adjust(img2, width2, height2, function, minvalue)
            cv2.imwrite(path + 'Normalization2-' + filename, img2)
            img2 = numpy.delete(img2, numpy.arange(maxheight, height2 - 1, 1), axis=0)
            img2 = numpy.delete(img2, numpy.arange(0, minheight - 1, 1), axis=0)
            if functions == 'train':
                cv2.imwrite('../datas/train_data/' + loadpath + '/Result-' + filename, img2)
            elif functions == 'test':
                cv2.imwrite('../datas/test_data/' + loadpath + '/Result-' + filename, img2)

        print('Polynomial')
