import os
import cv2
from BM3D import BM3D
from Binaryzation import Binaryzation
from MedianFilter import MedianFilter
from MorphologicalClosing import MorphologicalClosing
from MorphologicalOpening import MorphologicalOpening
from Fitting import Fitting
from Normalization import Normalization


def rename_image(file_dir, name):
    # 第一个参数是目标文件名;
    # 第二个参数是图片的名称
    i = 0
    print("正在重命名图片。。。。。。。")
    for file in os.listdir(file_dir):
        # 获取该路径文件下的所有图片
        src = os.path.join(os.path.abspath(file_dir), file)
        # 修改后图片的存储位置（目标文件夹+新的图片的名称）
        dst = os.path.join(os.path.abspath(file_dir), name + '-' + str(i) + '.png')
        os.rename(src, dst)  # 将图片重新命名
        i = i + 1
    print("-----------重命名完毕-----------")


class ImagePreprocess:
    def __init__(self, path, storepath, filename):
        self.path = path
        self.storepath = storepath
        self.filename = filename
        self.parameter = None

    def ImagePreprocess(self):
        # BM3D(self.path, self.filename)
        # Binaryzation(self.storepath, self.filename)
        # MedianFilter(self.storepath, self.filename)
        # MorphologicalOpening(self.storepath, self.filename)
        # MorphologicalClosing(self.storepath, self.filename)
        self.parameter = Fitting(self.storepath, self.filename)
        Normalization(self.parameter, self.storepath, self.filename)


if __name__ == '__main__':
    # rename_image('../datas/train/CNV', 'CNV')
    # rename_image('../datas/train/DME', 'DME')
    # rename_image('../datas/train/DRUSEN', 'DRUSEN')
    # rename_image('../datas/train/NORMAL', 'NORMAL')
    e1 = cv2.getTickCount()
    progress = ImagePreprocess("../datas/train/DRUSEN/", "../datas/", "DRUSEN-5.png")
    progress.ImagePreprocess()
    print(progress.parameter)
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()  # 计算函数执行时间
    print("The Processing time of the picture is %f s" % time)
