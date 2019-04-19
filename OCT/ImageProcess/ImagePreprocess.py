import os
from BM3D import BM3D
from Binaryzation import Binaryzation
from MedianFilter import MedianFilter
from MorphologicalClosing import MorphologicalClosing
from MorphologicalOpening import MorphologicalOpening
from Fitting import Fitting


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
        self.function = None


    def ImagePreprocess(self):
        BM3D(self.path, self.filename)
        Binaryzation(self.storepath, self.filename)
        MedianFilter(self.storepath, self.filename)
        MorphologicalOpening(self.storepath, self.filename)
        MorphologicalClosing(self.storepath, self.filename)
        self.function = Fitting(self.storepath, self.filename)


if __name__ == '__main__':
    # rename_image('../datas/train/CNV', 'CNV')
    # rename_image('../datas/train/DME', 'DME')
    # rename_image('../datas/train/DRUSEN', 'DRUSEN')
    # rename_image('../datas/train/NORMAL', 'NORMAL')
    progress = ImagePreprocess("../datas/train/CNV/", "../datas/", "CNV-0.png")
    progress.ImagePreprocess()
    print(progress.function)
