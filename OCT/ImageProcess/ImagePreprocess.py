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
    k = 0
    print('正在重命名图片。。。。。。。')
    for file in os.listdir(file_dir):
        # 获取该路径文件下的所有图片
        src = os.path.join(os.path.abspath(file_dir), file)
        # 修改后图片的存储位置（目标文件夹+新的图片的名称）
        dst = os.path.join(os.path.abspath(file_dir), name + '-' + str(k) + '.png')
        os.rename(src, dst)  # 将图片重新命名
        k = k + 1
    print('-----------重命名完毕-----------')


class ImagePreprocess:
    def __init__(self, path, storepath, filename, function):
        self.path = path
        self.storepath = storepath
        self.filename = filename
        self.parameter = None
        self.function = function

    def preprocess(self):
        # BM3D(self.path, self.filename)
        Binaryzation(self.storepath, self.filename)
        MedianFilter(self.storepath, self.filename)
        MorphologicalOpening(self.storepath, self.filename)
        MorphologicalClosing(self.storepath, self.filename)
        self.parameter = Fitting(self.storepath, self.filename)
        Normalization(self.parameter, self.storepath, self.filename, function)


if __name__ == '__main__':
    function = 'test'  # 处理训练集还是测试集 train/test
    # 图像重命名
    # rename_image('../datas/'+function+'/CNV', 'CNV')
    # rename_image('../datas/'+function+'/DME', 'DME')
    # rename_image('../datas/'+function+'/DRUSEN', 'DRUSEN')
    # rename_image('../datas/'+function+'/NORMAL', 'NORMAL')
    e1 = cv2.getTickCount()
    filename = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    for j in range(0, 1):
        for i in range(1, 2):
            progress = ImagePreprocess('../datas/'+function+'/' + filename[j] + '/', '../datas/BM3D/', filename[j] +
                                       '-' + str(i) + '.png', function)
            print('Image Preprocess ' + filename[j] + '-'+str(i)+'.png')
            progress.preprocess()
            os.remove('../datas/BM3D/BM3D-' + filename[j] + '-' + str(i) + '.png')
            os.remove('../datas/BM3D/Binaryzation-' + filename[j] + '-' + str(i) + '.png')
            os.remove('../datas/BM3D/MedianFilter-' + filename[j] + '-' + str(i) + '.png')
            os.remove('../datas/BM3D/MorphologicalOpening-' + filename[j] + '-' + str(i) + '.png')
            os.remove('../datas/BM3D/MorphologicalClosing-' + filename[j] + '-' + str(i) + '.png')
            os.remove('../datas/BM3D/Normalization-' + filename[j] + '-' + str(i) + '.png')
            os.remove('../datas/BM3D/Tailoring-' + filename[j] + '-' + str(i) + '.png')
            os.remove('../datas/BM3D/Normalization2-' + filename[j] + '-' + str(i) + '.png')

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()  # 计算函数执行时间
    print('The Processing time of the picture is %f s' % time)
