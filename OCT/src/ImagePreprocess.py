from PIL import Image
import os


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
    def __init__(self, file_dir, name):
        self.file_dir = file_dir
        self.name = name

    def BM3D(self):
        print('BM3D')
        




if __name__ == '__main__':
    print("main")
    rename_image('../datas/train/CNV', 'CNV')
    rename_image('../datas/train/DME', 'DME')
    rename_image('../datas/train/DRUSEN', 'DRUSEN')
    rename_image('../datas/train/NORMAL', 'NORMAL')

