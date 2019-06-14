# encoding:utf-8
import cv2
from skimage import measure
import numpy


# 输入：一张二值图像，无须指定面积阈值，
# 输出：会返回保留了面积最大的连通域的图像
def save_max_objects(img):
    labels = measure.label(img)  # 返回打上标签的img数组
    jj = measure.regionprops(labels)  # 找出连通域的各种属性。  注意，这里jj找出的连通域不包括背景连通域
    if len(jj) == 1:
        out = img
    else:
        # 通过与质心之间的距离进行判断
        num = labels.max()  # 连通域的个数
        del_array = numpy.array([0] * (num + 1))  # 生成一个与连通域个数相同的空数组来记录需要删除的区域（从0开始，所以个数要加1）
        for k in range(num):  # 这里如果遇到全黑的图像的话会报错
            if k == 0:
                initial_area = jj[0].area
                save_index = 1  # 初始保留第一个连通域
            else:
                k_area = jj[k].area  # 将元组转换成array

                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1

        del_array[save_index] = 1
        del_mask = del_array[labels]
        out = img * del_mask
    return out


def MedianFilter(path, filename):
    img = cv2.imread(path + "Binaryzation-" + filename)

    # 中值滤波
    result = cv2.medianBlur(img, 7)
    result = save_max_objects(result)

    cv2.imwrite(path + "MedianFilter-" + filename, result)
