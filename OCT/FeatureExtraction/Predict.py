import cv2
import numpy as np
import os
from sklearn.externals import joblib


# returns descriptor of image at pth
def feature_extract(pth, bowDiction_, sift_):
    im = cv2.imread(pth)
    gray_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return bowDiction_.compute(gray_, sift_.detect(gray_))


path = '../datas/test_data/'

# list of our class names
test_names = os.listdir(path)

test_paths = []
names_path = []
for p in test_names:
    test_paths1 = os.listdir(path+p)
    for j in test_paths1:
        test_paths.append(path+p+'/'+j)
        names_path.append(p)

print('测试集数量', len(test_paths))

sift = cv2.xfeatures2d.SIFT_create(100)

dictionary = np.load('dictionary.npy')
sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)
print('bow dictionary', np.shape(dictionary))

test_desc = []
test_labels = []
i = 0
for p in test_paths:
    test_desc.extend(feature_extract(p, bowDiction, sift))
    if names_path[i] == 'CNV':
        test_labels.append(1)
    if names_path[i] == 'DME':
        test_labels.append(2)
    if names_path[i] == 'DRUSEN':
        test_labels.append(3)
    if names_path[i] == 'NORMAL':
        test_labels.append(4)
    i = i+1

clf = joblib.load('SVM_train_model.m')
predict = clf.predict(np.array(test_desc))
count = 0
for i in range(len(test_paths)):
    if predict[i] == test_labels[i]:
        count += 1
print('测试集准确率', count/len(test_paths))
# print(predict)

rf = joblib.load('RF_train_model.m')
predict = rf.predict(np.array(test_desc))
count = 0
for i in range(len(test_paths)):
    if predict[i] == test_labels[i]:
        count += 1
print('测试集准确率', count/len(test_paths))
# print(predict)
