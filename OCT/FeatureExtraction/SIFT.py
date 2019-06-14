import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# returns descriptor of image at pth
def feature_extract(pth, bowDiction_, sift_):
    im = cv2.imread(pth)
    gray_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return bowDiction_.compute(gray_, sift_.detect(gray_))


path = '../datas/train_data/'

# list of our class names
training_names = os.listdir(path)

training_paths = []
names_path = []
# get full list of all training images
for p in training_names:
    training_paths1 = os.listdir(path+p)
    for j in training_paths1:
        training_paths.append(path+p+'/'+j)
        names_path.append(p)

print('训练集数量', len(training_paths))

sift = cv2.xfeatures2d.SIFT_create(100)
BOW = cv2.BOWKMeansTrainer(50)

# 导出字典
# 提取SIFT特征，使用K-Means构建聚类
# for p in training_paths:
#     img = cv2.imread(p)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     kp, des = sift.detectAndCompute(gray, None)  # 找到关键点
#     BOW.add(des)
#
# dictionary = BOW.cluster()
#
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# sift2 = cv2.xfeatures2d.SIFT_create()
# bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
# bowDiction.setVocabulary(dictionary)
# print('bow dictionary', np.shape(dictionary))
# np.save('dictionary.npy', dictionary)
# print('导出字典   dictionary.npy')

# 导入字典
print('导入字典   dictionary.npy')
dictionary = np.load('dictionary.npy')
sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)


train_desc = []
train_labels = []
i = 0
for p in training_paths:
    train_desc.extend(feature_extract(p, bowDiction, sift))
    if names_path[i] == 'CNV':
        train_labels.append(1)
    if names_path[i] == 'DME':
        train_labels.append(2)
    if names_path[i] == 'DRUSEN':
        train_labels.append(3)
    if names_path[i] == 'NORMAL':
        train_labels.append(4)
    i = i+1

# print(train_desc)
print('svm items', len(train_desc), len(train_desc[0]))

model = svm.SVC(decision_function_shape="ovo", kernel='poly', C=0.8, gamma=18, degree=3, coef0=0.5,
                probability=True, verbose=True, cache_size=500)
model.fit(np.array(train_desc), np.array(train_labels))
scores = accuracy_score(np.array(train_labels), model.predict(np.array(train_desc)))
print('训练集准确率', scores)

# predicted = model.predict(np.array(train_desc))
# print('训练集结果', predicted)

joblib.dump(model, 'SVM_train_model.m')
print('导出模型   SVM_train_model.m')


rf = RandomForestClassifier(n_jobs=-1, verbose=1.0, max_features=18, n_estimators=500, max_depth=20,
                            min_samples_split=10, min_samples_leaf=10, oob_score=True, random_state=0)

rf.fit(np.array(train_desc), np.array(train_labels))
scores = accuracy_score(np.array(train_labels), rf.predict(np.array(train_desc)))
print('随机森林训练集准确率：', scores)
joblib.dump(rf, 'RF_train_model.m')
print('导出模型   RF_train_model.m')
