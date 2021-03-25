import cv2
import os
import numpy as np
import catboost as cbt
import xgboost as xgb
import time
from sklearn.decomposition import PCA
from pandas import DataFrame

TESTSIZE = 1000
TRAINSIZE = 7500
IMGWIDTH = 64
IMGHEIGHT = 72
POINTNUM = 98
DIM = 1000
path = os.getcwd()
os.chdir(path)
imgs_name = []
for i in range(0, TRAINSIZE):
    imgs_name.append('train\\' + str('%04d' % i) + '.jpg')
imgs = []
shapes = []
for item in imgs_name:
    ori_img = cv2.imread(item, 0)
    shapes.append(ori_img.shape)
    new_img = cv2.resize(ori_img, (IMGWIDTH, IMGHEIGHT))
    imgs.append(new_img)
points_name = []
points = []
for i in range(0, TRAINSIZE):
    points_name.append('train\\' + str('%04d' % i) + '.pts')
for i in range(0, TRAINSIZE):
    points.append([])
    with open(points_name[i], 'r') as file:
        for eatchline in file:
            tmp = eatchline.split(',')
            picx = float(tmp[0])
            picy = float(tmp[1])
            points[i].append([picx, picy])
train_x = []
train_y_x = []
train_y_y = []
for i in range(0, TRAINSIZE):
    train_x.append([])
    for height in range(0, IMGHEIGHT):
        for width in range(0, IMGWIDTH):
            train_x[i].append(imgs[i][height][width])
for i in range(0, TRAINSIZE):
    list_x = []
    list_y = []
    for point in range(0, POINTNUM):
        list_x.append(points[i][point][0])
        list_y.append(points[i][point][1])
    train_y_x.append(np.array(list_x) / shapes[i][1] * IMGWIDTH)
    train_y_y.append(np.array(list_y) / shapes[i][0] * IMGHEIGHT)
train_x = DataFrame(np.array(train_x))
train_y_x = np.array(train_y_x).T
train_y_y = np.array(train_y_y).T
test_shapes = []
test_img_name = []
for i in range(0, TESTSIZE):
    test_img_name.append('test\\' + str('%04d' % i) + '.jpg')
for item in test_img_name:
    ori_img = cv2.imread(item, 0)
    test_shapes.append(ori_img.shape)
    new_img = cv2.resize(ori_img, (IMGWIDTH, IMGHEIGHT))
    imgs.append(new_img)
test_x = []
for i in range(0, TESTSIZE):
    test_x.append([])
    for height in range(0, IMGHEIGHT):
        for width in range(0, IMGWIDTH):
            test_x[i].append(imgs[i][height][width])
test_x = DataFrame(np.array(test_x))
# dimensionality reduction for training set
pca_train = PCA(n_components=DIM)
train_x = pca_train.fit_transform(train_x)
print(train_x.shape)

# for test set
pca_test = PCA(n_components=DIM)
test_x = pca_test.fit_transform(test_x)
print(test_x.shape)

predict = []
print("model train begin")
for index in range(0, POINTNUM):
    print(time.strftime('%Y-%m-%d  %H:%M:%S %j %U', time.localtime()))
    print("this is " + str(index) + "-th x")
    x_model = cbt.CatBoostRegressor(max_depth=5, learning_rate=0.1, n_estimators=1260, silent=True)
    train_y = DataFrame(train_y_x[index])
    train_in_x = train_x
    train_y = train_y
    print(time.strftime('%Y-%m-%d  %H:%M:%S %j %U', time.localtime()))
    print("this is " + str(index) + "-th train")
    x_model.fit(train_x, train_y)
    print(time.strftime('%Y-%m-%d  %H:%M:%S %j %U', time.localtime()))
    print("this is " + str(index) + "-th predict x")
    list_x = x_model.predict(test_x)
    for i in range(0, TESTSIZE):
        list_x[i] = list_x[i] / IMGWIDTH * test_shapes[i][1]
    predict.append(list_x)
    # predict label
    print(time.strftime('%Y-%m-%d  %H:%M:%S %j %U', time.localtime()))
    print("this is " + str(index) + "-th y")
    y_model = cbt.CatBoostRegressor(max_depth=5, learning_rate=0.1, n_estimators=260, silent=True)
    train_y = DataFrame(train_y_y[index])
    train_in_x = train_x
    train_y = train_y
    print(time.strftime('%Y-%m-%d  %H:%M:%S %j %U', time.localtime()))
    print("this is " + str(index) + "-th train y")
    y_model.fit(train_in_x, train_y)
    print(time.strftime('%Y-%m-%d  %H:%M:%S %j %U', time.localtime()))
    print("this is " + str(index) + "-th predict y")
    list_y = y_model.predict(test_x)
    for i in range(0, TESTSIZE):
        list_y[i] = list_y[i] / IMGHEIGHT * test_shapes[i][0]
    predict.append(list_y)
sub = DataFrame(np.array(predict).T)
print(sub)
sub.to_csv("submission.csv")
