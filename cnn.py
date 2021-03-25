from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils, generic_utils
import numpy as np
import cv2

FILE_PATH = 'face_landmark.h5'
trainpath = 'C:/Users/zhong/PycharmProjects/face_detect/dataset/train/'
testpath = 'C:/Users/zhong/PycharmProjects/face_detect/dataset/test/'


def __data_label__(path):
    datalist = []
    labellist = []
    for i in range(0, 7500):
        imgname = trainpath + str('%04d' % i) + ".jpg"
        f = open(trainpath + str('%04d' % i) + ".pts", "r")
        new_label = []
        for line in f.readlines():
            a = line.replace("\n", "")
            b = a.split(",")
            new_label.append(b[0])
            new_label.append(b[1])
        labellist.append(new_label)
        image = load_img(trainpath + str('%04d' % i) + ".jpg")
        datalist.append(np.array(img_to_array(image).tolist()))
    img_data = np.array(datalist)
    img_data /= 255
    label = np.array(labellist)
    print(img_data)
    return img_data, label



###############
# 开始建立CNN模型
###############


# 生成一个model
def __CNN__():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(218, 178, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.summary()
    return model


def train(model, testdata, testlabel, traindata, trainlabel):
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(traindata, trainlabel, batch_size=16, epochs=20,
              validation_data=(testdata, testlabel))
    model.evaluate(testdata, testlabel, batch_size=16, verbose=1,)


def save(model, file_path=FILE_PATH):
    print('Model Saved.')
    model.save_weights(file_path)


def load(model, file_path=FILE_PATH):
    print('Model Loaded.')
    model.load_weights(file_path)


def predict(model, image):
    img = image.resize((1, 218, 178, 3))
    img = image.astype('float32')
    img /= 255
    result = model.predict(img)
    result = result*1000+10
    print(result)
    return result


def point(x, y):
    cv2.circle(img, (x, y), 1, (0, 0, 255), 10)


def resize_image(image, height, width):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    if h < height:
        top = height - h
    elif w < width:
        right = width - w
    else:
        pass

    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, 0, 0, right, cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant, (height, width))


############
# 主模块
############
if __name__ == '__main__':
    model = __CNN__()
    testdata, testlabel = __data_label__(testpath)
    traindata, trainlabel = __data_label__(trainpath)
    train(model,testdata, testlabel, traindata, trainlabel)
    model.save(FILE_PATH)
    model.load_weights(FILE_PATH)
    img = []
    for i in range(0, 1000):
        image = load_img(trainpath + str('%04d' % i) + ".jpg")
        image = resize_image(image, 500, 500)
        predict(model, image)
