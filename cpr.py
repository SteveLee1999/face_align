import os
import cv2
import pandas as pd
import catboost
from tqdm import tqdm


path_base = '../dataset/'
path_train = path_base + 'train/'
path_test = path_base + 'test/'
LANDMARK_NUM = 98
IMG_WIDTH = 64
IMG_HEIGHT = 72


class CPR:
    def __init__(self, landmark_num):
        # construct landmark_num * 2 regressors
        self.regressors = list()
        for i in range(landmark_num * 2):
            regressor = catboost.CatBoostRegressor(silent=True)
            self.regressors.append(regressor)
        self.landmark_num = landmark_num

    def fit(self, total_x, total_y):
        for i in tqdm(range(self.landmark_num)):
            fit_x = total_x[i]

            # fit width
            fit_y = total_y[2 * i]
            self.regressors[2 * i].fit(fit_x, fit_y)

            # fit height
            fit_y = total_y[2 * i + 1]
            self.regressors[2 * i + 1].fit(fit_x, fit_y)

    def predict(self, test_x):
        all_prediction = list()
        for i in range(self.landmark_num):
            predicted_x = self.regressors[2 * i].predict(test_x[i])
            predicted_y = self.regressors[2 * i + 1].predict(test_x[i])

            all_prediction.append(predicted_x)
            all_prediction.append(predicted_y)
        return all_prediction


def read_train_data():
    pictures = list()
    landmarks = list()

    for file in os.listdir(path_train):
        if '.jpg' in file:
            continue

        dump = False
        landmark = list()
        with open(path_train + file) as fp:
            for line in fp:
                line = line.strip()
                width = int(float(line.split(',')[0]))
                height = int(float(line.split(',')[1]))
                if width < 0 or height < 0:
                    dump = True
                    break

                landmark.append([width, height])
        if dump:
            continue

        picture_id = file.split('.')[0]
        origin_picture = cv2.imread(path_train + picture_id + '.jpg', 0)

        # normalize
        for i in range(len(landmark)):
            landmark[i][0] *= IMG_WIDTH / len(origin_picture[0])
            landmark[i][1] *= IMG_HEIGHT / len(origin_picture)
        normalized_picture = cv2.resize(origin_picture, (IMG_WIDTH, IMG_HEIGHT))

        pictures.append(normalized_picture)
        landmarks.append(landmark)
    return pictures, landmarks


def calculate_average_landmark(landmarks):
    landmarks_sum = dict()
    average_landmark = list()
    for landmark in landmarks:
        for i in range(LANDMARK_NUM):
            if 2 * i not in landmarks_sum.keys():
                landmarks_sum[2 * i] = 0
            if 2 * i + 1 not in landmarks_sum.keys():
                landmarks_sum[2 * i + 1] = 0
            landmarks_sum[2 * i] += landmark[i][0]
            landmarks_sum[2 * i + 1] += landmark[i][1]

    for i in range(LANDMARK_NUM):
        average_landmark.append([int(landmarks_sum[2 * i] / len(landmarks)),
                                 int(landmarks_sum[2 * i + 1] / len(landmarks))])
    return average_landmark


def index_features(picture, average_landmark):
    feature = list()
    for column in [1, 0, -1]:
        for row in [-1, 0, 1]:
            width = average_landmark[0] + row
            height = average_landmark[1] + column
            if 0 <= width < IMG_WIDTH and 0 <= height < IMG_HEIGHT:
                feature.append(picture[height][width])
            else:
                feature.append(0)
    return feature


def generate_fit_data(pictures, landmarks, average_landmark):
    fit_x = list()
    fit_y = list()

    for i in range(LANDMARK_NUM):
        features = list()
        for picture in pictures:
            feature = index_features(picture, average_landmark[i])
            features.append(feature)
        fit_x.append(features)

        labels_x = list()
        labels_y = list()
        for landmark in landmarks:
            labels_x.append(landmark[i][0])
            labels_y.append(landmark[i][1])
        fit_y.append(labels_x)
        fit_y.append(labels_y)

    return fit_x, fit_y


def generate_test_data(average_landmark):
    pictures = list()
    ratio = list()
    test_x = list()

    for file in os.listdir(path_test):
        origin_picture = cv2.imread(path_test + file, 0)

        # normalize
        ratio.append([len(origin_picture[0]) / IMG_WIDTH, len(origin_picture) / IMG_HEIGHT])
        normalized_picture = cv2.resize(origin_picture, (IMG_WIDTH, IMG_HEIGHT))
        pictures.append(normalized_picture)

    # index feature
    for i in range(LANDMARK_NUM):
        features = list()
        for picture in pictures:
            feature = index_features(picture, average_landmark[i])
            features.append(feature)
        test_x.append(features)
    return test_x, ratio


'''
workflow:
    remove neg landmark samples
    convert to png
    calculate average landmark
    index features
'''
if __name__ == '__main__':
    pictures, landmarks = read_train_data()
    average_landmark = calculate_average_landmark(landmarks)
    pictures_test, ratio = generate_test_data(average_landmark)

    fit_x, fit_y = generate_fit_data(pictures, landmarks, average_landmark)
    model = CPR(LANDMARK_NUM)
    model.fit(fit_x, fit_y)

    test_result = model.predict(pictures_test)
    print(test_result)

    # recover
    for i in range(LANDMARK_NUM):
        for j in range(len(pictures_test)):
            test_result[2 * i][j] *= ratio[j][0]
            test_result[2 * i + 1][j] *= ratio[j][1]

    diction = dict()
    for i in range(LANDMARK_NUM):
        diction[2 * i] = test_result[2 * i]
        diction[2 * i + 1] = test_result[2 * i + 1]

    result = pd.DataFrame(diction)
    result.to_csv('./submission.csv')
