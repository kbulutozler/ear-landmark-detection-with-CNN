import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from sklearn.utils import shuffle

def load_data(test=False, size=13):

    if (test):
        size = 105
    for i in range(0, size):

        img_path = 'data/train/c_images/c_train_' + str(i) + '.png'
        if (test):
            img_path = 'data/test/images/test_' + str(i) + '.png'

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        if (i == 0):
            X = x
            continue
        X = np.vstack((X, x))

    for i in range(0, size):


        txt_path = 'data/train/c_landmarks/c_train_' + str(i) + '.txt'
        if (test):
            txt_path = 'data/test/landmarks/test_' + str(i) + '.txt'

        with open(txt_path, 'r') as f:
            lines_list = f.readlines()

            for j in range(3, 58):
                string = lines_list[j]
                str1, str2 = string.split(' ')
                x_ = float(str1)
                x_ = round(x_, 3)
                y_ = float(str2)
                y_ = round(y_, 3)
                if (j == 3):
                    temp_x = np.array(x_)
                    temp_y = np.array(y_)
                    continue

                temp_x = np.hstack((temp_x, x_))
                if (i == 0):
                    print(x_)
                temp_y = np.hstack((temp_y, y_))
                if (i == 0):
                    print(y_)

        if (i == 0):
            Y = np.hstack((temp_x, temp_y))
            Y = Y[None, :]
            continue

        temp = np.hstack((temp_x, temp_y))
        temp = temp[None, :]
        Y = np.vstack((Y, temp))

    return X, Y

def adjustDots(o_x, o_y, width, height, resize, i, path):

    test=False

    txt_path = 'data/train/landmarks/train_' + str(i) + '.txt'
    if (test):
        txt_path = 'data/test/landmarks/test_' + str(i) + '.txt'

    with open(txt_path, 'r') as f:
        lines_list = f.readlines()

        for j in range(3, 58):
            string = lines_list[j]
            str1, str2 = string.split(' ')
            x_ = float(str1)
            x_ = round(x_, 3)
            y_ = float(str2)
            y_ = round(y_, 3)
            if (j == 3):
                temp_x = np.array(x_)
                temp_y = np.array(y_)
                continue

            temp_x = np.hstack((temp_x, x_))
            temp_y = np.hstack((temp_y, y_))


    Y = np.hstack((temp_x, temp_y))


    for j in range(55):
        Y[j] -= o_x;
        Y[j+55] -= o_y;
        #adjusted to the origin

        Y[j] = (Y[j] / width) * resize
        Y[j+55] = (Y[j+55] / height) * resize
        # adjusted to the new size

        Y[j] = round(Y[j],3)
        Y[j+55] = round(Y[j+55], 3)


    txt_path = path + str(i) + '.txt'

    with open(txt_path, 'w') as f:
        f.write("version: 2\n")
        f.write("n_points: 55\n")
        f.write("{\n")
        for j in range(55):
            line = str(Y[j]) + " " + str(Y[j+55]) + "\n"
            f.write(line)
        f.write("}\n")
        f.close()