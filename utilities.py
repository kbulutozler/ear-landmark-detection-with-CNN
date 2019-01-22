import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def load_data(test=False, size=3000, test_size=630, single_img=False, single_img_path='give a path please'):


    if (test):
        size = test_size

    if(single_img):
        size = 1

    for i in range(0, size):        # take the images

        img_path = 'data/train/images/train_' + str(i) + '.png'
        if (test):
            img_path =  'data/test/images/test_' + str(i) + '.png'

        if (single_img):

            img_path = single_img_path
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224)) # first resize the original single image
            cv2.imwrite('data/single/single_img.png',img)       # save the resized original single image
            img = image.load_img('data/single/single_img.png')      # load the resized original single image
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            y = None
            return x, y         # return the single image as preprocessed for the model

        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        if (i == 0):            # if first image
            X = x
            continue
        X = np.vstack((X, x))       # if not first image



    for i in range(0, size):        # take the landmarks


        txt_path = 'data/train/landmarks/train_' + str(i) + '.txt'
        if (test):
            txt_path = 'data/test/landmarks/test_' + str(i) + '.txt'



        with open(txt_path, 'r') as f:
            lines_list = f.readlines()

            for j in range(3, 58):      # in landmark text files, landmarks start at 3rd line end in 57th
                string = lines_list[j]
                str1, str2 = string.split(' ')
                x_ = float(str1)
                y_ = float(str2)
                if (j == 3):            # if first landmark point
                    temp_x = np.array(x_)
                    temp_y = np.array(y_)
                    continue

                # if not first landmark point

                temp_x = np.hstack((temp_x, x_))
                temp_y = np.hstack((temp_y, y_))


        if (i == 0):            # if first image's landmarks
            Y = np.hstack((temp_x, temp_y))
            Y = Y[None, :]
            continue

        # if not first image's landmarks
        temp = np.hstack((temp_x, temp_y))
        temp = temp[None, :]
        Y = np.vstack((Y, temp))

    return X, Y
