
import cv2

import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input





def loadtocrop(test=False):

    if(test):
        size = 105
    else:
        size = 500

    for i in range(size):

        print("hop " + str(i))

        if(test):
            o_landmark_path = 'data/test/o_landmarks/test_' + str(i) + '.txt'
            o_image_path = 'data/test/o_images/test_' + str(i) + '.png'
            landmark_path = 'data/test/landmarks/test_' + str(i) + '.txt'
            image_path = 'data/test/images/test_' + str(i) + '.png'
        else:
            o_landmark_path = 'data/train/o_landmarks/train_' + str(i) + '.txt'
            o_image_path = 'data/train/o_images/train_' + str(i) + '.png'
            landmark_path = 'data/train/landmarks/train_' + str(i) + '.txt'
            image_path = 'data/train/images/train_' + str(i) + '.png'

        smallest_x = 999999
        smallest_y = 999999
        greatest_x = 0
        greatest_y = 0

        img = cv2.imread(o_image_path)


        with open(o_landmark_path, 'r') as f:
            lines_list = f.readlines()

            for j in range(3, 58):
                string = lines_list[j]
                str1, str2 = string.split(' ')
                x_ = float(str1)
                #x_ = (x_ / width) * 224
                x_ = round(x_, 3)
                y_ = float(str2)
                #y_ = (y_ / height) * 224
                y_ = round(y_, 3)


                if(x_ < smallest_x):
                    smallest_x = int(x_)
                if(x_ > greatest_x):
                    greatest_x = int(x_)
                if (y_ < smallest_y):
                    smallest_y = int(y_)
                if (y_ > greatest_y):
                    greatest_y = int(y_)



                if (j == 3):
                    temp_x = np.array(x_)
                    temp_y = np.array(y_)

                else:
                    temp_x = np.hstack((temp_x, x_))
                    temp_y = np.hstack((temp_y, y_))

        print(smallest_x)
        print(greatest_x)
        print(smallest_y)
        print(greatest_y)

        if(smallest_x>5 and smallest_y>5):
            smallest_x -= 5
            smallest_y -= 5
        else:
            smallest_x = 0
            smallest_y = 0
        greatest_x += 5
        greatest_y += 5


        for k in range(55):
            temp_x[k] -= smallest_x
            temp_y[k] -= smallest_y

        width = greatest_x - smallest_x
        height = greatest_y - smallest_y
        for k in range(55):
            temp_x[k] = (temp_x[k] / width)
            temp_y[k] = (temp_y[k] / height)


        if (i == 0):
            Y = np.hstack((temp_x, temp_y))
            Y = Y[None, :]

        else:
            temp = np.hstack((temp_x, temp_y))
            temp = temp[None, :]
            Y = np.vstack((Y, temp))

        cv2.rectangle(img, (smallest_x, smallest_y), (greatest_x, greatest_y), (255, 255, 255), 3)
        crop_image = img[smallest_y:greatest_y, smallest_x:greatest_x]
        resize_image = cv2.resize(crop_image, (300, 300))
        cv2.imwrite(image_path, resize_image)

        with open(landmark_path, 'w') as f:
            f.write("version: 2\n")
            f.write("n_points: 55\n")
            f.write("{\n")
            for p in range(55):
                line = str(Y[i][p]) + " " + str(Y[i][p + 55]) + "\n"
                f.write(line)
            f.write("}\n")
            f.close()