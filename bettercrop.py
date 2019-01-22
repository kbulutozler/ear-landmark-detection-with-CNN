import cv2
import numpy as np

# the purpose of this file is creating a dataset of images consist of only ears. Original images consist of full bodies
def loadtocrop(test=False):

    if(test):
        size = 105  # test size is 105
    else:
        size = 500  # training size is 500

    ctr = -1 # for file numbers

    for shp in range(6):
        for i in range(size):

            ctr += 1

            if(test):
                o_landmark_path = 'data/test/o_landmarks/test_' + str(i) + '.txt'       # original image path
                o_image_path = 'data/test/o_images/test_' + str(i) + '.png'             # original landmark path
                landmark_path = 'data/test/landmarks/test_' + str(ctr) + '.txt'         # preprocessed image path
                image_path = 'data/test/images/test_' + str(ctr) + '.png'               # preprocessed landmark path
            else:
                o_landmark_path = 'data/train/o_landmarks/train_' + str(i) + '.txt'     # original image path
                o_image_path = 'data/train/o_images/train_' + str(i) + '.png'           # original landmark path
                landmark_path = 'data/train/landmarks/train_' + str(ctr) + '.txt'       # preprocessed image path
                image_path = 'data/train/images/train_' + str(ctr) + '.png'             # preprocessed landmark path

            smallest_x = 999999
            smallest_y = 999999
            greatest_x = 0
            greatest_y = 0

            img = cv2.imread(o_image_path)          # take the original image


            with open(o_landmark_path, 'r') as f:
                lines_list = f.readlines()         # all lines as list

                for j in range(3, 58):          # in landmark text files, landmarks start at 3rd line end in 57th
                    string = lines_list[j]
                    str1, str2 = string.split(' ')
                    x_ = float(str1)
                    x_ = round(x_, 3)
                    y_ = float(str2)
                    y_ = round(y_, 3)


                    if(x_ < smallest_x):
                        smallest_x = int(x_)
                    if(x_ > greatest_x):
                        greatest_x = int(x_)
                    if (y_ < smallest_y):
                        smallest_y = int(y_)
                    if (y_ > greatest_y):
                        greatest_y = int(y_)

                    if (j == 3):                # if first landmark point
                        temp_x = np.array(x_)
                        temp_y = np.array(y_)

                    else:                           # if not first landmark point
                        temp_x = np.hstack((temp_x, x_))
                        temp_y = np.hstack((temp_y, y_))

            # some padding
            if(smallest_x>5 and smallest_y>5):
                smallest_x -= 5
                smallest_y -= 5
            else:
                smallest_x = 0
                smallest_y = 0
            greatest_x += 5
            greatest_y += 5

            for k in range(55):     # adjustment to the cropped image
                temp_x[k] -= smallest_x
                temp_y[k] -= smallest_y

            width = greatest_x - smallest_x
            height = greatest_y - smallest_y

            for k in range(55):     # normalization
                temp_x[k] = (temp_x[k] / width)
                temp_y[k] = (temp_y[k] / height)

                if (shp == 1): #flip vertical
                    temp_x[k] = 1 - temp_x[k]

                if(shp == 2): #rotate left
                    wow = temp_x[k]
                    temp_x[k] = temp_y[k]
                    temp_y[k] = wow
                    temp_y[k] = 1 - temp_y[k]

                if (shp == 3):  # rotate right
                    wow = temp_x[k]
                    temp_x[k] = temp_y[k]
                    temp_y[k] = wow
                    temp_x[k] = 1 - temp_x[k]

                if (shp == 4):  # flip vertical + rotate left
                    temp_x[k] = 1 - temp_x[k]
                    wow = temp_x[k]
                    temp_x[k] = temp_y[k]
                    temp_y[k] = wow
                    temp_y[k] = 1 - temp_y[k]

                if (shp == 5):  # flip vertical + rotate right
                    temp_x[k] = 1 - temp_x[k]
                    wow = temp_x[k]
                    temp_x[k] = temp_y[k]
                    temp_y[k] = wow
                    temp_x[k] = 1 - temp_x[k]


            if (i == 0):        # if first image's landmarks
                Y = np.hstack((temp_x, temp_y))
                Y = Y[None, :]

            else:               # if not first image's landmarks
                temp = np.hstack((temp_x, temp_y))
                temp = temp[None, :]
                Y = np.vstack((Y, temp))

            # crop and resize to desired dimensions
            cv2.rectangle(img, (smallest_x, smallest_y), (greatest_x, greatest_y), (255, 255, 255), 3)
            crop_image = img[smallest_y:greatest_y, smallest_x:greatest_x]
            resize_image = cv2.resize(crop_image, (224, 224))

            if(shp == 1):           # flip vertical
                resize_image = cv2.flip(resize_image, 1)

            if(shp == 2):           # rotate left
                resize_image = np.rot90(resize_image)

            if(shp == 3):           # rotate right
                resize_image = np.rot90(resize_image,3)

            if (shp == 4):          # flip vertical + rotate left
                resize_image = cv2.flip(resize_image, 1)
                resize_image = np.rot90(resize_image)

            if (shp == 5):          # flip vertical + rotate right
                resize_image = cv2.flip(resize_image, 1)
                resize_image = np.rot90(resize_image, 3)

            # same the image
            cv2.imwrite(image_path, resize_image)


            # save the landmarks
            with open(landmark_path, 'w') as f:
                f.write("version: 2\n")
                f.write("n_points: 55\n")
                f.write("{\n")
                for p in range(55):
                    line = str(Y[i][p]) + " " + str(Y[i][p + 55]) + "\n"
                    f.write(line)
                f.write("}\n")
                f.close()


