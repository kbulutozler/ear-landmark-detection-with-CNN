import cv2
from utilities import adjustDots
for i in range(15):
    image_name = 'data/train/images/train_' + str(i) + '.png'
    save_folder_name = 'data/train/c_images/'
    left_ear_cascade = cv2.CascadeClassifier('crop files/haarcascade_mcs_leftear.xml')
    right_ear_cascade = cv2.CascadeClassifier('crop files/haarcascade_mcs_rightear.xml')
    if left_ear_cascade.empty():
        raise IOError('Unable to load the left ear cascade classifier xml file')
    if right_ear_cascade.empty():
        raise IOError('Unable to load the right ear cascade classifier xml file')
    img = cv2.imread(image_name)
    img2 = img[:,:,:].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left_ear = left_ear_cascade.detectMultiScale(gray, 1.1, 5)
    right_ear = right_ear_cascade.detectMultiScale(gray, 1.1, 5)
    resize_image = img
    for (x, y, w, h) in left_ear:
            print("left " + str(i))

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            crop_image = img2[(y - 15):(y + h + 15), (x - 15):(x + w + 15)]
            height, width, channels = crop_image.shape
            adjustDots(x - 15, y - 15, width, height, 224, i, 'data/train/c_landmarks/c_train_')
            resize_image = cv2.resize(crop_image, (224, 224))
            cv2.imwrite(save_folder_name + 'c_train_' + str(i) + '.png', resize_image)

    for (x, y, w, h) in right_ear:
            print("right " + str(i))

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            crop_image = img2[(y - 15):(y + h + 15), (x - 15):(x + w + 15)]
            height, width, channels = crop_image.shape
            adjustDots(x - 15, y - 15, width, height, 224, i, 'data/train/c_landmarks/c_train_')
            resize_image = cv2.resize(crop_image, (224, 224))
            cv2.imwrite(save_folder_name + 'c_train_' + str(i) + '.png', resize_image)


