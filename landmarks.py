import cv2
import matplotlib.pyplot as plt


def put_landmarks(i, pred):
    img_path = 'data/train/c_images/c_train_' + str(i) + '.png'

    img_original = plt.imread(img_path)

    for j in range(0,55):
        plt.scatter([pred[j]], [pred[j+55]])
        #print(str(pred[j]) + ' ' + str(pred[j+55]))


    plt.imshow(img_original)
    img_result_path = 'data/train/results/result_' + str(i) + '.png'
    plt.savefig(img_result_path)
    plt.close()
