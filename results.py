from my_CNN_model import load_current_model
from utilities import load_data
from landmarks import put_landmarks

model = load_current_model('my_model')

single_img=False
X, Y = load_data(test=True, test_size=630, single_img=single_img, single_img_path='data/single/single_img.png')     # please make sure your single image consists of only ear


for i in range(0,len(X)):

    temp = X[i]
    temp = temp[None,:] # adjust the dimensions for the model
    prediction = model.predict(temp)

    for p in range(len(prediction[0])):     # adjust the landmark points for 224x224 image (they were normalized in range 0 to 1)

        prediction[0][p] = int(prediction[0][p] * 224)

    put_landmarks(i, prediction[0], single_img=False)        # the function to drop landmark points on the image