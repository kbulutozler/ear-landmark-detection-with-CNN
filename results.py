from my_CNN_model import load_my_CNN_model, test_my_CNN_model, summarize_my_CNN_model
from utilities import load_data

from landmarks import put_landmarks

model = load_my_CNN_model('my_model')

X, Y = load_data(test=True)
for i in range(0,len(X)):


    temp = X[i]
    temp = temp[None,:]
    prediction = model.predict(temp)

    for p in range(len(prediction[0])):

        prediction[0][p] = int(prediction[0][p] * 224)


    for p in range(len(Y[i])):
        Y[i][p] = int(Y[i][p] * 224)

    #put_landmarks(i, Y[i])
    put_landmarks(i, prediction[0])