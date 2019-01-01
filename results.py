from my_CNN_model import load_my_CNN_model, test_my_CNN_model, summarize_my_CNN_model
from utilities import load_data

from landmarks import put_landmarks

model = load_my_CNN_model('my_model')

X_test, Y_test = load_data(test=True)
for i in range(10):

    temp = X_test[i]
    temp = temp[None,:]
    prediction = model.predict(temp)
    print(prediction[0].shape)
    print(Y_test[i].shape)
    #put_landmarks(i, Y_test[i])
    put_landmarks(i, prediction[0])