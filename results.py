from my_CNN_model import load_my_CNN_model, test_my_CNN_model, summarize_my_CNN_model
from utilities import load_data

from landmarks import put_landmarks

#model = load_my_CNN_model('my_model')

X, Y = load_data(test=False)
for i in range(0,len(X)):

    temp = X[i]
    temp = temp[None,:]
    #prediction = model.predict(temp)

    put_landmarks(i, Y[i])
    #put_landmarks(i, prediction[0])