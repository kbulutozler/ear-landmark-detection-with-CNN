from my_CNN_model import load_my_CNN_model, test_my_CNN_model
from utilities import load_data

X_test, Y_test = load_data(test=True)
model = load_my_CNN_model('my_model')

test_my_CNN_model(model, X_test, Y_test)