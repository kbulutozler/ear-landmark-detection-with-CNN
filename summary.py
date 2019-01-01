
from my_CNN_model import summarize_my_CNN_model, load_my_CNN_model
from keras.backend import eval

model = load_my_CNN_model('my_model')

summarize_my_CNN_model(model)

print(eval(model.optimizer.lr))