
from my_CNN_model import summarize_model, load_current_model
from keras.backend import eval

model = load_current_model('my_model')

summarize_model(model)

print(eval(model.optimizer.lr))