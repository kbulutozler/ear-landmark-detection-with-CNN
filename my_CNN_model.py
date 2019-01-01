from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

def get_my_CNN_model_architecture():


    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(200, 200, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))




    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))

    model.add(Dense(110))


    return model

def compile_my_CNN_model(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def train_my_CNN_model(model, X_train, y_train):
    return model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1, validation_split=0.1)

def save_my_CNN_model(model, fileName):
    model.save(fileName + '.h5')

def load_my_CNN_model(fileName):
    return load_model(fileName + '.h5')

def test_my_CNN_model(model, X_test, Y_test):
    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

def summarize_my_CNN_model(model):
    model.summary()
