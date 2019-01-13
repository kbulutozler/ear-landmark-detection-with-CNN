from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense

def get_my_CNN_model_architecture():
    model = Sequential()
    model.add(Conv2D(8, (3, 3), input_shape=(300, 300, 3), kernel_initializer='random_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))









    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(110))

    return model

def compile_my_CNN_model(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def train_my_CNN_model(model, X_train, Y_train, epochs, batch_size):
    return model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

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
