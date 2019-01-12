from utilities import load_data, soft_acc
from my_CNN_model import *
from keras.optimizers import SGD, Adam



# Load training set
X_train, Y_train = load_data(size=500)

# Setting the CNN architecture
my_model = get_my_CNN_model_architecture()

# Compiling the CNN model with an appropriate optimizer and loss and metrics
learning_rate = 0.1
epochs = 100
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

compile_my_CNN_model(my_model, optimizer = adam, loss = 'mean_squared_error', metrics=['accuracy'])

# Training the model
hist = train_my_CNN_model(my_model, X_train, Y_train, epochs=100, batch_size=64)

# Saving the model
save_my_CNN_model(my_model, 'my_model')

# Summary of the model
summarize_my_CNN_model(my_model)




