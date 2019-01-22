from utilities import load_data
from my_CNN_model import *
from keras.optimizers import SGD, Adam
import numpy as np

# Load
X_train, Y_train = load_data(size=3000)

# Shuffle
np.random.seed(142)
np.random.shuffle(X_train)
np.random.seed(142)
np.random.shuffle(Y_train)

# Architecture
my_model = get_my_CNN_model_architecture()

# adam optimizer
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile
compile_model(my_model, optimizer = adam, loss = 'mean_squared_error', metrics=['accuracy'])

# Train
hist = train_model(my_model, X_train, Y_train, epochs=300, batch_size=64)

# Save
save_model(my_model, 'my_model')

# Summary
summarize_model(my_model)




