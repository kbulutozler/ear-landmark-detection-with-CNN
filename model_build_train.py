from utilities import load_data
from my_CNN_model import *
import cv2


# Load training set
X_train, y_train = load_data()

# Setting the CNN architecture
my_model = get_my_CNN_model_architecture()

# Compiling the CNN model with an appropriate optimizer and loss and metrics
compile_my_CNN_model(my_model, optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Training the model
hist = train_my_CNN_model(my_model, X_train, y_train)

# Saving the model
save_my_CNN_model(my_model, 'my_model')

# Saving the model
summarize_my_CNN_model(my_model)




