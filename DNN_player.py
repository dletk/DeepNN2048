from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import *
from sklearn import preprocessing

from keras.models import load_model

import pandas as pd
import numpy as np


# Read the data into dataframe, the last column of the data is the target column
df_data = pd.read_csv("./all_2048_8000games.csv",
                      sep=",", header=None)

# Drop duplicate board, so the network will not be confused if 2 boards with same situation has 2 different move
df_data.drop_duplicates(
    subset=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], inplace=True)

# Get the feature data and prepare the proper format for keras
X_data = df_data.drop(16, 1)
# X_data = preprocessing.scale(X_data, axis=1)
X_data = X_data.as_matrix()

# Prepare the target data
Y_data = df_data[16]
# print(Y_data.describe())
Y_data = to_categorical(Y_data)


"""
This part is used to create the initial model
"""
# Create the deep neural network
# model_NN = Sequential()
# model_NN.add(Dense(900, activation="relu", input_shape=(16,)))
# model_NN.add(Dense(300, activation="relu"))
# model_NN.add(Dense(200, activation="relu"))
# model_NN.add(Dense(100, activation="relu"))
# model_NN.add(Dense(16, activation="relu"))

# The output layer
# model_NN.add(Dense(4, activation="softmax"))

# Compile the model
# optimizer = Adam(lr=0.0001)
# model_NN.compile(optimizer=optimizer,
#                  loss="categorical_crossentropy", metrics=["accuracy"])

"""
This part is used to load and keep training the current model
"""

# Load the current model in to keep training
model_NN = load_model("all_2048_8000games.h5")

# Fitting the model
print("===========> Begin fitting")
stopping = EarlyStopping(monitor="val_loss", patience=5)
model_NN.fit(X_data, Y_data, batch_size=512, epochs=100, validation_split=0.2)

model_NN.save("all_2048_8000games_cont_training.h5")

sample = np.array([[0, 4, 2, 32, 4, 2, 0, 2, 8, 0, 0, 0, 2, 0, 0, 0]])

print(model_NN.predict(sample))
