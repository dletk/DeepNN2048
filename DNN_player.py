from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import *
from sklearn import preprocessing

import pandas as pd
import numpy as np


# Read the data into dataframe, the last column of the data is the target column
df_data = pd.read_csv("./data/transformed_data.csv",
                      sep=",", header=None).sample(frac=1)

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

num_nodes = 100

# Create the deep neural network
model_NN = Sequential()
model_NN.add(Dense(num_nodes, activation="relu", input_shape=(16,)))
model_NN.add(Dense(num_nodes, activation="relu"))
model_NN.add(Dense(num_nodes, activation="relu"))
model_NN.add(Dense(num_nodes, activation="relu"))
model_NN.add(Dense(num_nodes, activation="relu"))
model_NN.add(Dense(num_nodes, activation="relu"))
model_NN.add(Dense(num_nodes, activation="relu"))
model_NN.add(Dense(num_nodes, activation="relu"))
model_NN.add(Dense(num_nodes, activation="relu"))
# The output layer
model_NN.add(Dense(4, activation="softmax"))

# Compile the model
optimizer = SGD(lr=0.001)
model_NN.compile(optimizer=optimizer,
                 loss="categorical_crossentropy", metrics=["accuracy"])

# Fitting the model
print("===========> Begin fitting")
stopping = EarlyStopping(monitor="val_loss", patience=5)
model_NN.fit(X_data, Y_data, batch_size=512, epochs=100, validation_split=0.3)

model_NN.save("model_2048_NN.h5")

sample = np.array([[0, 4, 2, 32, 4, 2, 0, 2, 8, 0, 0, 0, 2, 0, 0, 0]])

print(model_NN.predict(sample))
