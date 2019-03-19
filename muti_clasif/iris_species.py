import numpy as np
import keras as K
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    print("Iris dataset using Keras")
    np.random.seed(4)
    tf.set_random_seed(13)

    #Load data
    print("Loading iris data into memory \n")
    train_file = ''
    test_file = ''

    train_x = np.loadtxt(train_file, usecols=[0,1,2,3], delimiter=",", skiprows=0, dtype=np.float32)
    train_y = np.loadtxt(train_file, usecols=[4,5,6], delimiter=",", skiprows=0, dtype=np.float32)

    test_x = np.loadtxt(test_file, usecols=range(0,4), delimiter=",", skiprows=0, dtype=np.float32)
    test_y = np.loadtxt(test_file, usecols=range(4,7), delimiter=",", skiprows=0, dtype=np.float32)

    #model
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam()
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=5, input_dim=4, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=6,  kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=3, kernel_initializer=init, activation='sofmax'))
    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

    #train model
    b_size = 1
    max_epochs = 10
    print("Starting training")
    h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
    print("Training finished \n")

    #4 evaluate model



