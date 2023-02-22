import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import ReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt

if __name__ == "__main__":
    IMG_ROWS = 28
    IMG_COLS = 28
    #constant
    BATCH_SIZE = 128 # nombre de données utilisée pour l'entrainement
    NB_EPOCH = 50 #nombre d'entrainmenet
    NB_CLASSES = 10
    VERBOSE = 1 #affichage
    VALIDATION_SPLIT = 0.2 #pourcentage utilisée pour tester notre modèle

    #load dataset
    X_train = np.loadtxt("./ressource/DATA/mnistData/trainMnist")
    X_train = X_train.reshape((-1, IMG_ROWS, IMG_COLS, 1))
    print("Xtrain shape : {}".format(X_train.shape))

    X_test = np.loadtxt("./ressource/DATA/mnistData/testMnist")
    X_test = X_test.reshape((-1, IMG_ROWS, IMG_COLS, 1))
    print("Xtest shape : {}".format(X_test.shape))

    Y_train = np.loadtxt("./ressource/DATA/mnistData/mnistTrainClass")
    print("y_train shape : {}".format(Y_train.shape))

    Y_test = np.loadtxt("./ressource/DATA/mnistData/mnistTestClass")
    print("y_train shape : {}".format(Y_test.shape))




    # network
    model = Sequential()
    model.add(Conv2D(filters=28, kernel_size=3, padding='same', input_shape=(IMG_ROWS, IMG_COLS, 1))) #couche donc 28 filtres, l'image va passer à travers tous ces images
    model.add(ReLU()) #couche de polarisation
    model.add(Conv2D(filters=28, kernel_size=3, padding='same', input_shape=(IMG_ROWS, IMG_COLS, 1))) #couche donc 28 filtres, l'image va passer à travers tous ces images
    model.add(ReLU()) #couche de polarisation
    model.add(MaxPooling2D(pool_size=(2, 2))) #lissage de l'image
    model.add(MaxPooling2D(pool_size=(2, 2))) #lissage de l'image
    model.add(Flatten()) # mettre à plat
    model.add(Dense(NB_CLASSES, activation=("softmax"))) # couche avec NB_CLASSES neuronnes qui donne une probabilté du chiffre, activation -> on choisit la plus grande

    model.summary()

    # train
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,verbose=VERBOSE)

    score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
    print("Test score:", score[0])
    print('Test accuracy:', score[1])



    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #save model
    model_json = model.to_json()
    open('TP.json', 'w').write(model_json)
    #And the weights learned by our deep network on the training set
    model.save_weights('TP.h5', overwrite=True)