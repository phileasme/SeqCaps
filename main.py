from keras.layers import LeakyReLU, Dense, Input
from keras.layers import Dropout, Bidirectional
from keras.layers import Flatten, SpatialDropout1D
from keras.layers import K, Activation, ConvLSTM2D
from keras.layers.recurrent import LSTM, GRU
from keras.engine import Layer
from keras.datasets import mnist
from keras.preprocessing import sequence
from keras.models import Model
from keras.utils import np_utils
from capsule_1d import *
import numpy as np
Routings = 3
Num_capsule = 10
Dim_capsule = 16
dropout = 0.25
maxlen = 784
repeats = 1
def get_model():
    input1 = Input(shape=(1,maxlen,))

    # x = Bidirectional(Dilated_TCN(1, 10, 16, 2, 1, 784, 'norm_relu', use_skip_connections=False, return_param_str=True))(input1)
    for i in range(repeats):
        # x = GRU(256, dropout=dropout,
        #                           activation='relu',
        #                           recurrent_dropout=0.28,
        #                           return_sequences=True)(input1 if i == 0 else capsule)
        x = Bidirectional(LSTM(256,
                  activation='relu',
                  dropout=dropout,
                  recurrent_dropout=.28,
                  return_sequences=True))(input1  if i == 0 else capsule)
        capsule = Capsule(
            num_capsule=Num_capsule,
            dim_capsule=Dim_capsule,
            routings=Routings)(x)


    # capsule = x
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout)(capsule)
    capsule = LeakyReLU()(capsule)

    output = Dense(10, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model

def load_seq_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    nb_classes = 10
    x_tr, x_te = x_train, x_test
    x_tr = x_train.reshape(-1, 1, 784)
    x_te = x_test.reshape(-1, 1, 784)

    y_tr = np_utils.to_categorical(y_train, nb_classes)
    y_te = np_utils.to_categorical(y_test, nb_classes)

    return x_tr, y_tr, x_te, y_te

def main():
    x_train, y_train, x_test, y_test = load_seq_mnist()

    model = get_model()

    batch_size = 32
    epochs = 50

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test))
    y_pred = model.predict(x_test, batch_size=100)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

if __name__ == '__main__':
    main()
