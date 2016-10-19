from keras import backend as K
from keras.layers.convolutional import conv_output_length
from keras.layers.core import Layer
import numpy as np
from keras import initializations, activations


class Xception(Layer):

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='tf',
                 **kwargs):
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in ['tf', 'th']
        self.dim_ordering = dim_ordering

        self.initial_weights = weights
        self.input = K.placeholder(ndim=4)
        super(Xception, self).__init__(**kwargs)

    def build(self):
        self.conv_Ws = []

        if self.dim_ordering == 'tf':
            stack_size = self.input_shape[3]
            self.W1_shape = (1, 1, stack_size, self.nb_filter)
        elif self.dim_ordering == 'th':
            stack_size = self.input_shape[1]
            self.W1_shape = (self.nb_filter, stack_size, 1, 1)

        for i in range(self.nb_filter):
            if self.dim_ordering == 'tf':
                self.conv_Ws.append(self.init((self.nb_row, self.nb_col, 1, 1)))
            if self.dim_ordering == 'th':
                self.conv_Ws.append(self.init((1, 1, self.nb_row, self.nb_col)))

        self.W1 = self.init(self.W1_shape)
        self.b = K.zeros((self.nb_filter,))
        self.trainable_weights = [self.W1, self.b] + self.conv_Ws
        self.regularizers = []

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        conv_out = K.conv2d(X, self.W1, strides=(1, 1),
                            border_mode='same',
                            dim_ordering=self.dim_ordering)
        if self.dim_ordering == 'tf':
            channels = [conv_out[:, :, :, i:i+1] for i in range(self.nb_filter)]
        elif self.dim_ordering == 'th':
            channels = [conv_out[:, i:i+1, :, :] for i in range(self.nb_filter)]
        conv_channels = [K.conv2d(channels[i], self.conv_Ws[i],
                         strides=self.subsample, border_mode=self.border_mode,
                         dim_ordering=self.dim_ordering) for i in range(self.nb_filter)]
        if self.dim_ordering == 'tf':
            conv_out = K.concatenate(conv_channels, axis=3)
        if self.dim_ordering == 'th':
            conv_out = K.concatenate(conv_channels, axis=1)

        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        output = self.activation(output)
        return output




def test():
    from keras.datasets import mnist, cifar10
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Activation, Convolution2D, MaxPooling2D, Dropout
    from keras.utils import np_utils

    do = 'tf'

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
    # x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

    if do == 'tf':
        x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 3, 1))
        x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 3, 1))

    if do == 'th':
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

    print x_train.shape

    # x_train = x_train.astype('float32').reshape((len(x_train), 27, 27, 1)) / 255.
    # x_test = x_test.astype('float32').reshape((len(x_test), 27, 27, 1)) / 255.

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # model = Sequential()
    # model.add(Xception(32, 3, 3, border_mode='same',
    #                    activation='relu', dim_ordering=do,
    #                    input_shape=x_train.shape[1:]))
    # model.add(Xception(32, 3, 3, border_mode='same',
    #                    activation='relu', dim_ordering=do))
    # model.add(MaxPooling2D((2, 2), dim_ordering=do))

    # model.add(Xception(64, 3, 3, border_mode='same',
    #                    activation='relu', dim_ordering=do))
    # model.add(Xception(64, 3, 3, border_mode='same',
    #                    activation='relu', dim_ordering=do))
    # model.add(MaxPooling2D((2, 2), dim_ordering=do))

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering=do,
                            activation='relu',
                            input_shape=x_train.shape[1:]))
    model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering=do,
                            activation='relu'))
    model.add(MaxPooling2D((2, 2), dim_ordering=do))

    model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering=do,
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering=do,
                            activation='relu'))
    model.add(MaxPooling2D((2, 2), dim_ordering=do))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(x_train, y_train, batch_size=32, show_accuracy=True,
              validation_data=(x_test, y_test), nb_epoch=200, shuffle=True)




if __name__ == '__main__':
    test()
