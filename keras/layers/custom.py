from keras import backend as K
from keras.layers.convolutional import conv_output_length
from keras.layers.core import Layer
import numpy as np
from keras import initializations


'''
unit testing
- check that pmi computation is sensible
- check magnitude of auxiliary loss
- check that shit works

single layer testing
- check that the weights learned don't deviate too much from pmi
- check orthogonality of filters learned
- check that a very good feature extractor is learned in the absence of labels

mutiple layer testing
- check that we beat the state of the art on every possible problem
'''


class SemiSupervizedConvolution2D(Layer):

    def __init__(self, nb_filter, nb_row, nb_col, strides=(1, 1),
                 dim_ordering='tf', momentum=0.9, init='glorot_uniform', **kwargs):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.strides = strides
        self.dim_ordering = dim_ordering
        self.momentum = momentum
        self.init = initializations.get(init)

        self.input = K.placeholder(ndim=4)
        super(SemiSupervizedConvolution2D, self).__init__(**kwargs)

    def build(self):
        if self.dim_ordering == 'th':
            stack_size = self.input_shape[1]
            input_width = self.input_shape[2]
            input_height = self.input_shape[3]
            self.input_space_dim = stack_size * self.nb_row * self.nb_col
            self.kernel_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
            self.identity_kernel_shape = (self.input_space_dim, stack_size, 1, 1)
        elif self.dim_ordering == 'tf':
            stack_size = self.input_shape[3]
            input_width = self.input_shape[1]
            input_height = self.input_shape[2]
            self.input_space_dim = stack_size * self.nb_row * self.nb_col
            self.kernel_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
            self.identity_kernel_shape = (1, 1, stack_size, self.input_space_dim)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.kernel = self.init(self.kernel_shape)
        self.biases = K.zeros((self.nb_filter,))
        self.trainable_weights = [self.kernel, self.biases]

        self.feature_sums = K.zeros((self.input_space_dim,))
        self.feature_covariance_sums = K.zeros((self.input_space_dim, self.input_space_dim))
        self.non_trainable_weights = [self.feature_sums, self.feature_covariance_sums]

        np_identity_kernel = np.ones(self.input_space_dim * stack_size).reshape(self.identity_kernel_shape)
        self.identity_kernel = K.variable(np_identity_kernel)
        x = self.get_input(train=False)
        x_flat = K.conv2d(x, self.identity_kernel,
                          border_mode='same',
                          dim_ordering=self.dim_ordering,
                          image_shape=self.input_shape,
                          filter_shape=self.identity_kernel_shape)
        batch_size = self.input_shape[0]
        x_flat = K.reshape(x_flat, (batch_size * input_width * input_height, self.input_space_dim))

        feature_sums_update = self.momentum * self.feature_sums + (1 - self.momentum) * K.sum(x_flat, axis=0)
        feature_covariance_sums_update = self.momentum * self.feature_covariance_sums + (1 - self.momentum) * K.dot(K.transpose(x_flat), x_flat)
        self.updates = [(self.feature_sums, feature_sums_update),
                        (self.feature_covariance_sums, feature_covariance_sums_update)]

        self.regularizers = [self.pmi_regularizer]

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
                                  'same', self.strides[0])
        cols = conv_output_length(cols, self.nb_col,
                                  'same', self.strides[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        conv_out = K.conv2d(X, self.kernel, strides=self.strides,
                            border_mode='same',
                            dim_ordering=self.dim_ordering,
                            image_shape=self.input_shape,
                            filter_shape=self.kernel_shape)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.biases, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.biases, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = K.relu(output)
        return output

    def get_pmi(self):
        print 'feature_sums:', self.feature_sums.get_shape()
        print 'feature_covariance_sums:', self.feature_covariance_sums.get_shape()

        feature_sums = K.reshape(self.feature_sums, (self.input_space_dim, 1))
        products = K.dot(feature_sums, K.transpose(feature_sums))
        print 'products:', products.get_shape()

        pmi = products * self.feature_covariance_sums
        pmi = K.log(pmi + K.epsilon())
        pmi *= self.feature_covariance_sums
        return pmi

    def pmi_regularizer(self, loss):
        # input_space_dim = stack_size * self.nb_row * self.nb_col
        # kernel_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        # x_flat = K.reshape(x, (-1, input_space_dim))
        # match: (self.nb_row, self.nb_col, stack_size) gets resized to input_space_dim
        # in both cases (same order in kernel and x)
        k_mat = K.reshape(self.kernel, (self.input_space_dim, self.nb_filter))
        identity = K.variable(np.identity(self.input_space_dim))

        print 'k_mat:', k_mat.get_shape()

        print 'input_space_dim:', self.input_space_dim
        print k_mat.get_shape()

        embeddings = K.dot(identity, k_mat)
        pmi_star = K.dot(embeddings, K.transpose(k_mat))
        print 'pmi_star:', pmi_star.get_shape()

        pmi = self.get_pmi()
        # potential trick: multiply loss below to give more weight to lower layers
        return loss + 0.00000001 * K.mean(K.abs(pmi - pmi_star))
        #return loss + 0.1 * K.mean(K.abs(self.kernel))



def test():
    from keras.datasets import mnist, cifar10
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Activation, MaxPooling2D, Dropout, Convolution2D
    from keras.utils import np_utils
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32').reshape((len(x_train), 32, 32, 3)) / 255.
    x_test = x_test.astype('float32').reshape((len(x_test), 32, 32, 3)) / 255.
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    rows, cols = x_train.shape[1:3]

    model = Sequential()
    model.add(SemiSupervizedConvolution2D(16, 3, 3, batch_input_shape=(16, rows, cols, 3)))
    model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.25))

    model.add(SemiSupervizedConvolution2D(64, 3, 3))
    model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(x_train, y_train, batch_size=16, show_accuracy=True,
              validation_data=(x_test, y_test), nb_epoch=20)


def reference():
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Activation, Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32').reshape(x_train.shape + (1, )) / 255.
    x_test = x_test.astype('float32').reshape(x_test.shape + (1, )) / 255.
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    rows, cols = x_train.shape[1:3]

    model = Sequential()
    model.add(Convolution2D(9, 3, 3, border_mode='same',
                            dim_ordering='tf', activation='relu',
                            batch_input_shape=(16, rows, cols, 1)))
    model.add(MaxPooling2D((2, 2), dim_ordering='tf'))
    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            dim_ordering='tf', activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(x_train, y_train, batch_size=16, show_accuracy=True,
              validation_data=(x_test, y_test), nb_epoch=20)




if __name__ == '__main__':
    test()