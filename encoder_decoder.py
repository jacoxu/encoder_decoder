# -*- encoding:utf-8 -*-
"""
    测试Encoder-Decoder 2016/03/22
"""
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import RepeatVector, TimeDistributedDense, Activation
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder
import time
import numpy as np
import re

__author__ = 'http://jacoxu.com'


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def vectorize_stories(input_list, tar_list, word_idx, input_maxlen, tar_maxlen, vocab_size):
    x_set = []
    Y = np.zeros((len(tar_list), tar_maxlen, vocab_size), dtype=np.bool)
    for _sent in input_list:
        x = [word_idx[w] for w in _sent]
        x_set.append(x)
    for s_index, tar_tmp in enumerate(tar_list):
        for t_index, token in enumerate(tar_tmp):
            Y[s_index, t_index, word_idx[token]] = 1

    return pad_sequences(x_set, maxlen=input_maxlen), Y


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def main():
    input_text = ['1 2 3 4 5'
                  , '6 7 8 9 10'
                  , '11 12 13 14 15'
                  , '16 17 18 19 20'
                  , '21 22 23 24 25']
    tar_text = ['one two three four five'
                , 'six seven eight nine ten'
                , 'eleven twelve thirteen fourteen fifteen'
                , 'sixteen seventeen eighteen nineteen twenty'
                , 'twenty_one twenty_two twenty_three twenty_four twenty_five']

    input_list = []
    tar_list = []

    for tmp_input in input_text:
        input_list.append(tokenize(tmp_input))
    for tmp_tar in tar_text:
        tar_list.append(tokenize(tmp_tar))

    vocab = sorted(reduce(lambda x, y: x | y, (set(tmp_list) for tmp_list in input_list + tar_list)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1  # keras进行embedding的时候必须进行len(vocab)+1
    input_maxlen = max(map(len, (x for x in input_list)))
    tar_maxlen = max(map(len, (x for x in tar_list)))
    output_dim = vocab_size
    hidden_dim = 20

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Input max length:', input_maxlen, 'words')
    print('Target max length:', tar_maxlen, 'words')
    print('Dimension of hidden vectors:', hidden_dim)
    print('Number of training stories:', len(input_list))
    print('Number of test stories:', len(input_list))
    print('-')
    print('Vectorizing the word sequences...')
    word_to_idx = dict((c, i + 1) for i, c in enumerate(vocab))  # 编码时需要将字符映射成数字index
    idx_to_word = dict((i + 1, c) for i, c in enumerate(vocab))  # 解码时需要将数字index映射成字符
    inputs_train, tars_train = vectorize_stories(input_list, tar_list, word_to_idx, input_maxlen, tar_maxlen, vocab_size)

    decoder_mode = 1  # 0 最简单模式，1 [1]向后模式，2 [2] Peek模式，3 [3]Attention模式
    if decoder_mode == 3:
        encoder_top_layer = LSTM(hidden_dim, return_sequences=True)
    else:
        encoder_top_layer = LSTM(hidden_dim)

    if decoder_mode == 0:
        decoder_top_layer = LSTM(hidden_dim, return_sequences=True)
        decoder_top_layer.get_weights()
    elif decoder_mode == 1:
        decoder_top_layer = LSTMDecoder(hidden_dim=hidden_dim, output_dim=hidden_dim
                                        , output_length=tar_maxlen, state_input=False, return_sequences=True)
    elif decoder_mode == 2:
        decoder_top_layer = LSTMDecoder2(hidden_dim=hidden_dim, output_dim=hidden_dim
                                         , output_length=tar_maxlen, state_input=False, return_sequences=True)
    elif decoder_mode == 3:
        decoder_top_layer = AttentionDecoder(hidden_dim=hidden_dim, output_dim=hidden_dim
                                             , output_length=tar_maxlen, state_input=False, return_sequences=True)

    en_de_model = Sequential()
    en_de_model.add(Embedding(input_dim=vocab_size,
                              output_dim=hidden_dim,
                              input_length=input_maxlen))
    en_de_model.add(encoder_top_layer)
    if decoder_mode == 0:
        en_de_model.add(RepeatVector(tar_maxlen))
    en_de_model.add(decoder_top_layer)

    en_de_model.add(TimeDistributedDense(output_dim))
    en_de_model.add(Activation('softmax'))
    print('Compiling...')
    time_start = time.time()
    en_de_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    time_end = time.time()
    print('Compiled, cost time:%fsecond!' % (time_end - time_start))
    for iter_num in range(5000):
        en_de_model.fit(inputs_train, tars_train, batch_size=3, nb_epoch=1, show_accuracy=True)
        out_predicts = en_de_model.predict(inputs_train)
        for i_idx, out_predict in enumerate(out_predicts):
            predict_sequence = []
            for predict_vector in out_predict:
                next_index = np.argmax(predict_vector)
                next_token = idx_to_word[next_index]
                predict_sequence.append(next_token)
            print('Target output:', tar_text[i_idx])
            print('Predict output:', predict_sequence)

        print('Current iter_num is:%d' % iter_num)

if __name__ == '__main__':
    main()
