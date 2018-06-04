# coding: utf-8
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge, Dense, Dropout, Bidirectional

# Model variables
from config import Config
from helper import f1_score_metrics

n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 128
n_epoch = 3
max_seq_length = 128
rate_drop_dense = 0.2

def exponent_neg_manhattan_distance(left, right):
    # left, right = args
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def load_model(shape, embedding_matrix):


    # Creating word embedding layer
    embedding_layer = Embedding(Config.MAX_FEATURES, Config.embedding_dims, weights=[embedding_matrix],
                                input_length=Config.MAX_TEXT_LENGTH, trainable=True)
    # Creating LSTM Encoder
    lstm_layer = Bidirectional(
        LSTM(n_hidden, dropout=rate_drop_dense, recurrent_dropout=rate_drop_dense))


    # The visible layer
    left_input = Input(shape=shape, dtype='float32', name = 'input_1')
    right_input = Input(shape=shape, dtype='float32', name = 'input_2')

    # Embedded version of the inputs
    embedded_sequences_1 = embedding_layer(left_input)
    embedded_sequences_2 = embedding_layer(right_input)

    x1 = lstm_layer(embedded_sequences_1)
    x2 = lstm_layer(embedded_sequences_2)


    # encoded_left = Embedding(output_dim=512, input_dim=1000, input_length=max_seq_length)(left_input)
    # encoded_right = Embedding(output_dim=512, input_dim=1000, input_length=max_seq_length)(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    # shared_lstm = LSTM(n_hidden,activation = 'relu', name = 'lstm_1_2')
    # x = Flatten()(input)

    # left_output = shared_lstm(encoded_left)
    # right_output = shared_lstm(encoded_right)

    # malstm_distance = Lambda(exponent_neg_manhattan_distance,
    #        output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                            output_shape=lambda x: (x[0][0], 1))([x1, x2])
    # malstm_distance = merge([processed_a, processed_b], mode='sum')
    # dense1_layer = Dense(16, activation='sigmoid')(malstm_distance)
    # predictions = Dense(1, activation='sigmoid')(dense1_layer)
    # bn = BatchNormalization(drop_out)
    # predictions = Dense(1, activation='sigmoid')(drop_out)
    # yes_or_no = Dense(1, activation='sigmoid')(predictions)
    # Calculates the distance as defined by the MaLSTM model
    # malstm_distance = keras.layers.merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    # Pack it all up into a model
    malstm = Model([left_input, right_input], outputs=malstm_distance)
    malstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy", f1_score_metrics])
    malstm.summary()
    # predictions = Dense(1, activation='sigmoid')(merged_vector)
    return malstm

def load_model_bilstm(embedding_matrix):
    input1_tensor = keras.layers.Input(shape=(Config.MAX_TEXT_LENGTH,))
    input2_tensor = keras.layers.Input(shape=(Config.MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(Config.MAX_FEATURES, Config.embedding_dims,
                                                   weights=[embedding_matrix],
                                                   input_length=Config.MAX_TEXT_LENGTH,
                                                   trainable=True)
    seq_embedding_layer = keras.layers.Bidirectional(keras.layers.GRU(Config.MAX_TEXT_LENGTH, recurrent_dropout=rate_drop_dense))
    seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))
    merge_layer = keras.layers.multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])
    merge_layer = keras.layers.Dropout(rate_drop_dense)(merge_layer)
    dense1_layer = keras.layers.Dense(64, activation='relu')(merge_layer)
    ouput_layer = keras.layers.Dense(1, activation='sigmoid')(dense1_layer)
    model = keras.models.Model([input1_tensor, input2_tensor], ouput_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy", f1_score_metrics])
    model.summary()
    return model