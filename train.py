# coding: utf-8

import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

from config import Config
from ma_lstm_model import load_model, load_model_bilstm
from helper import build_vocab, read_vocab, build_vocab_data, get_embedding_matrix, f1_score_metrics, lost_function, \
    F1ScoreCallback
from time import time
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
import datetime

# File paths
from import_data import pandas_process

# define
TRAIN_CSV = 'data/atec_nlp_sim_train.csv'
TEST_CSV = 'data/atec_nlp_sim_train.csv'
MODEL_SAVING_DIR = 'data/'
questions_cols = ['question1', 'question2']
max_seq_length = 128
validation_size = 2000

# read data
train_df = pandas_process(TRAIN_CSV)
# build_vocab(train_df, "data/vocab.txt")
# words, word_to_id = read_vocab("data/vocab.txt")
# training_size = len(train_df) - validation_size


# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 128
n_epoch = 3


X_train = train_df
Y_train = train_df['score']

# X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)


tokenizer = Tokenizer(num_words=Config.MAX_FEATURES)
tokenizer.fit_on_texts(list(train_df["question1"]) + list(train_df["question2"]))

list_tokenized_question1 = tokenizer.texts_to_sequences(X_train["question1"])
list_tokenized_question2 = tokenizer.texts_to_sequences(X_train["question2"])
X_train_q1 = pad_sequences(list_tokenized_question1, maxlen=Config.MAX_TEXT_LENGTH)
X_train_q2 = pad_sequences(list_tokenized_question2, maxlen=Config.MAX_TEXT_LENGTH)

# list_tokenized_question1 = tokenizer.texts_to_sequences(X_validation.question1)
# list_tokenized_question2 = tokenizer.texts_to_sequences(X_validation.question2)
#
# X_val_q1 = pad_sequences(list_tokenized_question1, maxlen=Config.MAX_TEXT_LENGTH)
# X_val_q2 = pad_sequences(list_tokenized_question2, maxlen=Config.MAX_TEXT_LENGTH)
embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, Config.w2vpath, Config.embedding_matrix_path)


# Split to dicts
X_train = {'left': X_train_q1, 'right': X_train_q2}
Y_train = np.array(Y_train)
print (Y_train)
print (X_train['left'][0])
# X_validation = {'left':X_val_q1, 'right': X_val_q2}
# X_test = {'left': build_vocab_data(test_df.question1, word_to_id), 'right': build_vocab_data(test_df.question2, word_to_id)}

# Convert labels to their numpy representations



# Make sure everything is ok
# print("shape:{},{},{},{}".format(X_train['left'].shape,X_train['right'].shape,Y_train.shape, Y_validation.shape))
# assert X_train['left'].shape == X_train['right'].shape
# assert len(X_train['left']) == len(Y_train)


# input_shape = X_train['left'].shape
# malstm = model(input_shape, embedding_matrix1)
# Adadelta optimizer, with gradient clipping by norm
# optimizer = Adadelta(clipnorm=gradient_clipping_norm)
#
# malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy', f1_score_metrics])

def cv_train(data_q1, data_q2, label):
    seed = 20180426
    cv_folds = 10
    pred_oob = np.zeros(shape=(len(label), 1))
    from sklearn.model_selection import StratifiedKFold
    input_shape = data_q1.shape[1:]
    print("input_shape;{}".format(input_shape))

    skf = StratifiedKFold(n_splits=cv_folds, random_state=seed, shuffle=False)
    count = 0
    for ind_tr, ind_te in skf.split(data_q1, label):
        x_train_q1 = X_train_q1[ind_tr]
        x_train_q2 = X_train_q2[ind_tr]
        x_val_q1 = X_train_q1[ind_te]
        x_val_q2 = X_train_q2[ind_te]
        y_train = label[ind_tr]
        y_val = label[ind_te]
        # model = load_model_bilstm(embedding_matrix1)
        model = load_model(input_shape, embedding_matrix1)
        early_stopping = EarlyStopping(monitor='val_f1_score_metrics', patience=5, mode='max', verbose=1)
        bst_model_path = "model/malstm" + '_weight_%d.h5' % count
        model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_f1_score_metrics', mode='max',
                                           save_best_only=True, verbose=1, save_weights_only=True)
        hist = model.fit([x_train_q1, x_train_q2], y_train,
                         validation_data=([x_val_q1, x_val_q2], y_val),
                         epochs=6, batch_size=32, shuffle=True,
                         class_weight={1: 1.2833, 0: 0.4472},
                         callbacks=[early_stopping, model_checkpoint, F1ScoreCallback()])

        model.load_weights(bst_model_path)
        y_predict = model.predict([x_val_q1, x_val_q2], batch_size=256, verbose=1)
        # y_predict = model.predict([x_val_q1, x_val_q2], batch_size=256, verbose=1)
        pred_oob[ind_te] = y_predict
        # pred_oob  += y_predict
        y_predict = (y_predict > 0.5).astype(int)
        recall = recall_score(y_val, y_predict)
        print(count, "recal", recall)
        precision = precision_score(y_val, y_predict)
        print(count, "precision", precision)
        accuracy = accuracy_score(y_val, y_predict)
        print(count, "accuracy ", accuracy)
        f1 = f1_score(y_val, y_predict)
        print(count, "f1", f1)
        count += 1

    pred_oob1 = (pred_oob > 0.5).astype(int)
    recall = recall_score(label, pred_oob1)
    print("recal", recall)
    precision = precision_score(label, pred_oob1)
    print("precision", precision)
    accuracy = accuracy_score(label, pred_oob1)
    print("accuracy", accuracy)
    f1 = f1_score(label, pred_oob1)
    print("f1", f1)
    print("score",f1/2+accuracy/2)


cv_train(X_train['left'], X_train['right'], Y_train)
# Start training
# for i in range(0,3):
#     malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch, validation_data=([X_validation['left'], X_validation['right']], Y_validation), shuffle=True, class_weight={1: 1.2233, 0: 0.4472})
#
#     malstm.save_weights("model/malstm_atec_epoch_{}.h5".format(i + 1))
#
#     print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
#
#     print (malstm_trained.history['acc'])
#     print (malstm_trained.history['val_acc'])
#
#     print (malstm_trained.history['loss'])
#     print (malstm_trained.history['val_loss'])
#
