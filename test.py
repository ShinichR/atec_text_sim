# coding: utf-8
import codecs
import sys

from keras.preprocessing.sequence import pad_sequences

from config import Config
from ma_lstm_model import load_model_bilstm
from helper import build_vocab, read_vocab, build_vocab_data, get_embedding_matrix, f1_score_metrics
from time import time
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
import datetime

# File paths
from import_data import read_test, pandas_process

# define
TRAIN_CSV = 'data/atec_nlp_sim_train.csv'
TEST_CSV = 'data/atec_nlp_sim_train.csv'
MODEL_SAVING_DIR = 'data/'
questions_cols = ['question1', 'question2']
max_seq_length = 128
validation_size = 5000

# read data
train_df = pandas_process(TRAIN_CSV)
# build_vocab(train_df, "data/vocab.txt")
# words, word_to_id = read_vocab("data/vocab.txt")
# training_size = len(train_df) - validation_size


# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 32
n_epoch = 3

tokenizer = Tokenizer(num_words=Config.MAX_FEATURES)
tokenizer.fit_on_texts(list(train_df["question1"]) + list(train_df["question2"]))

#list_tokenized_question1 = tokenizer.texts_to_sequences(X_train["question1"])
#list_tokenized_question2 = tokenizer.texts_to_sequences(X_train["question2"])
#X_train_q1 = pad_sequences(list_tokenized_question1, maxlen=Config.MAX_TEXT_LENGTH)
#X_train_q2 = pad_sequences(list_tokenized_question2, maxlen=Config.MAX_TEXT_LENGTH)

embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, Config.w2vpath, Config.embedding_matrix_path)

def cv_test(data_q1, data_q2):
    seed = 20180426
    cv_folds = 10
    pred_oob = np.zeros(shape=(len(data_q1), 1))
    from sklearn.model_selection import StratifiedKFold
    input_shape = data_q1.shape[1:]
    print("input_shape;{}".format(input_shape))
    model = load_model_bilstm(embedding_matrix1)
    count = 0
    for index in range(cv_folds):
        bst_model_path = "bimodel/bilstm" + '_weight_%d.h5' % count
        model.load_weights(bst_model_path)
        y_predict = model.predict([data_q1, data_q2], batch_size=1024, verbose=0)
        pred_oob += y_predict
        count += 1
    pred_oob /= cv_folds
    pred_oob1 = (pred_oob > 0.5).astype(int)
    return pred_oob1
   
if __name__ == '__main__':
    in_path = sys.argv[1]
    output_path = sys.argv[2]
    # define
    TEST_CSV = in_path
    MODEL_SAVING_DIR = 'data/'
    max_seq_length = 128
    # Model variables
    n_hidden = 50
    gradient_clipping_norm = 1.25
    batch_size = 32
    n_epoch = 3

    # read data
    test_df = read_test(TEST_CSV)

    list_tokenized_question1 = tokenizer.texts_to_sequences(test_df["question1"])
    list_tokenized_question2 = tokenizer.texts_to_sequences(test_df["question2"])

    X_val_q1 = pad_sequences(list_tokenized_question1, maxlen=Config.MAX_TEXT_LENGTH)
    X_val_q2 = pad_sequences(list_tokenized_question2, maxlen=Config.MAX_TEXT_LENGTH)


    # Split to dicts
    X_test = {'left': X_val_q1, 'right': X_val_q2}
    # Convert labels to their numpy representations
    print (X_test['left'][0])

    pred_oob = cv_test(X_test['left'], X_test['right'])
    fout = codecs.open(output_path, 'w', encoding='utf-8')
    for index, la in enumerate(pred_oob):
        lineno = test_df['noid'][index]
        fout.write(lineno + '\t%d\n' % la)
    fout.close()


	
#

    #
