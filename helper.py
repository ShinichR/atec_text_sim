# coding: utf-8

import sys
import os
import jieba
import mmap

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from config import Config

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False
from collections import Counter

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

jieba.add_word('花呗')
jieba.add_word('借呗')
jieba.add_word('余额宝')


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''

    if not is_py3 and isinstance(text, str):
        text = text.decode('utf-8')
    text_list = list(text)
    return text_list

def text_to_word_list_by_jieba(text):
    ''' Pre process and convert texts to a list of words '''

    if not is_py3 and isinstance(text, str):
        text = text.decode('utf-8')
    seg_list = jieba.cut(text)
    return [seg for seg in seg_list]

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

def build_vocab(data_df, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    # data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_df.question1:
        all_data.extend(text_to_word_list(content))

    for content in data_df.question2:
        all_data.extend(text_to_word_list(content))

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip()  for _ in fp.readlines()]
    word_id = list(zip(words, range(len(words))))
    word_to_id = dict(word_id)
    return words, word_to_id

def build_vocab_data(question, word_to_id, max_length=600):
    """将文件转换为id表示"""
    question_list = np.array(question)
    data_id = []
    for i in range(len(question_list)):
        data_id.append([word_to_id[x] for x in question_list[i] if x in word_to_id])
    # print (data_id[0])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = pad_sequences(data_id, max_length)
    # print (x_pad[0])
    return x_pad

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_embedding_matrix(word_index, Emed_path, Embed_npy):
    if (os.path.exists(Embed_npy)):
        return np.load(Embed_npy)
    print('Indexing word vectors')
    embeddings_index = {}
    file_line = get_num_lines(Emed_path)
    print('lines ', file_line)
    with open(Emed_path, encoding='utf-8') as f:
        for line in tqdm(f, total=file_line):
            values = line.split()
            if (len(values) < Config.embedding_dims):
                print(values)
                continue
            word = ' '.join(values[:-Config.embedding_dims])
            coefs = np.asarray(values[-Config.embedding_dims:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))
    print('Preparing embedding matrix')
    nb_words = Config.MAX_FEATURES  # min(MAX_FEATURES, len(word_index))
    all_embs = np.stack(embeddings_index.values())
    print(all_embs.shape)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(loc=emb_mean, scale=emb_std, size=(nb_words, Config.embedding_dims))

    # embedding_matrix = np.zeros((nb_words, embedding_dims))
    count = 0
    for word, i in tqdm(word_index.items()):
        if i >= Config.MAX_FEATURES:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            count += 1
    np.save(Embed_npy, embedding_matrix)
    print('Null word embeddings: %d' % (nb_words - count))
    print('not Null word embeddings: %d' % count)
    print('embedding_matrix shape', embedding_matrix.shape)
    # print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


def f1_score_metrics(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def lost_function(y_true, y_pred):
    keras_var = K.variable(0.5)
    # y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
    y_pred = K.greater(y_pred, keras_var)
    y_pred = K.cast_to_floatx(y_pred)
    return K.mean(K.square(y_pred - y_true), axis=-1)


class F1ScoreCallback(Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(F1ScoreCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if (self.validation_data):
            y_predict = self.model.predict([self.validation_data[0], self.validation_data[1]],
                                           batch_size=self.predict_batch_size)
            y_predict = (y_predict > 0.5).astype(int)
            accuracy=accuracy_score(self.validation_data[2], y_predict)
            precision=precision_score(self.validation_data[2], y_predict)
            recall = recall_score(self.validation_data[2], y_predict)
            f1 = f1_score(self.validation_data[2], y_predict)
            print("precision %.3f recall %.3f f1_score %.3f accuracy %.3f "% (precision, recall,f1,accuracy))



