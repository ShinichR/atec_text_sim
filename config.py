# coding: utf-8


class Config:
    MAX_TEXT_LENGTH = 50
    MAX_FEATURES = 10000
    embedding_dims = 128
    w2vpath = "data/baike.128.no_truncate.glove.txt"
    embedding_matrix_path = 'data/temp_no_truncate.npy'