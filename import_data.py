#!/usr/bin/python
# coding:utf8
import codecs
from helper import text_to_word_list_by_jieba


def read_file(file_name):
    qa_pairs = []
    result = []
    with open(file_name, "r") as fr:
        for line in fr:
            words = line.split("\t")
            # print ("{},{},{}".format(words[1], words[2], words[3]))
            qa_pairs.append("{}\t{}".format(words[1], words[2]))
            result.append(words[3])

    return qa_pairs, result

def pandas_process(input_train):
    q1 = []
    q2 = []
    scores = []
    fin = codecs.open(input_train, 'r', encoding='utf-8')
    fin.readline()
    for line in fin:
        id, sen1, sen2, score = line.strip().split('\t')
        q1.append(text_to_word_list_by_jieba(sen1))
        q2.append(text_to_word_list_by_jieba(sen2))
        scores.append(int(score))
    fin.close()
    df = {"question1": q1, "question2": q1, "score":scores}
    return df

def read_test(input_file):
    q1 = []
    q2 = []
    ids = []
    fin = codecs.open(input_file, 'r', encoding='utf-8')
    fin.readline()
    for line in fin:
        id, sen1, sen2 = line.strip().split('\t')
        q1.append(text_to_word_list_by_jieba(sen1))
        q2.append(text_to_word_list_by_jieba(sen2))
        ids.append(id)
    fin.close()
    df = {"question1": q1, "question2": q1, "noid":ids}
    return df

#
# data = pd.read_csv('/Users/lei/work/alipay/atec_nlp_sim_train.csv', error_bad_lines=False)
# print data
# read_file('/Users/lei/work/alipay/atec_nlp_sim_train.csv')
