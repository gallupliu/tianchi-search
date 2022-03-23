#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:gallup
@file:preprocess.py
@time:2022/03/14
"""
import random
import tqdm
import pandas as pd
import numpy as np


# from pyspark.sql.types import *
# from pyspark import SparkConf, SparkContext
# from pyspark.sql.functions import col, lit, split, udf, concat, concat_ws, when, count, desc, row_number
# from pyspark.sql import functions as F, Window
# from pyspark.sql.types import ArrayType, DoubleType, FloatType, StringType, IntegerType
from sklearn.model_selection import train_test_split

def load_raw_data(file_path):
    """加载原始数据"""
    data = pd.read_csv(file_path, sep='\t')
    return data


def load_txt_data(file_path):
    D = []
    with open(file_path, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 2:
                D.append([int(l[0]), l[1]])
    return pd.DataFrame(D, columns=['query_id', 'query'])


def generate_pos_data(query_df, doc_df, qrele_df):
    """
    生成正样本
    :param query_df:
    :param doc_df:train_pos_df
    :param qrele_df:
    :return: return list such as [query_id,query,doc_id,doc]
    """
    df = qrele_df.merge(query_df, on=['query_id'], how='left')
    df = df.merge(doc_df, on=['doc_id'], how='left')
    return df


def gen_data_set(data, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    print(data.columns)
    print(data.groupby('user_id'))
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + negi], 0, len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))

    # random.shuffle(train_set)
    # random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


def gen_data_set_by_random_neg_sample(data, negsample=3):
    item_ids = data['doc_id'].unique()

    train_set = []
    for query_id, hist in data.groupby('query_id'):

        pos_list = hist['doc_id'].tolist()  # 每个query对应的正样本列表
        # print('id:{0} hist:{1}'.format(query_id, pos_list))
        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))  # 每个query对应的候选非正样本列表
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample,
                                        replace=True)  # 每个query对应的候选负样本列表

        train_set.append([query_id, pos_list[0], 1])
        for doc_id in neg_list.tolist():
            train_set.append([query_id, doc_id, 0])
    # random.shuffle(train_set)

    print(len(train_set))

    return train_set


def generate_train_data(df, doc_df, sample_type):
    """
    生成训练数据
    :param df:
    :param doc_df:
    :param sample_type:
    :return: return list such as [query_id,query,doc_id,title，label]
    """
    pass


if __name__ == '__main__':
    base_path = 'data/'
    data_path = '/Users/gallup/data/taobao_search/'
    doc_df = load_raw_data(data_path + 'corpus.tsv')
    doc_df.columns = ['doc_id', 'title']
    train_query_df = load_txt_data(data_path + 'train.query.txt')
    dev_query_df = load_txt_data(data_path + 'dev.query.txt')
    qrele_df = load_raw_data(data_path + 'qrels.train.tsv')
    qrele_df.columns = ['query_id', 'doc_id']
    print(doc_df.columns, train_query_df.columns, qrele_df.columns)
    print(len(doc_df), len(train_query_df), len(qrele_df))
    train_pos_df = generate_pos_data(train_query_df, doc_df, qrele_df)
    print(train_pos_df.columns, len(train_pos_df))
    print(train_pos_df.groupby('query_id'))
    # Index(['user_id', 'movie_id', 'rating', 'timestamp', 'title', 'genres',
    #        'gender', 'age', 'occupation', 'zip'],
    #       dtype='object')
    # < pandas.core.groupby.generic.DataFrameGroupBy
    # object
    # at
    # 0x7fa9e3813940 >

    train_set = gen_data_set_by_random_neg_sample(qrele_df, negsample=3)
    train_df = pd.DataFrame(train_set, columns=['query_id', 'doc_id', 'label'])
    df = generate_pos_data(train_query_df, doc_df, train_df)
    # train_df.to_csv('./train.csv', index=False)

    train_df, test_df = train_test_split(df.loc[:,["query", "title", "label"]], test_size=0.2, random_state=1)

    train_df.to_csv('./tianchi.train.data.csv', sep='\t', index=False)
    test_df.to_csv('./tianchi.test.data.csv', sep='\t', index=False)
