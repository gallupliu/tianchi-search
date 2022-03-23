#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:gallup
@file:tett.py
@time:2022/03/22
"""


from utils import *
import sys
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
from keras.layers import Dropout, Dense
import jieba

jieba.initialize()

from preprocess import load_raw_data, load_txt_data

# 基本参数
model_type = 'RoFormer'
pooling = 'cls'
task_name = 'tianchi'
dropout_rate = 0.3
assert model_type in [
    'BERT', 'RoBERTa', 'NEZHA', 'WoBERT', 'RoFormer', 'BERT-large',
    'RoBERTa-large', 'NEZHA-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small'
]
assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']

dropout_rate = float(dropout_rate)

if task_name == 'PAWSX':
    maxlen = 128
else:
    maxlen = 64

# 加载数据集
data_path = './drive/MyDrive/data'

datasets = {
    '%s-%s' % (task_name, f):
        load_data('%s/%s.%s.data' % (data_path, task_name, f))
    for f in ['train', 'test']
}

# bert配置
model_name = {
    'BERT': 'chinese_L-12_H-768_A-12',
    'RoBERTa': 'chinese_roberta_wwm_ext_L-12_H-768_A-12',
    'WoBERT': 'chinese_wobert_plus_L-12_H-768_A-12',
    'NEZHA': 'nezha_base_wwm',
    'RoFormer': 'chinese_roformer-sim-char_L-12_H-768_A-12',
    'BERT-large': 'uer/mixed_corpus_bert_large_model',
    'RoBERTa-large': 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16',
    'NEZHA-large': 'nezha_large_wwm',
    'SimBERT': 'chinese_simbert_L-12_H-768_A-12',
    'SimBERT-tiny': 'chinese_simbert_L-4_H-312_A-12',
    'SimBERT-small': 'chinese_simbert_L-6_H-384_A-12'
}[model_type]

config_path = './drive/MyDrive/%s/bert_config.json' % model_name
if model_type == 'NEZHA':
    checkpoint_path = './drive/MyDrive/%s/model.ckpt-691689' % model_name
elif model_type == 'NEZHA-large':
    checkpoint_path = './drive/MyDrive/%s/model.ckpt-346400' % model_name
else:
    checkpoint_path = './drive/MyDrive/%s/bert_model.ckpt' % model_name
dict_path = './drive/MyDrive/%s/vocab.txt' % model_name

# 建立分词器
if model_type in ['WoBERT', 'RoFormer']:
    tokenizer = get_tokenizer(
        dict_path, pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
else:
    tokenizer = get_tokenizer(dict_path)

# 建立模型
if model_type == 'RoFormer':
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='roformer',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
elif 'NEZHA' in model_type:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='nezha',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
else:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        pooling=pooling,
        dropout_rate=dropout_rate
    )

# 语料id化
all_names, all_weights, all_token_ids, all_labels = [], [], [], []
train_token_ids = []
for name, data in datasets.items():
    a_token_ids, b_token_ids, labels = convert_to_ids(data, tokenizer, maxlen)
    all_names.append(name)
    all_weights.append(len(data))
    all_token_ids.append((a_token_ids, b_token_ids))
    all_labels.append(labels)
    train_token_ids.extend(a_token_ids)
    train_token_ids.extend(b_token_ids)

if task_name != 'PAWSX':
    np.random.shuffle(train_token_ids)
    train_token_ids = train_token_ids[:10000]


class data_generator(DataGenerator):
    """训练语料生成器
    """

    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []


def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


# SimCSE训练
encoder.summary()
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
train_generator = data_generator(train_token_ids, 64)
encoder.fit(
    train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=1
)


#
# def evaluate(data):
#     total, right = 0., 0.
#     for x_true, y_true in data:
#         y_pred = model.predict(x_true).argmax(axis=1)
#         # y_true = y_true[:, 0]
#         y_true = np.argmax(y_true, axis=1)
#         total += len(y_true)
#         right += (y_true == y_pred).sum()
#     return right / total
#
#
# valid_generator = data_generator(test, batch_size=1)
#
#
# class Evaluator(keras.callbacks.Callback):
#     """评估与保存
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.best_val_acc = 0.
#
#     def on_epoch_end(self, epoch, logs=None):
#         val_acc = evaluate(valid_generator)
#         print('val_acc:{0},best_val_acc:{1}\n'.format(val_acc, self.best_val_acc))
#         if val_acc > self.best_val_acc:
#             self.best_val_acc = val_acc
#             query_model.save_weights('./poly_encoder/query_model.weights')
#             doc_model.save_weights('./poly_encoder/doc_model.weights')
#             model.save_weights('./poly_encoder/model.weights')
#             encoder.model.save_weights('./poly_encoder/encoder.weights')

# 语料向量化
all_vecs = []
for a_token_ids, b_token_ids in all_token_ids:
    a_vecs = encoder.predict([a_token_ids,
                              np.zeros_like(a_token_ids)],
                             verbose=True)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             verbose=True)
    all_vecs.append((a_vecs, b_vecs))

# 标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    # convert to 128ggggtbn  fs
    a_vecs = Dense(units=128,
                   activation='relu',
                   kernel_initializer=encoder.initializer)(a_vecs)
    b_vecs = Dense(units=128,
                   activation='relu',
                   kernel_initializer=encoder.initializer)(b_vecs)

    a_vecs = l2_normalize(a_vecs)
    b_vecs = l2_normalize(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))

# predict
base_path = './drive/MyDrive/'
data_path = './drive/MyDrive/taobao_search/'

doc_df = load_raw_data(data_path + 'corpus.tsv')
doc_df.columns = ['doc_id', 'title']
dev_query_df = load_txt_data(data_path + 'dev.query.txt')

dev_query_df = dev_query_df.loc[:, ["query"]]
for text in dev_query_df:
    query_token_ids = convert_single_text_to_ids(text, tokenizer, maxlen)

with open(data_path + 'query_embedding', 'w') as fout:
    for i, token_ids in enumerate(query_token_ids):
        vec = encoder.predict([token_ids,
                               np.zeros_like(token_ids)],
                              verbose=True)

        fout.write(str(i) + "\t" + vec)

title_df = doc_df.loc[:, ["title"]]
for text in title_df:
    doc_token_ids = convert_single_text_to_ids(text, tokenizer, maxlen)

doc_vecs = []
with open(data_path + 'doc_embedding', 'w') as fout:
    for i, token_ids in enumerate(doc_token_ids):
        doc_vec = encoder.predict([token_ids,
                                   np.zeros_like(token_ids)],
                                  verbose=True)

        fout.write(str(i) + "\t" + doc_vec)

if __name__ == '__main__':
    pass
