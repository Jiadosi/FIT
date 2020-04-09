
import pandas as pd     # 一个数据处理的库，一般用来做预处理
import numpy as np      # 科学计算的库，numpy里的数据类型可以直接输入模型
'''
import keras.backend as K       # keras是一个机器学习的框架，类似tensorflow，封装了一些模型
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Embedding, LSTM, Merge
from gensim import models   # 一个库，封装了一些机器学习模型
from gensim.models import KeyedVectors
from sklearn import metrics     # 一个库，封装了一些机器学习模型和预处理方法
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.preprocessing.sequence import pad_sequences
import itertools
import matplotlib.pyplot as plt     # 一个库，画图的
'''

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python import keras

import sys
sys.path.append('../')
import json
import os
import time
import matplotlib.pyplot as plt

from __utils4blockEmbed import *
from __lstm import *


# --configuration--
version = ['f', 'u']
arch = ['arm', 'mips', 'x86']
opt = ['0', '1', '2', '3']
# JSON_DIR = "../Instruction_Embedding/dataset/filtered_json_inst/"
JSON_DIR = "../dataset/3LACFGs_json_filtered/"
NODE_FEATURE_DIM = 11
batch_size = 5 # 10
max_seq_length = 350 # 10 blocks * 35条指令
# --configuration--

# filename in dataset
F_NAME = []
for v in version:
    for a in arch:
        for o in opt:
            jsonFile = 'openssl_1.0.1{}_{}_O{}_features.json'.format(v, a, o)
            if os.path.exists('../dataset/3LACFGs_json_filtered/'+jsonFile):
                F_NAME.append(jsonFile)
F_NAME = list(map(lambda x: JSON_DIR+x, F_NAME))
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'F_NAME')

# func name in dataset
FUNC_NAME_DICT = {}
FUNC_NAME_DICT = get_f_dict(F_NAME)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'FUNC_NAME_DICT')

# read graphs
Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)  # Gs是所有的图，classes是每个函数对应的图的index
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("{} graphs, {} functions".format(len(Gs), len(classes)))

# partition data
if os.path.isfile('./class_perm.npy'.format(NODE_FEATURE_DIM)):
    perm = np.load('./class_perm.npy'.format(NODE_FEATURE_DIM))
    print('perm exist')
else:
    perm = np.random.permutation(len(classes))
    np.save('./class_perm.npy'.format(NODE_FEATURE_DIM), perm)
    print('new perm')

Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test =\
            partition_data(Gs,classes,[0.8,0.1,0.1],perm)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print( "Train: {} graphs, {} functions".format(len(Gs_train), len(classes_train)))  # 49894 graphs
print( "Dev: {} graphs, {} functions".format(len(Gs_dev), len(classes_dev)))  # 6326 graphs
print( "Test: {} graphs, {} functions".format(len(Gs_test), len(classes_test)))  # 6318 graphs

# generate batch
epoch_data = generate_epoch_pair(Gs_train, classes_train, batch_size, FUNC_NAME_DICT) 
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'epoch_data', len(epoch_data))
# epoch data是一个列表，长度是|总数据／batch_size|，每个元素是个7元组，比如X1的长度是batch_size*2

'''
# siamese Model
    gnn = graphnn(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype, 
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )
    gnn.init(LOAD_PATH, LOG_PATH)
'''
# initial model
model = lstm(max_seq_length)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'finished init lstm')

perm = np.random.permutation(len(epoch_data))   #Random shuffle 随机选取训练集
for index in perm:
    cur_data = epoch_data[index]
    X1, X2, mask1, mask2, y, X1_insts, X2_insts = cur_data  # X1_insts的长度也是batch_size*2, X1_insts=[[[node],[node],[node],...[node], flag], [func], ...]
    
    # data prepro for block embedding
    # X1_insts_embed = X1_insts.copy()  # 还是保留基本块
    # X2_insts_embed = X2_insts.copy()
    X1_insts_embed, X2_insts_embed = [[] for _ in range(len(X1_insts))], [[] for _ in range(len(X2_insts))]  # 按照函数分割，函数内指令连在一起，指令条数不定
    for func_i in range(len(X1_insts)):
        # print(X1_insts[func_i][-1])  # flag
        for node_i in range(len(X1_insts[func_i])-1):
            X1_insts_embed[func_i] += inst2embed(X1_insts[func_i][node_i], X1_insts[func_i][-1])
            # X1_insts_embed[func_i][node_i] = inst2embed(X1_insts[func_i][node_i], X1_insts[func_i][-1])
    for func_i in range(len(X2_insts)):
        for node_i in range(len(X2_insts[func_i])-1):
            X2_insts_embed[func_i] += inst2embed(X2_insts[func_i][node_i], X2_insts[func_i][-1])
            # X2_insts_embed[func_i][node_i] = inst2embed(X2_insts[func_i][node_i], X2_insts[func_i][-1])

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X1_insts_embed, padding = 'post', truncating='post', maxlen=max_seq_length)
    print('padded_inputs shape: ', padded_inputs.shape)  # shape(10, 350, 100)
    y_train = np.array((1,1,1,1,1,1,1,1,1,1))
    print('# Fit model on training data')
    history = model.fit(padded_inputs, y_train, batch_size=5, epochs=3,)
    print(history.history)
    # cost = model.train_on_batch(padded_inputs)
    # print('cost on batch: ', cost)
    # prediction = model.predict(padded_inputs)
    # print('prediction: ', prediction)
    
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'batch')
    break
