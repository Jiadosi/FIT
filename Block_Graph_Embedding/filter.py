# -*- coding: UTF-8 -*-
import numpy as np
from __utils import *
from __graphnnSiamese import graphnn
import json
import argparse
import os
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--fea_dim', type=int, default=11,
        help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=128,
        help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2,
        help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=128,
        help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=10,
        help='iteration times')
parser.add_argument('--lr', type=float, default=0.0001,
        help='learning rate')
parser.add_argument('--epoch', type=int, default=50,
        help='epoch number')
parser.add_argument('--batch_size', type=int, default=10,
        help='batch size')
parser.add_argument('--load_path', type=str,
        default='./saved_model/405/graphnn-model_best',
        help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')
parser.add_argument('--w2v_path', type=str, default='./myModel',
        help='path for w2v model')
parser.add_argument('--lstm_hidden', type=int, default=128,
        help='hidden size in lstm')
parser.add_argument('--top_similar', type=int, default=50,
        help='filter top n similar func')


if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("\033[1;36m=================================\033[0m")
    print("\033[1;36m", args, "\033[0m")
    print("\033[1;36m=================================\033[0m")

    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    LOG_PATH = args.log_path
    W2VMODEL = args.w2v_path
    LSTM_HIDDEN = args.lstm_hidden
    TOPN = args.top_similar

    # w2v model dict
    w2v = {}
    w2v['arm'] = loading(os.path.join(W2VMODEL, 'arm'))
    w2v['mips'] = loading(os.path.join(W2VMODEL, 'mips'))
    w2v['x86'] = loading(os.path.join(W2VMODEL, 'x86'))
    
    # Model
    gnn = graphnn(
            n_hidden = LSTM_HIDDEN,
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype, 
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )
    gnn.init(LOAD_PATH, LOG_PATH)

    JSON_DIR = "./data/3lacfgSSL_{}/filtered_json_inst/jsonWithInst/".format(NODE_FEATURE_DIM)
    F_NAME = list(map(lambda x: JSON_DIR + x, os.listdir(JSON_DIR)))
    FUNC_NAME_DICT = {}
    FUNC_NAME_DICT = get_f_dict(F_NAME)

    for f_name in F_NAME:
        Gs = read_graph_in_one_file(f_name, FUNC_NAME_DICT)
        embeddings = get_embed_epoch(gnn, Gs, BATCH_SIZE, w2v)
        infos = get_infos(Gs, embeddings, embeddings[-1])  # last graph as the vul graph
        infos = infos[:TOPN]  # take top 50
        for info in infos:
            print('\033[1;36m {0} -- {1} score: {2}\033[0m'.format(get_key(FUNC_NAME_DICT, info.graph.label), get_key(FUNC_NAME_DICT, Gs[-1].label), info.score))
        print("\033[1;36m=================================\033[0m")
        break

