import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from __graphnnSiamese import graphnn
from __utils import *
import os
import argparse
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


parser = argparse.ArgumentParser()
parser.add_argument('--fea_dim', type=int, default=11,
        help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=128,
        help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2,  # fit 8?
        help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=128,
        help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
        help='iteration times')
parser.add_argument('--lr', type=float, default=0.0001, # 0.0001, 8e-5
        help='learning rate')
parser.add_argument('--epoch', type=int, default=100,
        help='epoch number')
parser.add_argument('--batch_size', type=int, default=10, # 5
        help='batch size')
parser.add_argument('--load_path', type=str, default=None,
        help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--save_path', type=str,
        default='./saved_model/408/graphnn-model', help='path for model saving')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')
parser.add_argument('--w2v_path', type=str, default='./myModel',
        help='path for w2v model')
parser.add_argument('--lstm_hidden', type=int, default=128,
        help='hidden size in lstm')


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
    SAVE_PATH = args.save_path
    LOG_PATH = args.log_path
    W2VMODEL = args.w2v_path
    LSTM_HIDDEN = args.lstm_hidden

    # dosi 11.8
    fpr_path = './npy/fit_embed{}_depth{}_iter{}_404_fpr.npy'.format(EMBED_DIM, EMBED_DEPTH, ITERATION_LEVEL)
    tpr_path = './npy/fit_embed{}_depth{}_iter{}_404_tpr.npy'.format(EMBED_DIM, EMBED_DEPTH, ITERATION_LEVEL)

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5

    # JSON_DIR = "./data/3lacfgSSL_{}/large_small_filtered_json/small/".format(NODE_FEATURE_DIM)
    JSON_DIR = "./data/3lacfgSSL_{}/filtered_json_inst/jsonWithInst/".format(NODE_FEATURE_DIM)
    F_NAME = list(map(lambda x: JSON_DIR + x, os.listdir(JSON_DIR)))

    FUNC_NAME_DICT = {}
    FUNC_NAME_DICT = get_f_dict(F_NAME)


    print("\033[1;36m=================================\033[0m")
    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)
    print( "\033[1;36m{} graphs, {} functions\033[0m".format(len(Gs), len(classes)))

    if os.path.isfile('data/class_perm_{}_404.npy'.format(NODE_FEATURE_DIM)):
        perm = np.load('./class_perm_{}_404.npy'.format(NODE_FEATURE_DIM))
    else:
        perm = np.random.permutation(len(classes))
        np.save('./class_perm_{}_404.npy'.format(NODE_FEATURE_DIM), perm)
    # if len(perm) < len(classes):
    if len(perm) != len(classes):  # dosi
        perm = np.random.permutation(len(classes))
        np.save('./class_perm_{}_404.npy'.format(NODE_FEATURE_DIM), perm)

    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test =\
            partition_data(Gs,classes,[0.8,0.1,0.1],perm)

    print( "\033[1;36mTrain: {} graphs, {} functions\033[0m".format(
            len(Gs_train), len(classes_train)))
    print( "\033[1;36mDev: {} graphs, {} functions\033[0m".format(
            len(Gs_dev), len(classes_dev)))
    print( "\033[1;36mTest: {} graphs, {} functions\033[0m".format(
            len(Gs_test), len(classes_test)))
    print("\033[1;36m=================================\033[0m")
    # print_arch_info('train', Gs_train)  # dosi 11.6
    # print_arch_info('dev', Gs_dev)
    # print_arch_info('test', Gs_test)

    # Fix the pairs for validation
    if os.path.isfile('data/valid_{}_404.json'.format(NODE_FEATURE_DIM)):
        with open('data/valid_{}_404.json'.format(NODE_FEATURE_DIM)) as inf:
            valid_ids = json.load(inf)
        valid_epoch = generate_epoch_pair(
                Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    else:
        valid_epoch, valid_ids = generate_epoch_pair(
            Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
        with open('data/valid_{}_404.json'.format(NODE_FEATURE_DIM), 'w') as outf:
            json.dump(valid_ids, outf)

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

    # Train
    auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
            BATCH_SIZE, w2v, load_data=valid_epoch)
    gnn.say("\033[1;36mInitial training auc = {0} @ {1}\033[0m".format(auc, datetime.now()))
    auc0, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
            BATCH_SIZE, w2v, load_data=valid_epoch)
    gnn.say("\033[1;36mInitial validation auc = {0} @ {1}\033[0m".format(auc0, datetime.now()))

    best_auc = 0
    for i in range(1, MAX_EPOCH+1):
        l = train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE, w2v)
        gnn.say("\033[1;36mEPOCH {3}/{0}, loss = {1} @ {2}\033[0m".format(
            MAX_EPOCH, l, datetime.now(), i))

        if (i % TEST_FREQ == 0):
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
                    BATCH_SIZE, w2v, load_data=None)
            gnn.say("\033[1;36mTesting model: training auc = {0} @ {1}\033[0m".format(
                auc, datetime.now()))
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev,
                    BATCH_SIZE, w2v, load_data=valid_epoch)
            gnn.say("\033[1;36mTesting model: validation auc = {0} @ {1}\033[0m".format(
                auc, datetime.now()))

            if auc > best_auc:
                path = gnn.save(SAVE_PATH+'_best')
                best_auc = auc
                gnn.say("\033[1;36mModel saved in {}\033[0m".format(path))
                np.save(fpr_path, fpr)
                np.save(tpr_path, tpr)

        if (i % SAVE_FREQ == 0):
            path = gnn.save(SAVE_PATH, i)
            gnn.say("\033[1;36mModel saved in {}\033[0m".format(path))
