# -*- coding: UTF-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from __graphnnSiamese import graphnn
import json
import pdb
import gc
from datetime import datetime
from gensim.models import word2vec, KeyedVectors

def get_f_name(DATA, SF, CM, OP, VS):
    F_NAME = []
    for sf in SF:
        for cm in CM:
            for op in OP:
                for vs in VS:
                    F_NAME.append(DATA+sf+cm+op+vs+".json")
    # print('debugging', F_NAME)
    return F_NAME


def get_f_dict(F_NAME):
    name_num = 0
    name_dict = {}
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if (g_info['fname'] not in name_dict):
                    name_dict[g_info['fname']] = name_num
                    name_num += 1
    # print('debugging', name_dict)
    return name_dict

class graph(object):
    def __init__(self, node_num = 0, label = None, name = None, func_feature = None):
        self.node_num = node_num
        self.label = label
        self._label = label
        self.name = name
        self.features = []
        self.succs = []
        self.preds = []
        self.insts = []  # dosi 1.7
        if (node_num > 0):
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])
                self.insts.append([]) # dosi 1.7

        # dosi 11.5
        self.func_feature = func_feature
                
    def add_node(self, feature = []):
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])
        
    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret

        
def read_graph(F_NAME, FUNC_NAME_DICT, FEATURE_DIM):
    graphs = []
    classes = []
    if FUNC_NAME_DICT != None:
        for f in range(len(FUNC_NAME_DICT)):
            classes.append([])

    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                label = FUNC_NAME_DICT[g_info['fname']]  # label是fname的index
                classes[label].append(len(graphs))  # classes[label]记录当前的graph大小，也就是记录下来该函数的graph在graphs里的index，因为有可能出现多次该函数，好骚
                # cur_graph = graph(g_info['n_num'], label, g_info['src'])
                cur_graph = graph(g_info['n_num'], label, g_info['src'], g_info['func_feature'])  # dosi 11.5

                for u in range(g_info['n_num']):
                    # cur_graph.features[u] = np.array(g_info['features'][u])
                    cur_graph.features[u] = np.array(g_info['features'][u][:11])  # dosi 1.7　前11是bb特征
                    cur_graph.insts[u] = g_info['features'][u][-1]  # dosi 1.7 最后1位是预处理后的指令集
                    for v in g_info['succs'][u]:
                        cur_graph.add_edge(u, v)
                graphs.append(cur_graph)

    return graphs, classes

def read_graph_in_one_file(f_name, FUNC_NAME_DICT):  # similar with read_graph, but return graphs in a single json file
    graphs = []
    with open(f_name) as inf:
        for line in inf:
            g_info = json.loads(line.strip())
            label = FUNC_NAME_DICT[g_info['fname']]
            cur_graph = graph(g_info['n_num'], label, g_info['src'], g_info['func_feature'])

            for u in range(g_info['n_num']):
                cur_graph.features[u] = np.array(g_info['features'][u][:11])
                cur_graph.insts[u] = g_info['features'][u][-1]
                for v in g_info['succs'][u]:
                    cur_graph.add_edge(u, v)
            graphs.append(cur_graph)
    return graphs

def partition_data(Gs, classes, partitions, perm):
    C = len(classes)  # 函数个数
    st = 0.0  # start开始的函数位置
    ret = []
    for part in partitions:
        cur_g = []
        cur_c = []
        ed = st + part * C  # part*C是函数个数，end也就是结束的函数位置
        for cls in range(int(st), int(ed)):
            prev_class = classes[perm[cls]]  # 随机取样出一个函数，获得它所有的图
            cur_c.append([])
            for i in range(len(prev_class)):  # 取出该函数的所有图，把图存到cur_g里
                cur_g.append(Gs[prev_class[i]])
                cur_g[-1].label = len(cur_c)-1
                cur_c[-1].append(len(cur_g)-1)

        ret.append(cur_g)  # cur_g是这一阶段的所有函数的所有图的一个列表
        ret.append(cur_c)  # cur_c是函数个数大小的一个列表，每个元素是一个列表，对应一个函数的图的位置
        st = ed

    return ret  # ret长度为6，一个train图的列表，一个train的label列表，一个dev图的列表，一个dev的label列表，一个test图的列表，一个test的label列表


def generate_epoch_pair(Gs, classes, M, output_id = False, load_id = None):  # M=5
    epoch_data = []
    id_data = []   # [ ([(G0,G1),(G0,G1), ...], [(G0,H0),(G0,H0), ...]), ... ]

    if load_id is None:
        st = 0
        while st < len(Gs):
            if output_id:
                X1, X2, m1, m2, y, pos_id, neg_id, X1_insts, X2_insts = get_pair(Gs, classes,
                        M, st=st, output_id=True)  # dosi 1.13
                id_data.append( (pos_id, neg_id) )
            else:
                X1, X2, m1, m2, y, X1_insts, X2_insts = get_pair(Gs, classes, M, st=st)  # dosi 1.13
            epoch_data.append( (X1,X2,m1,m2,y, X1_insts, X2_insts) )  # dosi 1.13
            st += M
    else:   # Load from previous id data
        id_data = load_id  # load_id里放的是一对对id
        for id_pair in id_data:  # id_pair是一个长度为2的列表，第一个是pos列表，第二个是neg列表，长度为M，也就是Batch_size
            X1, X2, m1, m2, y, X1_insts, X2_insts = get_pair(Gs, classes, M, load_id=id_pair)  # X1是pos的一对对图的特征矩阵，X2是neg，m1是X1的边信息，m2是X2的编信息，y是标记１和－１
            epoch_data.append( (X1, X2, m1, m2, y, X1_insts, X2_insts) )  # dosi 1.13

    if output_id:
        return epoch_data, id_data
    else:
        return epoch_data


def get_pair(Gs, classes, M, st = -1, output_id = False, load_id = None):
    if load_id is None:
        C = len(classes)  # Ｃ函数个数

        if (st + M > len(Gs)):  # 调整batch_size
            M = len(Gs) - st
        ed = st + M

        pos_ids = [] # [(G_0, G_1)]
        neg_ids = [] # [(G_0, H_0)]

        for g_id in range(st, ed):
            g0 = Gs[g_id]
            cls = g0.label
            tot_g = len(classes[cls])  # 和g0同属于一个函数的图一共有tot_g个
            if (len(classes[cls]) >= 2):
                g1_id = classes[cls][np.random.randint(tot_g)]  # 在和g0同属于一个函数的图中随机选取一个图作为g1
                while g_id == g1_id:  # g1不能是g0
                    g1_id = classes[cls][np.random.randint(tot_g)]
                pos_ids.append( (g_id, g1_id) )  # g0g1构成一组pos

            cls2 = np.random.randint(C)  # 随机选一个函数cls2
            while (len(classes[cls2]) == 0) or (cls2 == cls):  # cls2不能是cls
                cls2 = np.random.randint(C)

            tot_g2 = len(classes[cls2])
            h_id = classes[cls2][np.random.randint(tot_g2)]  # 随机选cls2的一个图h
            neg_ids.append( (g_id, h_id) )  # g0h构成一组neg
    else:
        pos_ids = load_id[0]
        neg_ids = load_id[1]
        
    M_pos = len(pos_ids)  # batch_size
    M_neg = len(neg_ids)  # batch_size
    M = M_pos + M_neg  # batch_size*2

    maxN1 = 0  # 一对里的第一个图的最大节点个数
    maxN2 = 0  # 一对里的第二个图的最大节点个数
    for pair in pos_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)
    for pair in neg_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)

    feature_dim = len(Gs[0].features[0])  # 特征个数
    # X1_input = np.zeros((M, maxN1, feature_dim))  # M个[maxN1×feature_dim的全零矩阵]，这个用来存每个bb的特征
    # X2_input = np.zeros((M, maxN2, feature_dim))

    # dosi: modify X1, X2
    X1_input = np.zeros((M, maxN1+1, feature_dim))  # M个[(maxN1+1)×feature_dim+4的全零矩阵]，这个用来存每个bb的特征,用最后一行来存11个函数层特征
    X2_input = np.zeros((M, maxN2+1, feature_dim))
    # dosi 1.7 add insts matrix corresponding to each function in X_input
    # X1_input_insts = np.empty((M, maxN1, 35), dtype=np.string_)  # M个[(maxN1x35)]的字符串矩阵，用来存每个函数的所有基本快的所有指令 
    # X2_input_insts = np.empty((M, maxN2, 35), dtype=np.string_)
    X1_input_inst_list, X2_input_inst_list = [[[] for _ in range(maxN1)] for _ in range(M)], [[[] for _ in range(maxN2)] for _ in range(M)]  # dosi 4.5

    node1_mask = np.zeros((M, maxN1, maxN1))  # M个maxN1×maxN1的全零矩阵，这个用来存bb之间的边，存在边就置1
    node2_mask = np.zeros((M, maxN2, maxN2))
    y_input = np.zeros((M))  # 长度为M的全零array[0,0,...,0]，将来会变成1和-1的标记，1表示相似，-1表示不相似吧
    
    for i in range(M_pos):
        y_input[i] = 1
        g1 = Gs[pos_ids[i][0]]  # 取出pos图1
        g2 = Gs[pos_ids[i][1]]  # 取出pos图2
        # dosi: fill the last row of each graph @fit
        X1_input[i, -1, :] = np.array(g1.func_feature)  # dosi 11.5
        # dosi: fill the last row of each graph @fit
        X2_input[i, -1, :] = np.array(g2.func_feature)  # dosi 11.5, 1.7fix bug,X1_input->X2_input

        for u in range(g1.node_num):
            X1_input[i, u, :] = np.array( g1.features[u] )  # 把g1的节点特征存到X1_input的前半部分
            for v in g1.succs[u]:
                node1_mask[i, u, v] = 1
            # dosi 1.7 fill in insts matrix
            X1_input_inst_list[i][u] = g1.insts[u]  # dosi 404
        X1_input_inst_list[i].append(g1.name)

        for u in range(g2.node_num):  # 把g2的节点特征存到X2_input的前半部分
            X2_input[i, u, :] = np.array( g2.features[u] )

            for v in g2.succs[u]:
                node2_mask[i, u, v] = 1
            # dosi 1.7 fill in insts matrix
            X2_input_inst_list[i][u] = g2.insts[u]  # dosi 404
        X2_input_inst_list[i].append(g2.name)
        
    for i in range(M_pos, M_pos + M_neg):
        y_input[i] = -1
        g1 = Gs[neg_ids[i-M_pos][0]]  # 取出neg图1
        g2 = Gs[neg_ids[i-M_pos][1]]  # 取出neg图2
        # dosi: fill the last row of each graph
        X1_input[i, -1, :] = np.array(g1.func_feature)  # dosi 11.5
        # dosi: fill the last row of each graph
        X2_input[i, -1, :] = np.array(g2.func_feature)  # dosi 11.5, 1.7fix bug,X1_input->X2_input
        for u in range(g1.node_num):
            X1_input[i, u, :] = np.array( g1.features[u] )  # 把g1的节点特征存到X1_input的后半部分
            for v in g1.succs[u]:
                node1_mask[i, u, v] = 1
            # dosi 1.7 fill in insts matrix
            X1_input_inst_list[i][u] = g1.insts[u]  # dosi 404
        X1_input_inst_list[i].append(g1.name)
        for u in range(g2.node_num):
            X2_input[i, u, :] = np.array( g2.features[u] )
            for v in g2.succs[u]:
                node2_mask[i, u, v] = 1
            # dosi 1.7 fill in insts matrix
            X2_input_inst_list[i][u] = g2.insts[u]  # dosi 404
        X2_input_inst_list[i].append(g2.name)
    # dosi 1.13 return
    if output_id:
        return X1_input,X2_input,node1_mask,node2_mask,y_input,pos_ids,neg_ids, X1_input_inst_list, X2_input_inst_list
    else:
        return X1_input,X2_input,node1_mask,node2_mask,y_input, X1_input_inst_list, X2_input_inst_list


def get_X_and_mask(Gs, M, st):
    if st + M > len(Gs):
        M = len(Gs) - st
    ed = st + M

    maxN = 0
    for i in range(st, ed):
        maxN = max(maxN, Gs[i].node_num)

    feature_dim = len(Gs[0].features[0])
    X = np.zeros((M, maxN + 1, feature_dim))
    X_inst_list = [[[] for _ in range(maxN)] for _ in range(M)]
    mask = np.zeros((M, maxN, maxN))

    for i in range(M):
        G = Gs[i + st]
        X[i, -1, :] = np.array(G.func_feature)
        for u in range(G.node_num):
            X[i, u, :] = np.array(G.features[u])
            for v in G.succs[u]:
                mask[i, u, v] = 1
            X_inst_list[i][u] = G.insts[u]
        X_inst_list[i].append(G.name)

    return X, mask, X_inst_list


def inst2embed(inst_list, arch, w2v):  # fetch inst embedding from w2v, try masking in the future
    if len(inst_list) > 35:  # cut
        inst_list = inst_list[:35]
    if 'arm' in arch:
        model = w2v['arm']
    elif 'mips' in arch:
        model = w2v['mips']
    elif 'x86' in arch:
        model = w2v['x86']
    else:
        print('\033[1;36mSomething wrong in fetching arch {}\033[0m'.format(arch))
    ret = []
    notfound = [0 for _ in range(100)]
    # notfound = np.zeros(100, dtype=np.float32)
    for inst in inst_list:
        if inst not in model.wv:
            ret.append(notfound)
        else:
            ret.append(model[inst].tolist())
    if len(ret) < 35:  # pad
        for _ in range(35-len(ret)):
            ret.append(notfound)
    return ret


def loading(modelPath):
    model = word2vec.Word2Vec.load(modelPath)
    return model


def train_epoch(model, graphs, classes, batch_size, w2v_model, load_data=None):
    if load_data is None:
        epoch_data = generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    perm = np.random.permutation(len(epoch_data))   #Random shuffle 随机选取训练集
    # print(len(epoch_data))
    cum_loss = 0.0
    for index in perm:
        cur_data = epoch_data[index]
        # X1, X2, mask1, mask2, y = cur_data
        X1, X2, mask1, mask2, y, X1_insts, X2_insts = cur_data  # dosi 1.13
        # dosi, data prepro for block embed
        # print('dosi: going for inst2embed. zzzzzzzzzz')
        X1_insts_embed = [[[] for _ in range(len(X1_insts[0])-1)] for _ in range(len(X1_insts))]
        X2_insts_embed = [[[] for _ in range(len(X2_insts[0])-1)] for _ in range(len(X2_insts))]
        for func_i in range(len(X1_insts)):
            for node_i in range(len(X1_insts[func_i])-1):
                X1_insts_embed[func_i][node_i] = inst2embed(X1_insts[func_i][node_i], X1_insts[func_i][-1], w2v_model)
        for func_i in range(len(X2_insts)):
            for node_i in range(len(X2_insts[func_i])-1):
                X2_insts_embed[func_i][node_i] = inst2embed(X2_insts[func_i][node_i], X2_insts[func_i][-1], w2v_model)
        X1_insts_embed = np.array(X1_insts_embed)
        X2_insts_embed = np.array(X2_insts_embed)

        # print('fetch inst embedding per batch @ {0}'.format(datetime.now()))
        loss = model.train(X1, X2, mask1, mask2, y, X1_insts_embed, X2_insts_embed)
        # print('training per batch @ {0}'.format(datetime.now()))
        # print('dosi: del or killed')
        # del X1_insts_embed
        # del X2_insts_embed
        # gc.collect()
        
        # loss = model.train(X1, X2, mask1, mask2, y, X1_insts, X2_insts)  # dosi 1.13
        cum_loss += loss

    return cum_loss / len(perm)


def get_auc_epoch(model, graphs, classes, batch_size, w2v_model, load_data=None):
    tot_diff = []
    tot_truth = []

    if load_data is None:
        epoch_data= generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    for cur_data in epoch_data:
        X1, X2, m1, m2,y, X1_insts, X2_insts  = cur_data  # dosi 1.13
        
        # dosi, data prepro for block embed
        X1_insts_embed = [[[] for _ in range(len(X1_insts[0])-1)] for _ in range(len(X1_insts))]
        X2_insts_embed = [[[] for _ in range(len(X2_insts[0])-1)] for _ in range(len(X2_insts))]
        for func_i in range(len(X1_insts)):
            for node_i in range(len(X1_insts[func_i])-1):
                X1_insts_embed[func_i][node_i] = inst2embed(X1_insts[func_i][node_i], X1_insts[func_i][-1], w2v_model)
        for func_i in range(len(X2_insts)):
            for node_i in range(len(X2_insts[func_i])-1):
                X2_insts_embed[func_i][node_i] = inst2embed(X2_insts[func_i][node_i], X2_insts[func_i][-1], w2v_model)
        X1_insts_embed = np.array(X1_insts_embed)
        X2_insts_embed = np.array(X2_insts_embed)
        # print('fetch inst embedding per batch @ {0}'.format(datetime.now()))
        diff = model.calc_diff(X1, X2, m1, m2, X1_insts_embed, X2_insts_embed)
        # print('cal_diff per batch @ {0}'.format(datetime.now()))
        # print diff
        # print( 'diff', (1-diff)/2)
        tot_diff += list(diff)
        tot_truth += list(y > 0)

    diff = np.array(tot_diff)
    truth = np.array(tot_truth)

    fpr, tpr, thres = roc_curve(truth, (1-diff)/2)
    # print(thres)
    model_auc = auc(fpr, tpr)

    return model_auc, fpr, tpr, thres


def get_embed_epoch(model, Gs, M, w2v_model):
    # Gs_embedded = np.empty((len(Gs), 132))  # 128 + 4
    Gs_embedded = np.empty((len(Gs), 128))
    st = 0
    while st < len(Gs):
        X, mask, X_insts = get_X_and_mask(Gs, M, st=st)
        X_insts_embed = [[[] for _ in range(len(X_insts[0])-1)] for _ in range(len(X_insts))]
        for func_i in range(len(X_insts)):
            for node_i in range(len(X_insts[func_i])-1):
                X_insts_embed[func_i][node_i] = inst2embed(X_insts[func_i][node_i], X_insts[func_i][-1], w2v_model)
        embed = model.get_embed(X, mask, X_insts_embed)
        ed = st + len(embed)
        Gs_embedded[st : ed] = embed
        st += M
    return Gs_embedded


def minus_cosine(vec_0, vec_1):
    cosine = np.dot(vec_0, vec_1) / np.sqrt(np.dot(vec_0, vec_0) * np.dot(vec_1, vec_1) + 1e-10)
    return -cosine


class info(object):
    def __init__(self, score, G):
        self.score = score
        self.graph = G


def get_infos(Gs, Gs_embedded, G_embedded):
    infos = []

    for i in range(len(Gs)):
        score = minus_cosine(Gs_embedded[i], G_embedded)
        # Gs and Gs_embedded are in the same order
        infos.append(info(score, Gs[i]))

    infos.sort(key=lambda x:x.score)

    return infos


def print_arch_info(name, Gs):
    x86 = arm = mips = 0
    strange = []
    for G in Gs:
        if "x86" in G.name:
            x86 += 1
        elif "arm" in G.name:
            arm += 1
        elif "mips" in G.name:
            mips += 1
        elif not G.name in strange:
            strange.append(G.name)
    print("\033[1;36m{}: {} x86, {} arm, {} mips\033[0m".format(name, x86, arm, mips))
    print(strange)


def get_key (dict, value):
    for k, v in dict.items():
        if v == value:
            return k
