import pickle
import json
import os

from raw_graphs import *

dirpath = './output/'  # input path
files = os.listdir(dirpath)

for fl in files:
    if 'openssl' in fl:  # input files filter
        jsonpath = './json/' + fl.split('.ida')[0] + '_features.json'  # output path
        filepath = os.path.join(dirpath, fl)
        print(filepath)
        with open(filepath, 'r') as f:
            cfgs = pickle.load(f)

        with open(jsonpath, 'w') as jf:
            for cfg in cfgs.raw_graph_list:
                dic = {}
                dic['src'] = filepath
                dic['fname'] = cfg.funcname
                dic['n_num'] = len(cfg.g.node)
                suc = []
                for n, nbrsdict in cfg.g.adjacency():
                   suc.append(list(nbrsdict.keys()))
                dic['succs'] = suc
                dic['features'] = []
                for i in xrange(len(cfg.g.node)):
                    fvec = []
                    fvec.append(len(cfg.g.node[i]['v'][0])) # 'consts'
                    fvec.append(len(cfg.g.node[i]['v'][1])) # 'strings'
                    fvec.append(cfg.g.node[i]['v'][2]) # 'offs'
                    fvec.append(cfg.g.node[i]['v'][3]) # 'numAs'
                    fvec.append(cfg.g.node[i]['v'][4]) # 'numCalls'
                    fvec.append(cfg.g.node[i]['v'][5]) # 'numIns'
                    fvec.append(cfg.g.node[i]['v'][6]) # 'numLIs'
                    fvec.append(cfg.g.node[i]['v'][7]) # 'numTIs'
                    toStDis = cfg.g.node[i]['v'][8]
                    if type(toStDis) == dict:
                        fvec.append(0) # 'toStDis'
                    else:
                        fvec.append(toStDis) # 'toStDis'
                    fvec.append(cfg.g.node[i]['v'][9]) # 'toEdDis'
                    fvec.append(cfg.g.node[i]['v'][10]) # 'between'
                    dic['features'].append(fvec)
                dic['func_feature'] = cfg.fun_features[:-2]

                data = json.dumps(dic)
                jf.write(data + '\n')
