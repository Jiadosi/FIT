import pickle
import json
import os

from raw_graphs import *

dirpath = './output/'
files = os.listdir(dirpath)

for fl in files:
    if '.ida' in fl:
        jsonpath = '' + fl.split('.')[0] + '_features.json'
        filepath = os.path.join(dirpath, fl)
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
                    fvec.append(cfg.g.node[i]['v'][8]) # 'toStDis'
                    fvec.append(cfg.g.node[i]['v'][9]) # 'toEdDis'
                    dic['features'].append(fvec)
                dic['func_feature'] = cfg.fun_features[:-2]

                data = json.dumps(dic)
                jf.write(data + '\n')
