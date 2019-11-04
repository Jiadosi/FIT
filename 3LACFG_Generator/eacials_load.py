import pickle
from raw_graphs import *

with open('../output/local.ida', 'r') as f:
    cfgs = pickle.load(f)

for cfg in cfgs.raw_graph_list:
    print cfg.funcname
    print cfg.fun_features
    for i in xrange(len(cfg.g.node)):
        print cfg.g.node[i]