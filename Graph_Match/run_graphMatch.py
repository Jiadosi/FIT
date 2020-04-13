import os
import json
import argparse

from cfg import *
from basicBlock import *
from instruction import *
from graph_match import *


def json2cfg(js):
    cfgraph = cfg(js['fname'])
    idx = 0
    for bb in js['features']:
        inst_list = []
        cur_bb = basicBlock(idx)
        for inst in bb[11]:
            inst_list.append(instruction(inst.split()[0]))
        cur_bb.set_instructions(inst_list)
        cur_bb.set_no_of_arithmetic_inst(bb[3])
        cur_bb.set_no_of_transfer_inst(bb[7])
        cur_bb.set_no_of_function_calls(bb[4])
        cur_bb.set_betweenness_new(bb[10])
        cur_bb.set_no_of_incoming_calls(bb[2])
        cur_bb.set_entry_bb(bb[8])
        cur_bb.set_bb_exit(bb[9])
        
        cfgraph.set_vertices(cur_bb)
        idx += 1
    
    cfgraph.set_no_of_func_calls(js["func_feature"][0])
    cfgraph.set_no_of_logic_inst(js["func_feature"][1])
    cfgraph.set_no_of_trans_inst(js["func_feature"][2])
    cfgraph.set_no_of_inst(js["func_feature"][7])
    cfgraph.set_no_of_incoming_calls(js["func_feature"][6])
    cfgraph.set_no_of_local_var(js["func_feature"][9])
    cfgraph.set_no_of_global_var(js["func_feature"][3])
    
    # set edges
    idx = 0
    for e in js["succs"]:
        for dest in e:
            cfgraph.set_edges(cfgraph.vertices[idx], cfgraph.vertices[dest])
        if len(e) == 0:
            cfgraph.set_exit(cfgraph.vertices[idx])
        idx += 1
    # set in_degree and out_degree
    cfgraph.prepare_new()
    # set entry bb
    for n in cfgraph.vertices:
        if len(n.in_degree) == 0:
            cfgraph.set_entry(n)

    return cfgraph


parser = argparse.ArgumentParser()
parser.add_argument('--sus_dir', type=str, default='./suspicious/', help='path for suspicious files')
parser.add_argument('--json_dir', type=str, default='./testwhole/', help='path for json files')
parser.add_argument('--threashold', type=float, default=1.489, help='dissimilar score threashold') 


if __name__ == "__main__":

    args = parser.parse_args()
    SUS_PATH = args.sus_dir
    JSON_PATH = args.json_dir
    THREASHOLD = args.threashold

    for bin_name in os.listdir(SUS_PATH):
        print(bin_name)
        with open(os.path.join(SUS_PATH, bin_name), 'r') as f:
           s_funcs = f.read().strip().split('\n')

        a_cfgs = []

        json_name = bin_name + '_features.json'
        with open(os.path.join(JSON_PATH, json_name), 'r') as f:
            for line in f.readlines():
                if len(s_funcs) != 0:
                    cur_func = json.loads(line)
                    if cur_func['fname'] in s_funcs:
                        a_cfgs.append(json2cfg(cur_func))
                        s_funcs.remove(cur_func['fname'])

        t_cfg = json2cfg(json.loads(line))
        for cfg in a_cfgs:
            score = graph_match(t_cfg, cfg)
            if score <= THREASHOLD:
                print('\033[1;36m{0} -- {1} score: {2}\033[0m'.format(cfg.function_name, t_cfg.function_name, score))
