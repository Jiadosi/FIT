#!/usr/bin/env python
# coding=utf-8
import pickle

from HungarianMurty import *

from basicBlock import *
from cfg import *
from instruction import *

#match bb
def isMatch(bb1, bb2):
    #print(bb1.entry_bb,bb2.entry_bb,'-',bb1.bb_exit,bb2.bb_exit,'-',len(bb1.in_degree), len(bb2.in_degree),'-',len(bb1.out_degree),len(bb2.out_degree))
    if bb1.entry_bb == bb2.entry_bb and bb1.bb_exit == bb2.bb_exit and len(bb1.in_degree) == len(bb2.in_degree) and len(bb1.out_degree) == len(bb2.out_degree) and bb1.betweenness == bb2.betweenness and bb1.no_of_inst == bb2.no_of_inst and bb1.no_of_arithmetic_inst == bb2.no_of_arithmetic_inst and bb1.no_of_transfer_inst == bb2.no_of_transfer_inst:
        return True
    else:
        return False

#input the start bb of traverse and the target node, return the matched bb or False if can't find.
def levelOrder_traverse_match(start_bb_queue, target_bbs):
    global dic
    global dic2
    order = []   # record the order of discovering the matched target_bbs
    #match_res = []   # result of matching. list of tuples
    rest_queue = []  # After the first matching node, to record the remaining matched path.
    idx_queue = []  # record the index of matched bb in rest_queue
    seen = []
    paths = list(map(lambda x:[x], start_bb_queue))
    #allpath = []
    ret_dic = {}
    flag = False  # whether matched the first node , if we find the first node that matches the target, then the code turns to find the paths which also lead to the target bb, then set the bb on the path as unmatchable.
    #target = None
    first_matched_bb = None
    
    for t_bb in target_bbs:
        ret_dic[t_bb] = None

    while paths:
        path = paths.pop(0)
        node = path[-1]
        if node.matchable:
            if not flag:  #when the first node is not matched
                for t_bb in target_bbs:
                    if isMatch(t_bb, node):  # if there is a bb can be matched.
                        first_matched_bb = t_bb
                        target_bbs.remove(t_bb) 
                        flag = True
                        ret_dic[t_bb] = node
                        for n in path:
                            n.set_matchable()
                        for t_bb in target_bbs:
                            rest_queue.append([])
                            idx_queue.append([])
                        break
            else:
                for i in range(len(target_bbs)):  
                    if isMatch(target_bbs[i], node):
                        rest_queue[i].append(path)
                        idx_queue[i].append(len(path) - 1)
                        order.append(target_bbs[i])
        
        if node.out_degree:
            for i in range(len(rest_queue)):
                tmp_queue = []
                tmp_idx_queue = [] 
                for j in range(len(rest_queue[i])):
                    for out in node.out_degree:
                        if out in seen:
                            tmp_queue.append(list(rest_queue[i][j]))
                            tmp_idx_queue.append(idx_queue[i][j])
                            continue
                        else:
                            if rest_queue[i][j][-1] == node:
                                path_rest_queue = list(rest_queue[i][j])
                                path_rest_queue.append(out)
                                tmp_queue.append(path_rest_queue)
                                tmp_idx_queue.append(idx_queue[i][j])
                            else:
                                tmp_queue.append(list(rest_queue[i][j]))
                                tmp_idx_queue.append(idx_queue[i][j])
                                break
                rest_queue[i] = tmp_queue
                idx_queue[i] = tmp_idx_queue
        seen.append(node)
        for out in node.out_degree:  # handle the children nodes
            if out in seen:
                continue
            else:
                new_path = list(path)
                new_path.append(out)
                paths.append(new_path)
                seen.append(out)

    if not first_matched_bb:
        return ret_dic
    cur_list = [first_matched_bb.matched_bb]
    for t_bb in order:
        index = target_bbs.index(t_bb)
        is_t_bb_matched = False
        for path_num in range(len(rest_queue[index])):  # traverse the ramaining unmatched queue
            for m_bb in cur_list:   # traverse the matched bb 
                if m_bb in rest_queue[index][path_num]:
                    max_idx = 0
                    for tmp_bb in cur_list:
                        if tmp_bb in rest_queue[index][path_num]:
                            max_idx = max(max_idx, rest_queue[index][path_num].index(tmp_bb))
                    for n in range(len(rest_queue[index][path_num])):
                        rest_queue[index][path_num][n].set_matchable()
                        #if n == rest_queue[index][path_num][idx_queue[index][path_num]]:
                        if n == max_idx:
                            break
                    break
                else:
                    if not is_t_bb_matched:
                        is_t_bb_matched = True
                        for n in rest_queue[index][path_num]:
                            n.set_matchable()
                            if n == rest_queue[index][path_num][idx_queue[index][path_num]]:
                                break
                        cur_list.append(rest_queue[index][path_num][idx_queue[index][path_num]])
                        ret_dic[t_bb] = rest_queue[index][path_num][idx_queue[index][path_num]]
    return ret_dic
                


#    while start_bb_queue:
#        cur_bb = start_bb_queue.pop[0]
#        if isMatch(cur_bb, target):
#            return cur_bb
#        start_bb_queue += cur_bb.out_degree
#    return False

        
# input the opcode_list of two bb and output the ngram matching matrix. ngram matching used LCS Algorithm
def get_ngram_matrix(insA, insB):
    trigramA = []
    trigramB = []
    res = []
    for i in range(2, len(insA)):
        trigramA.append((insA[i-2], insA[i-1], insA[i]))

    for i in range(2, len(insB)):
        trigramB.append((insB[i-2], insB[i-1], insB[i]))
    
    for triA in trigramA:
        tmp = []
        for triB in trigramB:
            tmp_mat = [[0 for i in range(0, 4)] for j in range(0, 4)] #LCS
            for i in range(1, 4):
                for j in range(1, 4):
                    if triA[i-1] == triB[j-1]:
                        tmp_mat[i][j] = tmp_mat[i-1][j-1] + 1
                    else:
                        tmp_mat[i][j] = max(tmp_mat[i-1][j], tmp_mat[i][j-1])
            tmp.append(3 - tmp_mat[-1][-1])
        res.append(tmp)
    #print(res) 
    #add the upper right part
#    for i in range(0, len(trigramA)):
#        for j in range(len(trigramB), len(trigramA)+len(trigramB)):
#            #if j-i == len(trigramB):
#            #    res[i].append(3)
#            #else:
#            res[i].append(3-0)
#    #add the bottom-left part
#    for i in range(len(trigramA), len(trigramA) + len(trigramB)):
#        res.append([])
#        for j in range(0, len(trigramB)):
#            #if i-j == len(trigramA):
#            #    res[i].append(3)
#            #else:
#            res[i].append(3-0)
#
#    #add the bottom-right part
#    for i in range(len(trigramA), len(trigramA) + len(trigramB)):
#        for j in range(len(trigramB), len(trigramA) + len(trigramB)):
#            res[i].append(3-3)
    return res

def get_unigram_matrix(insA, insB):
    res = []
    for ia in insA:
        tmp = []
        for ib in insB:
            if ia == ib:
                tmp.append(0)
            else:
                tmp.append(1)
        res.append(tmp)
    return res

#absolutely match
def init_match2(cfg_vul, cfg_kernel): #updated init_match ^.^
    global dic
    matched_num = 0
    vul_startbb = cfg_vul.entry
    ker_startbb = cfg_kernel.entry
    seen = [vul_startbb]
    ker_bb_queue = [ker_startbb]
    vul_bb_queue = [(None, vul_startbb)]  # using a tuple to record bbs and their father
    while vul_bb_queue:
        father_thisround = vul_bb_queue[0][0]
        cur_vul_bbs_thisround = []
        while vul_bb_queue and vul_bb_queue[0][0] == father_thisround:
            cur_vul_bbs_thisround.append(vul_bb_queue.pop(0))

        #cur_vul_tuple = vul_bb_queue.pop(0)
        #cur_vul_bb = cur_vul_tuple[1]
        if father_thisround and father_thisround.matched_bb: # if cur_vul_bb has father and its father has been matched, then ker_bb_queue can be the children of its father
            ker_bb_queue = father_thisround.matched_bb.out_degree
        if not father_thisround:  # if cur_vul_bb has no father, then the ker_bb_queue should be the ker_startbb
            ker_bb_queue = [ker_startbb]
        match_res = levelOrder_traverse_match(ker_bb_queue, list(map(lambda x:x[1], cur_vul_bbs_thisround)))
        if match_res:
            for key, value in match_res.items():
                if value:
                    matched_num += 1
                    key.set_matched_bb(value)
                    value.set_matched_bb(key)
                    for n in key.out_degree:
                        if n not in seen:
                            vul_bb_queue.append((key, n))
                            seen.append(n)
                else:
                    for n in key.out_degree:
                        if n not in seen:
                            vul_bb_queue.append((father_thisround, n))
                            seen.append(n)
    return matched_num

def init_match(cfg_vul, cfg_kernel):
    matched_num = 0
    vul_startbb = cfg_vul.entry
    ker_startbb = cfg_kernel.entry
    seen = [vul_startbb]
    ker_bb_queue = [ker_startbb]
    vul_bb_queue = [(None, vul_startbb)]  # using a tuple to record bbs and their father
    while vul_bb_queue:
        cur_vul_tuple = vul_bb_queue.pop(0)
        cur_vul_bb = cur_vul_tuple[1]
        if cur_vul_tuple[0] and cur_vul_tuple[0].matched_bb: # if cur_vul_bb has father and its father has been matched, then ker_bb_queue can be the children of its father
            ker_bb_queue = cur_vul_tuple[0].matched_bb.out_degree
        if not cur_vul_tuple[0]:  # if cur_vul_bb has no father, then the ker_bb_queue should be the ker_startbb
            ker_bb_queue = [ker_startbb]
        match_res = levelOrder_traverse_match(ker_bb_queue, cur_vul_bb)
        if match_res:
            matched_num += 1
            match_res.set_matched_bb(cur_vul_bb) #set this field to the bb matched
            cur_vul_bb.set_matched_bb(match_res)
            for n in cur_vul_bb.out_degree:
                if n not in seen:
                    vul_bb_queue.append((cur_vul_bb, n))
                    seen.append(n)
            #vul_bb_queue += list(map(lambda x:(cur_vul_bb ,x) ,cur_vul_bb.out_degree))
        else:
            for n in cur_vul_bb.out_degree:
                if n not in seen:
                    vul_bb_queue.append((cur_vul_tuple[0], n))
                    seen.append(n)
            #vul_bb_queue += list(map(lambda x:(cur_vul_tuple[0], x), cur_vul_bb.out_degree))
    return matched_num

# if number of instructions in a bb less than 3, then use unigram.
def fuzzy_match3(cfg_vul, cfg_kernel):
    # print(cfg_vul.vertices[0].instructions)
    list_vul = []
    list_ker = []
    mat = []   # the matrix for Hungarian Algorithm
    for vul_node in cfg_vul.vertices:
        if vul_node.matched_bb:
            continue
        else:
            list_vul.append(vul_node)
            tmp = []
            for ker_node in cfg_kernel.vertices:
                if ker_node.matched_bb:
                    continue
                else:
                    point_bb = 3
                    list_ker.append(ker_node)
                    insA = [i.op.lower() for i in vul_node.instructions]   # python exercise: two methods to generate a list
                    insB = list(map(lambda x:x.op.lower(), ker_node.instructions))
                    # print(insA, insB)
                    if len(insA) < 3 or len(insB) < 3:
                        unigram_mat = get_unigram_matrix(insA, insB)
                        point_bb, m = hung(np.array(unigram_mat))
                        if point_bb == 0.0:
                            point_bb = 0.1
                            # print(unigram_mat)
                            # print(point_bb, len(m))
                        if len(m) == 0:
                            return -1, -1  # bad graph
                        tmp.append((point_bb / len(m)) * 3)
                    else:
                        ngram_mat = get_ngram_matrix(insA, insB)
                        point_bb, m = hung(np.array(ngram_mat))
                        tmp.append(point_bb / len(m))
            mat.append(tmp)
    # print(mat)
    points, matches = hung(np.array(mat))
    # print(points)
    for pair in matches:
        list_vul[pair[0]].set_matched_bb(list_ker[pair[1]])
        list_ker[pair[1]].set_matched_bb(list_vul[pair[0]])
    return points, len(matches)


# using Hungarian Algorithm to calculate the similarity of every pair of unmatched bbs based on the instruction similarity. Then form a matrix of bbs' similarity, and use Hungarian Algorithm again to optimize the matches of every pair of unmatched bbs. output is the total points of the optimized pairs and the numbers of matched pairs.
# if number of instruction less than 3, the matching result of the two bb should be 3.
def fuzzy_match2(cfg_vul, cfg_kernel):
    list_vul = []
    list_ker = []
    mat = []   # the matrix for Hungarian Algorithm
    for vul_node in cfg_vul.vertices:
        if vul_node.matched_bb:
            continue
        else:
            list_vul.append(vul_node)
            tmp = []
            for ker_node in cfg_kernel.vertices:
                if ker_node.matched_bb:
                    continue
                else:
                    point_bb = 3
                    list_ker.append(ker_node)
                    ngram_mat = get_ngram_matrix([i.op.lower() for i in vul_node.instructions], list(map(lambda x:x.op.lower(), ker_node.instructions)))
                    if len(ngram_mat) == 0 or len(ngram_mat[0]) == 0:
                        tmp.append(point_bb)
                    else:
                        # print(ngram_mat)
                        point_bb, m = hung(np.array(ngram_mat))
                        tmp.append(point_bb/len(m))
            mat.append(tmp)
    # print(mat)
    points, matches = hung(np.array(mat))
    # print(points)
    for pair in matches:
        list_vul[pair[0]].set_matched_bb(list_ker[pair[1]])
        list_ker[pair[1]].set_matched_bb(list_vul[pair[0]])
    return points, len(matches)


def fuzzy_match(cfg_vul, cfg_kernel):    # greedily match every unmatched node in ker_cfg and may miss the globally optimal solution
    #ker_bb_queue = [cfg_kernel.entry]
    vul_bb_queue = [cfg_vul.entry]
    points = []
    seen_vul = [cfg_vul.entry]
    while vul_bb_queue:
        cur_vul_bb = vul_bb_queue.pop(0)
        for n in cur_vul_bb.out_degree:
            if n not in seen_vul:
                vul_bb_queue.append(n)
                seen_vul.append(n)
        if not cur_vul_bb.matched_bb:
            res = float('inf')
            matched_ker_bb = None
            seen_ker = [cfg_kernel.entry]
            ker_bb_queue = [cfg_kernel.entry]
            while ker_bb_queue:
                cur_ker_bb = ker_bb_queue.pop(0)
                for n in cur_ker_bb.out_degree:
                    if n not in seen_ker:
                        ker_bb_queue.append(n)
                        seen_ker.append(n)
                if not cur_ker_bb.matched_bb:
                    tmp = hung(np.array(get_ngram_matrix([i.op for i in cur_vul_bb.instructions], list(map(lambda x:x.op, cur_ker_bb.instructions)))))
                    print(tmp)
                    if tmp < res:
                         res = tmp
                         matched_ker_bb = cur_ker_bb
            if matched_ker_bb:
                points.append(res)
                matched_ker_bb.set_matchable()
                matched_ker_bb.set_matched_bb(cur_vul_bb)
                cur_vul_bb.set_matchable()
                cur_vul_bb.set_matched_bb(matched_ker_bb)

def graph_match(cfg1, cfg2):
    num_init_match = init_match2(cfg1, cfg2)
    points, num_fuzzy_match = fuzzy_match3(cfg1, cfg2)
    if points == -1 and num_fuzzy_match == -1:
        return -1  # bad graph
    match_points = (points + (len(cfg1.vertices) - num_init_match - num_fuzzy_match) * 3) / len(cfg1.vertices)
    return match_points

def MEDHungarian(cfg1, cfg2):
    points, num_fuzzy_match = fuzzy_match3(cfg1, cfg2)
    if points == -1 and num_fuzzy_match == -1:
        return -1  # bad graph
    match_points = (points + (len(cfg1.vertices) - num_fuzzy_match) * 3) / len(cfg1.vertices)
    return match_points
    



#insA = ['m', 'm', 'c', 'j', 'p', 'ca', 'p', 'p', 'r']
#insB = ['m', 'm', 'x', 'c', 'je', 'ca', 'p', 'p']
#print(get_ngram_matrix(insA, insB))
#print(hung(np.array(get_ngram_matrix(insA, insB))))
#graph = {1:[2,3,4],2:[5,6],3:[4,11],5:[9,10],4:[7,8],7:[11,12]}
#cfg_files = ['./aarch64_uaf_conditional_0_cfg', 
#             './aarch64_uaf_conditional_1_cfg', 
#             './aarch64_uaf_direct_cfg', 
#             './aarch64_uaf_direct_int_cfg', 
#             './src_cfg', 
#             './x86_64_uaf_direct_cfg']
#with open(cfg_files[0], 'rb') as f:
#    cfg_list1 = pickle.load(f)
#
#with open(cfg_files[1], 'rb') as f:
#    cfg_list2 = pickle.load(f)
#
#print(len(cfg_list1))
#for cfg in cfg_list1:
#    print(cfg.function_name)
#print(len(cfg_list2))
#for cfg in cfg_list2:
#    print(cfg.function_name)
#
#for i in range(0, len(cfg_list1)):
#    print(cfg_list1[i].function_name, ' -- ' , cfg_list2[i].function_name)
#cfg1 = None
#cfg2 = None
#for i in range(len(cfg_list1)):
#    if cfg_list1[i].function_name == 'may_be_vulnerable':
#        cfg1 = cfg_list1[i]
#        print('1:',i)
#for i in range(len(cfg_list2)):
#    if cfg_list2[i].function_name == 'may_be_vulnerable':
#        cfg2 = cfg_list2[i]
#        print('2:',i)
#
#
#dic = {}
#dic2 = {}
#start = ord('A')
#for bb in cfg1.vertices:
#    dic[bb.addr] = chr(start)
#    start+=1
#    #print(bb)
#    print(dic[bb.addr],':', bb.entry_bb, bb.bb_exit, len(bb.in_degree), len(bb.out_degree))
#start2 = ord('a')
#for bb in cfg2.vertices:
#    dic2[bb.addr] = chr(start2)
#    start2 += 1
#    #print(bb)
##    print( bb.entry_bb, bb.bb_exit, len(bb.in_degree), len(bb.out_degree))
#
##for edge in cfg1.edges:
##    print(dic[edge[0].addr], '--->', dic[edge[1].addr])
#for edge in cfg2.edges:
#    print(dic2[edge[0].addr], '--->', dic2[edge[1].addr])
#num_init_match = 0
#if len(cfg1.vertices) > 1 and len(cfg2.vertices) > 1:
#    num_init_match = init_match2(cfg1, cfg2)
#
#print(num_init_match)
#for bb in cfg2.vertices:
#    print(bb.matched_bb)
#    if bb.matched_bb:
#        print(dic[bb.matched_bb.addr])
#print('\n\nfuzzy_match')

#points, num_fuzzy_match = fuzzy_match2(cfg1,cfg2)
#print('total:',len(cfg1.vertices))
#print('init_match_num:',num_init_match)
#print('fuzzy_match_num:',num_fuzzy_match)
#match_points = (points + (len(cfg1.vertices) - num_init_match - num_fuzzy_match) * 3) / len(cfg1.vertices)
#print(match_points)

#for bb in cfg2.vertices:
#    print(dic[bb.matched_bb.addr])
