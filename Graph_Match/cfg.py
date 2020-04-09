#class cfg
#@author dosi
#@category _NEW_
#@keybinding 
#@menupath 
#@toolbar 

import numpy as np

class cfg():
    def __init__(self, func):
        self.function_name = func
        self.entry = None
        self.exit = []
        self.vertices = []
        self.edges = []
        self.no_of_bbs = 0
        self.no_of_edges = 0
        self.no_of_func_calls = 0
        self.no_of_incoming_calls = 0
        self.no_of_logic_inst = 0
        self.no_of_trans_inst = 0
        self.no_of_inst = 0
        self.no_of_local_var = 0
        self.no_of_global_var = 0
        self.in_degree_dict = {}
        self.out_degree_dict = {}
        self._distance_matrix = None
        self._distance_matrix_trans = None
        self._shortest_path_from_entry = {}
        self._shortest_path_to_exit_list = []

    def set_no_of_func_calls(self, no):
        self.no_of_func_calls = no

    def set_no_of_incoming_calls(self, no):
        self.no_of_incoming_calls = no

    def set_no_of_inst(self, no):
        self.no_of_inst = no

    def set_no_of_logic_inst(self, no):
        self.no_of_logic_inst = no

    def set_no_of_trans_inst(self, no):
        self.no_of_trans_inst = no

    def set_no_of_local_var(self, no):
        self.no_of_local_var = no

    def set_no_of_global_var(self, no):
        self.no_of_global_var = no

    def set_entry_bb(self, bb):
        if self._shortest_path_from_entry:
            bb_idx = self.vertices.index(bb)
            bb.entry_bb = self._shortest_path_from_entry[bb_idx]

    def set_bb_exit(self, bb):
        if self._shortest_path_to_exit_list:
            bb_idx = self.vertices.index(bb)
            exit_bb_list = [x[bb_idx] for x in self._shortest_path_to_exit_list]
            bb.bb_exit = min(exit_bb_list)

    def _shortest_path(self):
        try:
            if self._distance_matrix:
                if self.entry:
                    self._shortest_path_from_entry = self._dijskstra(0, 1)
                if self.exit:
                    for e in self.exit:
                        e_idx = self.vertices.index(e)
                        self._shortest_path_to_exit_list.append(self._dijskstra(e_idx, 0))
        except Exception as e:
            print(e)
            print('something wrong in shortest_path, return None None')
            return None, None
    
    def _dijskstra(self, idx, entry):
        dis = {}
        _distance_matrix = [self._distance_matrix_trans, self._distance_matrix]
        try:
            for i in range(len(_distance_matrix[entry][idx])):
                if _distance_matrix[entry][idx][i] != 0:
                    dis[i] = _distance_matrix[entry][idx][i]
                
            visited = []
            min_dis = None
            min_dis_point = None
            for i in range(len(dis)):
                sorted_dis = sorted(dis.items(), key = lambda item: item[1])
                for p, d in sorted_dis:
                    if p not in visited:
                        min_dis_point = p
                        min_dis = d
                        visited.append(p)
                        break
                for j in range(1, len(_distance_matrix[entry])):
                    if _distance_matrix[entry][min_dis_point][j] < float('inf'):
                        update = min_dis + _distance_matrix[entry][min_dis_point][j]
                        if update < dis[j]:
                            dis[j] = update
        except:
            print('something wrong in dijskstra')
        return dis

    def _gen_dis_mat(self):
        #weight of edge equals to 1
        self._distance_matrix = [[float('inf') for _ in range(len(self.vertices))] for _ in range(len(self.vertices))]
        for i in range(len(self._distance_matrix)):
            self._distance_matrix[i][i] = 0
        try:
            if self.out_degree_dict:
                for src, desList in self.out_degree_dict.items():
                    srcIdx = self.vertices.index(src)
                    for des in desList:
                        desIdx = self.vertices.index(des)
                        self._distance_matrix[srcIdx][desIdx] = 1
            np_array = np.array(self._distance_matrix)
            self._distance_matrix_trans = np_array.T.tolist()
        except:
            print('something wrong in generate distance matrix')
        
    def in_degree(self, t_bb):
        if not self.in_degree_dict or not self.in_degree_dict.__contains__(t_bb):
            t_bb.in_degree = []
        else:
            t_bb.in_degree = self.in_degree_dict[t_bb]

    def _gen_in_degree_dict(self):
        if self.edges:
            for e in self.edges:
                if e[1] not in self.in_degree_dict:
                    self.in_degree_dict[e[1]] = []
                self.in_degree_dict[e[1]].append(e[0])
            
    
    def out_degree(self, t_bb):
        if not self.edges or not self.out_degree_dict.__contains__(t_bb):
            t_bb.out_degree = []
        else:
            t_bb.out_degree = self.out_degree_dict[t_bb]

    def _gen_out_degree_dict(self):
        if self.edges:
            for e in self.edges:
                if e[0] not in self.out_degree_dict:
                    self.out_degree_dict[e[0]] = []
                self.out_degree_dict[e[0]].append(e[1])

    def all_children(self, t_bb):
        #implemented in other py
        pass

    def set_entry(self, e_bb):
            self.entry = e_bb

    def set_exit(self, e_bb):
            self.exit.append(e_bb)

    def set_vertices(self, bb):
            self.vertices.append(bb)
            self.no_of_bbs += 1

    def set_edges(self, bb, dest_bb):
            self.edges.append((bb, dest_bb))
            self.no_of_edges += 1

    #call this function when cfgraph is completed
    def prepare(self):
        self._gen_in_degree_dict()
        self._gen_out_degree_dict()
        self._gen_dis_mat()
        self._shortest_path()
        for v in self.vertices:
            self.in_degree(v)
            self.out_degree(v)
            v.set_betweenness()
            #v.entry_bb, v.bb_exit = self.shortest_path(v)
            

if __name__ == '__main__':
    cfgraph = cfg('init')
    cfgraph.vertices = [0,1,2,3]
    cfgraph.edges = [(0,1), (1,2), (2,3), (0,3)]
    cfgraph.set_entry(0)
    cfgraph.set_exit(3)
    cfgraph._gen_in_degree_dict()
    cfgraph._gen_out_degree_dict()
    print(cfgraph.in_degree_dict)
    print(cfgraph.out_degree_dict)
    cfgraph._gen_dis_mat()
    print('dis_mat', cfgraph._distance_matrix)
    cfgraph._shortest_path()
    cfgraph.set_entry_bb(2)
    cfgraph.set_bb_exit(2)
