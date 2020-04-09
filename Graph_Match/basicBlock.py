#class basicBlock
#@author dosi
#@category _NEW_
#@keybinding 
#@menupath 
#@toolbar 


#TODO Add User Code Here
class basicBlock():
    def __init__(self, addr):
        self.addr = addr
        self.matchable = True # can be matched
        self.matched_bb = None # matched bb from other cfg
        self.instructions = []
        self.no_of_inst = 0
        self.no_of_arithmetic_inst = 0
        self.no_of_transfer_inst = 0
        self.no_of_incoming_calls = 0
        self.no_of_function_calls = 0
        self.entry_bb = 0
        self.bb_exit = 0
        self.in_degree = []
        self.out_degree = []
        self.betweenness = 0

    def set_matchable(self):
        self.matchable = False

    def set_matched_bb(self, bb):
        self.matched_bb = bb

    def set_instructions(self, ins_list):
        self.instructions = ins_list
        self.no_of_inst = len(self.instructions)

    def set_no_of_arithmetic_inst(self, no):
        self.no_of_arithmetic_inst = no

    def set_no_of_transfer_inst(self, no):
        self.no_of_transfer_inst = no

    def set_no_of_incoming_calls(self, no):
        self.no_of_incoming_calls = no

    def set_no_of_function_calls(self, no):
        self.no_of_function_calls = no

    def set_betweenness(self):
        self.betweenness = len(self.in_degree) + len(self.out_degree)


