import idaapi
import idautils
import idc
import os

from func import *
from raw_graphs import *

def analysis(path):
    cfgs = get_func_cfgs_c(FirstSeg())
    binary_name = idc.GetInputFile() + '.ida'
    fullpath = os.path.join(path, binary_name)
    pickle.dump(cfgs, open(fullpath,'w'))
    idc.Exit(0)

def main():
    path = '/Users/eacials/Downloads/Gencoding-master/output'
    idaapi.autoWait()
    analysis(path)
    idc.Exit(0)

if __name__ == "__main__":
    main()