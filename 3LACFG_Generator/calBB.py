import json
import os

dir = './json'
files = os.listdir(dir)

for file in files:
    fp = os.path.join(dir, file)
    print(fp)
    with open(fp, 'r') as f:
        jdata = json.load(f)
    bbc = 0
    for jline in jdata:
        bbc += jline['n_num']
    print(bbc/len(jdata))
