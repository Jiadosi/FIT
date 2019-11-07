import os
import json

def formDic(container, path):
    with open(path) as f:
        line = f.readline()
        while line:
            js = json.loads(line.strip())
            if js['n_num'] >= 5:
                if js['fname'] not in container:
                    container[js['fname']] = []
                container[js['fname']].append((path, js))
            line = f.readline()


path = './json/'
fileList = os.listdir(path)
mipsDic, armDic, x86Dic = {}, {}, {}
res = []

for f in fileList:
    features = f.split('_')
    if features[0] == 'openssl':
        if features[2] == 'mips':
            formDic(mipsDic, os.path.join(path, f))
        elif features[2] == 'arm':
            formDic(armDic, os.path.join(path, f))
        elif features[2] == 'x86':
            formDic(x86Dic, os.path.join(path, f))

cnt = 0
for key in mipsDic.keys():
    if key in armDic and key in x86Dic:
        cnt += 1
        print(key)
        print('inmips', len(mipsDic[key]))
        print('inarm', len(armDic[key]))
        print('x86Dic', len(x86Dic[key]))
        tot_occur = len(mipsDic[key]) + len(armDic[key]) + len(x86Dic[key])
        # res.append(key)
        print(tot_occur)
        
        for f in mipsDic[key]:
            fpath = os.path.join('./filtered_json/', f[0].split('/')[-1])
            with open(fpath, 'a+') as jf:
                data = json.dumps(f[1])
                jf.write(data + '\n')
        for f in armDic[key]:
            fpath = os.path.join('./filtered_json/', f[0].split('/')[-1])
            with open(fpath, 'a+') as jf:
                data = json.dumps(f[1])
                jf.write(data + '\n')
        for f in x86Dic[key]:
            fpath = os.path.join('./filtered_json/', f[0].split('/')[-1])
            with open(fpath, 'a+') as jf:
                data = json.dumps(f[1])
                jf.write(data + '\n')
print(cnt)

