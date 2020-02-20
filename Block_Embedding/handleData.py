import tensorflow as tf
import string
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import layers

import sys
sys.path.append('..')
import os
import json
from Instruction_Embedding.instEmbedding import loading

first_letter = ord(string.ascii_lowercase[0])

class LoadData(object):
    def __init__(self, valid_size=1000):
        self.text = self._read_data()
        self.valid_text = self.text[:valid_size]
        self.train_text = self.text[valid_size:]

    def _read_data(self, filename='../Instruction_Embedding/dataset/w2v_arm_dataset.txt'):
        with open(filename) as f:
            data = f.readlines()

        # pruning, keep bb smaller than 35 insts
        npData = []
        for line in data:
            line = line.split()
            if len(line) <= 35:
                npData.append(np.array(line))
        return np.array(npData)

    def _prepro_data(data):
        # padding
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, padding = 'post', truncating='post', maxlen=5)
        print('padded_inputs', padded_inputs)


def char2id(char):
    # 将字母转换成id
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print("Unexpencted character: %s " % char)
        return 0

def id2char(dictid):
    # 将id转换成字母
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '

def characters(probabilities):
    # 根据传入的概率向量得到相应的词
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    # 用于测试得到的batches是否符合原来的字符组合
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

def inst2embedding(filePath):
    with open(os.path.join('../Instruction_Embedding/dataset/filtered_json_inst', filePath)) as f:
        data = f.readlines()

    with open(os.path.join('../Instruction_Embedding/dataset/filtered_json_inst/instEmbed/', filePath), 'w') as f:
        # load w2v
        if 'arm' in filePath:
            print('loading arm model')
            model = loading('../Instruction_Embedding/myModel/arm')
        elif 'x86' in filePath:
            print('loading x86 model')
            model = loading('../Instruction_Embedding/myModel/x86')
        elif 'mips' in filePath:
            print('loading mips model')
            model = loading('../Instruction_Embedding/myModel/mips')

        for line in data:
            line = json.loads(line)
            instsEmbed = []
            for bb in line['features']:
                for inst in bb[-1]:
                    try:
                        embed = model[inst].tolist()
                        instsEmbed.append(embed)
                    except Exception as e:
                        print(e)
                bb.remove(bb[-2])
                bb[-1] = instsEmbed
            f.write(json.dumps(line))
            f.write('\n')


if __name__ == "__main__":
    # loadData = LoadData()
    # print(loadData.text.shape)
    for f in os.listdir('../Instruction_Embedding/dataset/filtered_json_inst'):
        if 'openssl' not in f:
            continue
        print(f)
        inst2embedding(f)