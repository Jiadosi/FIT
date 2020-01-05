import tensorflow as tf
import string
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import layers

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
        npData = np.array()
        for line in data:
            line = line.split()
            if len(line) <= 35:
                npData.append(np.array(line))
        return npData

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

if __name__ == "__main__":
    loadData = LoadData()
    print(loadData.text.shape)