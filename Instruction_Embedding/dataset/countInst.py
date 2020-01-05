#!/usr/bin/env python
# coding=utf-8

import os
import json

def preparing(dirPath):
    print('---preparing dataset---')
    for f in os.listdir(dirPath):
        filePath = ''
        if 'mips' in f:
            filePath = os.path.join(dirPath, f)
        if not filePath:
            continue
        print(filePath)
        with open(filePath, 'r') as f:
            data = f.readlines()
        maxLen = 0
        for line in data:
            g = json.loads(line)
            for bb in g['features']:
                maxLen = len(bb[-1]) if len(bb[-1]) > maxLen else maxLen
        print(maxLen)

def statistic():
    # with open('./w2v_arm_dataset.txt') as f:  # 4642
    #     data = f.readlines()
    # with open('./w2v_mips_dataset.txt') as f:  # 7491
    #     data = f.readlines()
    with open('./w2v_x86_dataset.txt') as f:  # 1710
        data = f.readlines()
    count = [0] * 1711
    # maxLen = 0
    for line in data:
        insts = line.split(' ')
        # maxLen = len(insts) if len(insts) > maxLen else maxLen
        count[len(insts)] += 1

    total = 0
    for i in range(1, len(count)):
        total += count[i]
        if i == 10:
            print('1-10:', total)
            total = 0
        elif i == 25:
            print('11-25: ', total)
            total = 0
        elif i == 35:
            print('26-35: ', total)
            total = 0
        elif i == 45:
            print('36-45: ', total)
            total = 0
        elif i == 50:
            print('46-50: ', total)
            total = 0
        elif i == 75:
            print('51-75: ', total)
            total = 0
        elif i == 100:
            print('76-100:', total)
            total = 0
    print('> 100: ', total)

if __name__ == "__main__":
    dirPath = './filtered_json_inst'
    # preparing(dirPath)
    statistic()

