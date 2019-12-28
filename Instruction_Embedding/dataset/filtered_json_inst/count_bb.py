#!/usr/bin/env python
# coding=utf-8

import json
import os


bb_x86 = 0
bb_mips = 0
bb_arm = 0
for f in os.listdir('./'):
    print(f)
    if 'arm' in f:
        with open(f, 'r') as ff:
            data = ff.readlines()
        for line in data:
            cfg = json.loads(line)
            bb_arm += cfg['n_num']
    elif 'mips' in f:
        with open(f, 'r') as ff:
            data = ff.readlines()
        for line in data:
            cfg = json.loads(line)
            bb_mips += cfg['n_num']
    elif 'x86' in f:
        with open(f, 'r') as ff:
            data = ff.readlines()
        for line in data:
            cfg = json.loads(line)
            bb_x86 += cfg['n_num']
print('arm', bb_arm)
print('x86', bb_x86)
print('mips', bb_mips)

