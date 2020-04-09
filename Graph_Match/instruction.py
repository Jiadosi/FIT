#!/usr/bin/env python
# coding=utf-8

class instruction:
    def __init__(self, op):
        self.op = op
        self.operand = []

    def set_operand(self, str):
        self.operand.append(str)
