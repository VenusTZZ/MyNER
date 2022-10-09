#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:predata.py
@time:2022/10/08/17：50
"""
import json

if __name__ == '__main__':
    f1 = open('predata1.txt', 'w', encoding='utf-8')
    with open('0-结构化-非结构化数据汇总-换行.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        k = len(lines)
        for line in lines:
            line = line.replace(" ", "")
            if line:
                print(line)

        while i < k - 2:
            f1.write(lines[i].strip() + ',\n')
            i += 1
        f1.write(lines[i].strip() + '\n')
    f1.write(']')
    f1.close()