#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:Mypredata.py
@time:2022/10/09/18：32
"""

if __name__ == '__main__':
    file = open('example.train', 'rt', encoding='utf-8')
    lines = file.readlines()
    # 删除行
    k = 0
    for line in lines[40011:-1]:
        lines.remove(line)
    print(k)
