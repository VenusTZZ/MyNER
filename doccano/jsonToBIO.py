#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:Venus
@file:jsonToBIO.py
@time:2022/10/01/19：27
"""
import json


def generate_json():
    '''将标注系统下载下来的文件转换为标准json格式'''
    f1 = open('out.json', 'w', encoding='utf-8')
    f1.write("[")
    with open('admin.jsonl', 'r', encoding='utf-8') as f2:
        lines = f2.readlines()
        k = len(lines)
        i = 0
        while i < k - 1:
            f1.write(lines[i].strip() + ',\n')
            i += 1
        f1.write(lines[i].strip() + '\n')
    f1.write(']')
    f1.close()


def tranfer2bio():
    f1 = open('./example.train', 'w', encoding='utf-8')
    with open("out.json", 'r', encoding='utf-8') as inf:
        load = json.load(inf)
        for i in range(int(len(load))):
            labels = load[i]['label']
            text = load[i]['text']
            tags = ['O'] * len(text)
            for j in range(len(labels)):
                label = labels[j]
                # print(label)
                tags[label[0]] = 'B-' + str(label[2])
                k = label[0] + 1
                while k < label[1]:
                    tags[k] = 'I-' + str(label[2])
                    k += 1
            print(tags)
            for word, tag in zip(text, tags):
                f1.write(word + '\t' + tag + '\n')
            f1.write("\n")

if __name__ == '__main__':

    generate_json()
    tranfer2bio()
