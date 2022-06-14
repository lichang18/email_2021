import csv
import os
from collections import Counter
#-*- coding : utf-8-*-
# coding:unicode_escape

# filepath = r'E:\data\imap'
#文件夹目录
import pandas as pd


def mta_classify(filepath):
    list1 = []  # (无代理无外省)
    list2 = []  # (无代理有外省)
    list3 = []  # (有代理)
    list4 = []  # (全是失败行为)
    fileNames = os.listdir(filepath)  # 获取当前路径下的文件名，返回List
    filelist = []
    for file in fileNames:   #遍历文件夹
        folder = os.path.join(filepath,file)

        country = []
        city = []



        with open(folder, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.split(";")
                if line[1] == '8' or line[1] == '79' or line[1] == '84': #失败行为编号
                    continue
                else:
                    country.append(line[7])
                    city.append(line[9])

            country_most = Counter(country).most_common(1)
            city_most = Counter(city).most_common(1)
            if len(country) < 2:
                list4.append(file)
            elif country_most[0][1] >= 1 / 2 * len(country) and city_most[0][1] < 1 / 2 * len(city):
                list2.append(file)
            elif country_most[0][1] >= 1 / 2 * len(country) and city_most[0][1] >= 1 / 2 * len(city):
                list1.append(file)
            else:
                list3.append(file)
    return list1, list2, list3, list4

def imap_classify(filepath):
    list1 = []  # (无代理无外省)
    list2 = []  # (无代理有外省)
    list3 = []  # (有代理)
    list4 = []  # (全是失败行为)
    fileNames = os.listdir(filepath)  # 获取当前路径下的文件名，返回List
    filelist = []
    for file in fileNames:   #遍历文件夹
        folder = os.path.join(filepath,file)
        #folder = filepath + "\\" + file  #文件中的文件名

        behavior = []
        country = []
        city = []



        with open(folder, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.split(";")
                if line[1] == '2': #失败行为编号
                    continue
                else:
                    country.append(line[7])
                    city.append(line[9])

            country_most = Counter(country).most_common(1)
            city_most = Counter(city).most_common(1)
            # print(country_most)
            # print(city_most)
            # print(len(country))
            if len(country) < 2:
                list4.append(file)
            elif country_most[0][1] >= 1 / 2 * len(country) and city_most[0][1] < 1 / 2 * len(city):
                list2.append(file)
            elif country_most[0][1] >= 1 / 2 * len(country) and city_most[0][1] >= 1 / 2 * len(city):
                list1.append(file)
            else:
                list3.append(file)
    return list1, list2, list3, list4

def pop_classify(filepath):
    list1 = []  # (无代理无外省)
    list2 = []  # (无代理有外省)
    list3 = []  # (有代理)
    list4 = []  # (全是失败行为)
    fileNames = os.listdir(filepath)  # 获取当前路径下的文件名，返回List
    filelist = []
    for file in fileNames:   #遍历文件夹
        folder = os.path.join(filepath,file)
        country = []
        city = []
        with open(folder, 'r', encoding='utf-8', errors='ignore') as f:
            s = set()
            m = set()
            for line in f:
                line = line.split(";")
                if line[1] == '122':
                    m.add(line[19])
                    continue
                elif line[6] == "":  #删除IP为空的行
                    continue
                elif line[19] == "":  #删除processQ为空的行
                    continue
                elif line[19] not in s:  #对processQ去重，只保留第一个
                    s.add(line[19])
                elif line[19] in s:
                    continue
                elif line[19] in m:  #和122一样的Q的行去重
                    continue
                else:
                    country.append(line[7])
                    city.append(line[9])
            #print(s)
            country_most = Counter(country).most_common(1)
            city_most = Counter(city).most_common(1)
            # print(country_most)
            # print(city_most)
            # print(len(country))
            if len(country) < 2:
                list4.append(file)
            elif country_most[0][1] >= 1 / 2 * len(country) and city_most[0][1] < 1 / 2 * len(city):
                list2.append(file)
            elif country_most[0][1] >= 1 / 2 * len(country) and city_most[0][1] >= 1 / 2 * len(city):
                list1.append(file)
            else:
                list3.append(file)
    return list1, list2, list3, list4
# print('List1(无代理无外省):', list1)
# print('List2(无代理有外省):', list2)
# print('List3(有代理):', list3)
# print('List4(都是失败行为):', list4)

if __name__ == "__main__":
    data_path = "E:\data\pop2"
    list1, list2, list3, list4 = pop_classify(filepath=data_path)




