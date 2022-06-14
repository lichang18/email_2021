import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt

#计算信息熵
def getEntropy(s):
    # 找到各个不同取值出现的次数
    if not isinstance(s, pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = s.groupby(by=s).count().values / float(len(s))
    return -(np.log2(prt_ary) * prt_ary).sum()


#计算条件熵: 条件s1下s2的条件熵
def getCondEntropy(s1, s2):
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]] = d.get(s1[i], []) + [s2[i]]
    return sum([getEntropy(d[k]) * len(d[k]) / float(len(s1)) for k in d])


#计算信息增益
def getEntropyGain(s1, s2):
    return getEntropy(s2) - getCondEntropy(s1, s2)


#计算增益率
def getEntropyGainRadio(s1, s2):
    return getEntropyGain(s1, s2) / getEntropy(s2)


#衡量离散值的相关性
import math


def getDiscreteCorr(s1, s2):
    return getEntropyGain(s1, s2) / math.sqrt(getEntropy(s1) * getEntropy(s2))


#计算概率平方和
def getProbSS(s):
    if not isinstance(s, pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = pd.groupby(s, by=s).count().values / float(len(s))
    return sum(prt_ary ** 2)


#计算基尼系数
def getGini(s1, s2):
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]] = d.get(s1[i], []) + [s2[i]]
    return 1 - sum([getProbSS(d[k]) * len(d[k]) / float(len(s1)) for k in d])


## 对离散型变量计算相关系数，并画出热力图, 返回相关性矩阵
def DiscreteCorr(C_data):
    ## 对离散型变量(C_data)进行相关系数的计算
    C_data_column_names = C_data.columns.tolist()
    ## 存储C_data相关系数的矩阵
    import numpy as np
    dp_corr_mat = np.zeros([len(C_data_column_names), len(C_data_column_names)])
    for i in range(len(C_data_column_names)):
        for j in range(len(C_data_column_names)):
            # 计算两个属性之间的相关系数
            temp_corr = getDiscreteCorr(C_data.iloc[:, i], C_data.iloc[:, j])
            dp_corr_mat[i][j] = temp_corr
    # 画出相关系数图
    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    sns.heatmap(dp_corr_mat, vmin=- 1, vmax=1, cmap=sns.color_palette('RdBu', n_colors=128),
                xticklabels=C_data_column_names, yticklabels=C_data_column_names)
    return pd.DataFrame(dp_corr_mat)

'''
if __name__ == "__main__":

    #每一列0-23小时的每个特征的信息熵计算，再画图看出排布
    filename = r'D:\youjianrizhi\taiwan\try\entropy_4_8.csv'
    file = np.loadtxt(filename, delimiter=',')
    names = locals()
    for i in range(0, 17):
        names['column' + str(i)] =np.loadtxt(filename, delimiter=',', usecols=(i))

    #每一列每天组成一个数组new+序号
    for k in range(0, 31):
        names['new' + str(k)] = []
    #每列每天组成一个数组new+天的序号
    k = 0
    for j in range(len(file)):
        if column2[j]!=column2[j+1]:#判断是否是不同的天，先实现手动每列，没有问题了在实现自动
            names['new' + str(k)].append(column2[j])#是不同天，记录上一天的最后一条，跳出循环
            k = k + 1
            continue
        else:
            names['new' + str(k)].append(column2[j])#是同一天，记录数据
    #print(new1)
    #每一个new是一个series，每一个s分别算出来信息熵
    for m in range(0,30):
        names['s'+str(m)] = pd.Series(names['new' + str(m)])

    for n in range(0,18):
        names['entropy' + str(n)] = np.array([getEntropy(names['s'+str(m)])])
    #print(entropy0)


    s1 = pd.Series(['X1', 'X1', 'X2', 'X2', 'X2', 'X2'])
    s2 = pd.Series(['Y1', 'Y1', 'Y1', 'Y2', 'Y2', 'Y2'])
    print('CondEntropy:', getCondEntropy(s1, s2))
    print('EntropyGain:', getEntropyGain(s1, s2))
    print('EntropyGainRadio', getEntropyGainRadio(s1, s2))
    print('DiscreteCorr:', getDiscreteCorr(s1, s1))
    print('Gini', getGini(s1, s2))

    
        column0 = np.loadtxt(filename, delimiter=',', usecols=(0))
        column1 = np.loadtxt(filename, delimiter=',', usecols=(1))
        column2 = np.loadtxt(filename, delimiter=',', usecols=(2))
        column3 = np.loadtxt(filename, delimiter=',', usecols=(3))
        column4 = np.loadtxt(filename, delimiter=',', usecols=(4))
        column5 = np.loadtxt(filename, delimiter=',', usecols=(5))
        column6 = np.loadtxt(filename, delimiter=',', usecols=(6))
        column7 = np.loadtxt(filename, delimiter=',', usecols=(7))
        column8 = np.loadtxt(filename, delimiter=',', usecols=(8))
        column9 = np.loadtxt(filename, delimiter=',', usecols=(9))
        column10 = np.loadtxt(filename, delimiter=',', usecols=(10))
        column11= np.loadtxt(filename, delimiter=',', usecols=(11))
        column12 = np.loadtxt(filename, delimiter=',', usecols=(12))
        column13 = np.loadtxt(filename, delimiter=',', usecols=(13))
        column14 = np.loadtxt(filename, delimiter=',', usecols=(14))
        column15 = np.loadtxt(filename, delimiter=',', usecols=(15))
        column16 = np.loadtxt(filename, delimiter=',', usecols=(16))
        column17 = np.loadtxt(filename, delimiter=',', usecols=(17))
'''

'''
#每月的熵，各个列的
filename = r'D:\youjianrizhi\taiwan\try\entropy_4_8.csv'
file = np.loadtxt(filename, delimiter=',')
column0 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(0)))
column1 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(1)))
column2 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(2)))
column3 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(3)))
column4 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(4)))
column5 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(5)))
column6 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(6)))
column7 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(7)))
column8 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(8)))
column9 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(9)))
column10 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(10)))
column11 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(11)))
column12 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(12)))
column13 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(13)))
column14 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(14)))
column15 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(15)))
column16 = getEntropy(np.loadtxt(filename, delimiter=',', usecols=(16)))
print(column1)
'''
#每周的熵，各个列的
filename = r'D:\youjianrizhi\taiwan\try\entropy_4_8.csv'
file = np.loadtxt(filename, delimiter=',')
names = locals()
for i in range(0, 17):
    names['column' + str(i)] = np.loadtxt(filename, delimiter=',', usecols=(i))
#每一列每天组成一个数组new+序号
    for k in range(0, 30):
        names['new' + str(k)] = []
#每列每天组成一个数组new+天的序号

k = 0
if k==7:
    k = 0
    continue
else:
    k = k+1

for j in range(len(file)):
    if column10[j]!=column10[j+1]:#判断是否是不同的天，先实现手动每列，没有问题了在实现自动
        names['new' + str(k)].append(column10[j])#是不同天，记录上一天的最后一条，跳出循环
        k = k + 1
        continue
    else:
        names['new' + str(k)].append(column10[j])#是同一天，记录数据
    #print(new1)
    #每一个new是一个series，每一个s分别算出来信息熵
for m in range(0,30):
    names['s'+str(m)] = pd.Series(names['new' + str(m)])

for n in range(0,18):
    names['entropy' + str(n)] = np.array([getEntropy(names['s'+str(m)])])
    print(entropy0)