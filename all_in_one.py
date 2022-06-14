from pyclbr import Function
from typing import Callable, List
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
import os
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity

mta_header = "abroad	behavior	yyyyMMdd	hhmmss	account	domainName	IP	country	province	city	operator	netAffiliation	institute	latitude	longitude	ssl	rcpt	state	size	keywords	Q	Tid	fromDN	AuthFailCnt	optimes	errinfo	origip	Xmailer	SenderEmail	Local	org_unit_id	lrcptcnt	rrcptcnt	LmtpRcptCnt	DataRuleID	Score	AttachCnt	DataFngCnt	DataSFngCnt	DKimVerifyResult	Subject	SubjectCnt	Handle	Content	ttime	UsrQuarterCnt	UsrTodayCnt	conn	region	quarterCnt	todayCnt	todayRcptCnt	hangup	errStat	delivered	userSendInterval	helo	mdRet	errCnt	deliveredCnt	domainQuarterCnt	domainTodayCnt	MailCnt".split("\t")
mta_header.append("Label")
mta_header = [item.strip() for item in mta_header]

imap_header = "abroad	 behavior	 yyyyMMdd	 hhmmss	 account	 domainName	 IP	 country	 province	 city	 operator	 netAffiliation	 institute	 latitude	 longitude	 keywordBehind	 root	 udid	 mid	 set	 removeFlag	 size	 msid	 xmailer	 rcpt	 MIMEVersion	 fid".split("\t")
imap_header.append("Label")
imap_header = [item.strip() for item in imap_header]

pop_header = "abroad	behavior	create_Date	time	account	domainName	IP	country	province	city	operator	netAffiliation	institute	dynamicAttr	dynamicNum	keywords	mBoxID	MSID	mid	processQ 	num 	data 	reason 	udid 	msgcount 	msgsize 	currentFlow ".split("\t")
pop_header.append("Label")
pop_header = [item.strip() for item in pop_header]

def _predict(account_list: List, result_path: str, data_path: str, preprocess_method: Callable[[pd.DataFrame], pd.DataFrame], header: List):
    file = pd.concat([pd.read_csv(os.path.join(data_path, account), encoding="utf-8", header=None, sep=";", names=header, index_col=False) for account in account_list])
    file.reset_index(inplace=True)

    ## 获取数值化特征
    df = preprocess_method(file)

    ### 读取数据
    train_index = df.loc[df['Label']==1].index
    test_index = df.loc[df['Label']==0].index

    d = df.drop(columns=["Label"]).iloc[train_index]
    b = df.drop(columns=["Label"]).iloc[test_index]
    # 进行标准化
    ss_X = StandardScaler()
    X_labeled = ss_X.fit_transform(d)
    # 提取所有正样本，作为测试集的一部分
    X_fraud  = ss_X.transform(b)

    # 提取负样本，并且按照8:2切成训练集和测试集
    X_train, X_test = train_test_split(X_labeled, test_size=0.2, random_state=1000)

    # 设置Autoencoder的参数
    # 隐藏层节点数分别为16，8，8，16
    # epoch为50，batch size为32
    input_dim = X_train.shape[1]
    encoding_dim = 16
    num_epoch = 50
    batch_size = 32

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh",
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['mae'])

    # 模型保存为SofaSofa_model.h5，并开始训练模型
    checkpointer = ModelCheckpoint(filepath="model.h1",
                                verbose=0,
                                save_best_only=True)
    history = autoencoder.fit(X_train, X_train,
                            epochs=num_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(X_test, X_test),
                            verbose=1,
                            callbacks=[checkpointer]).history

    # 读取模型
    autoencoder = load_model('model.h1')

    # 利用训练好的autoencoder重建测试集
    pred_test = autoencoder.predict(X_test)
    pred_fraud = autoencoder.predict(X_fraud)
    pred_labeled = autoencoder.predict(X_labeled)

    # 计算还原误差MSE和MAE
    mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)
    mse_fraud = np.mean(np.power(X_fraud - pred_fraud, 2), axis=1)
    mse_labeled = np.mean(np.power(X_labeled - pred_labeled, 2), axis=1)
    mae_test = np.mean(np.abs(X_test - pred_test), axis=1)
    mae_fraud = np.mean(np.abs(X_fraud - pred_fraud), axis=1)
    mae_labeled = np.mean(np.abs(X_labeled - pred_labeled), axis=1)
    
    # KDE数据预处理
    # 去除nan
    mae_fraud_filter = mae_fraud[~np.isnan(mae_fraud)]

    # 归一化：为了不改变数据分布，使用(0, 1)标准化
    lb = mae_fraud_filter[np.argmin(mae_fraud_filter)]
    ub = mae_fraud_filter[np.argmax(mae_fraud_filter)]

    mae_fraud_minmax = (mae_fraud_filter - lb) / (ub - lb)

    # 找到所有波谷
    a = mae_fraud_minmax.reshape(-1, 1)
    kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.05).fit(a)

    s = np.linspace(0, 1)
    e = kde.score_samples(s.reshape(-1, 1))

    mi = argrelextrema(e, np.less)[0]
    lims = s[mi]

    # 将波谷放大回原来的区间
    lims = lims * (ub - lb) + lb

    # 设置critical values
    med_fraud = np.median(mae_fraud)          #fraud的中位数 --> 为了避免筛选后的数据太多
    per_test = np.percentile(mae_test, 75)    #test的75%分位点 --> 为了避免筛选后的数据太少

    # 根据critical values筛选阈值
    lim = lims[np.where(lims < med_fraud)]
    lim = lim[np.where(lim > per_test)]

    threshold = per_test
    # 输出阈值
    if lim.size > 0:
        threshold = lim.min()
        print('根据波谷筛选出的阈值', lim.min())
    else:
        threshold = per_test
        if per_test > med_fraud:
            result_path = result_path[:-4] + ".need_check.csv"
            print('没有找到阈值:没有找到波谷，且负样本75%分位点大于中位数')
        else:
            result_path = result_path[:-4] + ".75.csv"
            print('没有找到波谷，根据负样本75%分位点筛选出的阈值', per_test)

    # 根据阈值打标签
    label = []
    for i in range(len(X_fraud)):
        if mae_fraud[i]>threshold:
            label.append(0)
        else:
            label.append(1)
            
    file.loc[test_index,"predict"] = label
    file.loc[test_index,"mse"] = mse_fraud
    file.loc[test_index,"mae"] = mae_fraud
    file.loc[train_index,"mse"] = mse_labeled
    file.loc[train_index,"mae"] = mae_labeled

    file.to_csv(result_path, index=False,encoding="utf-8")

def _mta_preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df['abroad'] = raw_data['abroad'].apply(lambda x:1 if x else 0)

    # 日期分为月 日
    df['MM'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[1]))
    df['dd'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[2]))

    # 时间分为时 分 秒
    df['hh'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[0]))
    df['mm'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[1]))
    df['ss'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[2]))

    # IP分四段
    df["ip1"] = raw_data['IP'].apply(lambda x:int(x.split(".")[0]))
    df["ip2"] = raw_data['IP'].apply(lambda x:int(x.split(".")[1]))
    df["ip3"] = raw_data['IP'].apply(lambda x:int(x.split(".")[2]))
    df["ip4"] = raw_data['IP'].apply(lambda x:int(x.split(".")[3]))


    for name, col in raw_data.iteritems():
        if name in ["abroad", "yyyyMMdd", "hhmmss", "IP"]:
            continue
        if col.isnull().all(): # 全是空的列赋0
            df[name] = 0
        elif col.dtype == object:  # 将每个文件中的列特征值转换成数字
            if name in ["rcpt", "errinfo", "province", "city", "operator", "netAffiliation", "institute", "account", "country", "SenderEmail", "fromDN"]:
                enc = preprocessing.LabelEncoder()
                df[name] = enc.fit_transform(col.apply(str))
        else:
            df[name] = col
    
    # nan值补为0
    df.fillna(0, inplace=True)
    return df

def _imap_preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df['abroad'] = raw_data['abroad'].apply(lambda x:1 if x else 0)

    # 日期分为月 日
    df['MM'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[1]))
    df['dd'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[2]))

    # 时间分为时 分 秒
    df['hh'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[0]))
    df['mm'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[1]))
    df['ss'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[2]))

    # IP分四段
    df["ip1"] = raw_data['IP'].apply(lambda x:int(x.split(".")[0]))
    df["ip2"] = raw_data['IP'].apply(lambda x:int(x.split(".")[1]))
    df["ip3"] = raw_data['IP'].apply(lambda x:int(x.split(".")[2]))
    df["ip4"] = raw_data['IP'].apply(lambda x:int(x.split(".")[3]))


    for name, col in raw_data.iteritems():
        if name in ["abroad", "yyyyMMdd", "hhmmss", "IP"]:
            continue
        if col.isnull().all(): # 全是空的列忽略
            continue
        elif col.dtype == object:  # 将每个文件中的列特征值转换成数字
            if name in ["province", "city", "operator", "netAffiliation", "institute", "account", "country", "keywordBehind"]:
                enc = preprocessing.LabelEncoder()
                df[name] = enc.fit_transform(col.apply(str))
        else:
            df[name] = col
    
    # nan值补为0
    df.fillna(0, inplace=True)
    return df

def _pop_preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df = df.dropna(axis='index', how='all', subset=[6,19])  #删除IP和Q不存在的行
    df['abroad'] = raw_data['abroad'].apply(lambda x:1 if x else 0)

    # 日期分为月 日
    df['MM'] = raw_data['create_Date'].apply(lambda x:int(x.split("-")[1]))
    df['dd'] = raw_data['create_Date'].apply(lambda x:int(x.split("-")[2]))

    # 时间分为时 分 秒
    df['hh'] = raw_data['time'].apply(lambda x:int(x.split(":")[0]))
    df['mm'] = raw_data['time'].apply(lambda x:int(x.split(":")[1]))
    df['ss'] = raw_data['time'].apply(lambda x:int(x.split(":")[2]))

    # IP分四段
    raw_data["IP"].fillna("0.0.0.0", inplace=True)
    df["ip1"] = raw_data['IP'].apply(lambda x:int(x.split(".")[0]))
    df["ip2"] = raw_data['IP'].apply(lambda x:int(x.split(".")[1]))
    df["ip3"] = raw_data['IP'].apply(lambda x:int(x.split(".")[2]))
    df["ip4"] = raw_data['IP'].apply(lambda x:int(x.split(".")[3]))


    for name, col in raw_data.iteritems():
        if name in ["abroad", "create_Date", "time", "IP"]:
            continue
        if col.isnull().all(): # 全是空的列忽略
            continue
        elif col.dtype == object:  # 将每个文件中的列特征值转换成数字
            if name in ["province", "city", "operator", "netAffiliation", "institute", "account", "country", "keywords", "mBoxID", "mid", "currentFlow"]:
                enc = preprocessing.LabelEncoder()
                df[name] = enc.fit_transform(col.apply(str))
        else:
            df[name] = col
    
    # nan值补为0
    df.fillna(0, inplace=True)
    return df

def mta_predict(account_list, result_path, data_path):
    _predict(account_list=account_list,
             result_path=result_path,
             data_path=data_path,
             preprocess_method=_mta_preprocess,
             header=mta_header)
    
def imap_predict(account_list, result_path, data_path):
   _predict(account_list=account_list,
             result_path=result_path,
             data_path=data_path,
             preprocess_method=_imap_preprocess,
             header=imap_header)
    
def pop_predict(account_list, result_path, data_path):
    
    _predict(account_list=account_list,
             result_path=result_path,
             data_path=data_path,
             preprocess_method=_pop_preprocess,
             header=pop_header)
    
    
if __name__ == "__main__":
    
    account_list = ['chaifj@cnic.cn.csv', 'csdata@cnic.cn.csv', 'cy@cnic.cn.csv']
    mta_predict(
        account_list=account_list,
        result_path="./result.csv"
    )
