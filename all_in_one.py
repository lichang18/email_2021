
from typing import Callable, List
from sklearn import preprocessing
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
import os
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity

web_header ="abroad	behavior	yyyyMMdd	hhmmss	account	domainName	IP	country	province	city	operator	netAffiliation	institute	latitude	longitude	result	osLanguage	os	friendlyName	locale	timestamp	dateTime	requestURL	opTime	Q	fid	fromAddress	flag	read	attachId	keywords	requestVar	voice	pref	fn	group	id	part	requestReadReceipt	bcc	cc	to	from	size	ids	transmitted	mode	usedCapacityKB	mid	msid	deleted	name	type	returnInfo	saveSentCopy	content	isHtml	subject	action	value	length	lable	rev	respTime	target	field	operand	id5	id4	id3	id2	id1	unreadMessageSize	unreadMessageCount	messageSize	messageCount	resultVar	creationDate	resultOthers	cnt	status	email	imapFolder	sentTInfo	ips	uuid	folderName	fileName	shareLink	Label".split("\t")
web_header = [item.strip() for item in web_header]

mta_header = "abroad	behavior	yyyyMMdd	hhmmss	account	domainName	IP	country	province	city	operator	netAffiliation	institute	latitude	longitude	ssl	rcpt	state	size	keywords	Q	Tid	fromDN	AuthFailCnt	optimes	errinfo	origip	Xmailer	SenderEmail	Local	org_unit_id	lrcptcnt	rrcptcnt	LmtpRcptCnt	DataRuleID	Score	AttachCnt	DataFngCnt	DataSFngCnt	DKimVerifyResult	Subject	SubjectCnt	Handle	Content	ttime	UsrQuarterCnt	UsrTodayCnt	conn	region	quarterCnt	todayCnt	todayRcptCnt	hangup	errStat	delivered	userSendInterval	helo	mdRet	errCnt	deliveredCnt	domainQuarterCnt	domainTodayCnt	MailCnt".split("\t")
mta_header.append("Label")
mta_header = [item.strip() for item in mta_header]

imap_header = "abroad	 behavior	 yyyyMMdd	 hhmmss	 account	 domainName	 IP	 country	 province	 city	 operator	 netAffiliation	 institute	 latitude	 longitude	 keywordBehind	 root	 udid	 mid	 set	 removeFlag	 size	 msid	 xmailer	 rcpt	 MIMEVersion	 fid".split("\t")
imap_header.append("Label")
imap_header = [item.strip() for item in imap_header]

pop_header = "abroad	behavior	create_Date	time	account	domainName	IP	country	province	city	operator	netAffiliation	institute	dynamicAttr	dynamicNum	keywords	mBoxID	MSID	mid	processQ 	num 	data 	reason 	udid 	msgcount 	msgsize 	currentFlow ".split("\t")
pop_header.append("Label")
pop_header = [item.strip() for item in pop_header]

def _predict(account_list: List, result_path: str, data_path: str, preprocess_method: Callable[[pd.DataFrame], pd.DataFrame], header: List, model_name: str):
    file = pd.concat([pd.read_csv(os.path.join(data_path, account), encoding="utf-8", header=None, sep=";", names=header, index_col=False) for account in account_list])
    file.reset_index(inplace=True)

    ## ?????????????????????
    df = preprocess_method(file)

    ### ????????????
    train_index = df.loc[df['Label']==1].index
    test_index = df.loc[df['Label']==0].index

    d = df.drop(columns=["Label"]).iloc[train_index]
    b = df.drop(columns=["Label"]).iloc[test_index]
    # ???????????????
    ss_X = StandardScaler()
    X_labeled = ss_X.fit_transform(d)
    # ???????????????????????????????????????????????????
    X_fraud  = ss_X.transform(b)

    # ??????????????????????????????8:2???????????????????????????
    X_train, X_test = train_test_split(X_labeled, test_size=0.2, random_state=1000)

    # ??????Autoencoder?????????
    # ???????????????????????????16???8???8???16
    # epoch???50???batch size???32
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

    # ???????????????SofaSofa_model.h5????????????????????????
    checkpointer = ModelCheckpoint(filepath=model_name,
                                verbose=0,
                                save_best_only=True)
    history = autoencoder.fit(X_train, X_train,
                            epochs=num_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(X_test, X_test),
                            verbose=1,
                            callbacks=[checkpointer]).history

    # ????????????
    autoencoder = load_model(model_name)

    # ??????????????????autoencoder???????????????
    pred_test = autoencoder.predict(X_test)
    pred_fraud = autoencoder.predict(X_fraud)
    pred_labeled = autoencoder.predict(X_labeled)

    # ??????????????????MSE???MAE
    mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)
    mse_fraud = np.mean(np.power(X_fraud - pred_fraud, 2), axis=1)
    mse_labeled = np.mean(np.power(X_labeled - pred_labeled, 2), axis=1)
    mae_test = np.mean(np.abs(X_test - pred_test), axis=1)
    mae_fraud = np.mean(np.abs(X_fraud - pred_fraud), axis=1)
    mae_labeled = np.mean(np.abs(X_labeled - pred_labeled), axis=1)
    
    # KDE???????????????
    # ??????nan
    mae_fraud_filter = mae_fraud[~np.isnan(mae_fraud)]

    # ????????????????????????????????????????????????(0, 1)?????????
    lb = mae_fraud_filter[np.argmin(mae_fraud_filter)]
    ub = mae_fraud_filter[np.argmax(mae_fraud_filter)]

    mae_fraud_minmax = (mae_fraud_filter - lb) / (ub - lb)

    # ??????????????????
    a = mae_fraud_minmax.reshape(-1, 1)
    kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.05).fit(a)

    s = np.linspace(0, 1)
    e = kde.score_samples(s.reshape(-1, 1))

    mi = argrelextrema(e, np.less)[0]
    lims = s[mi]

    # ?????????????????????????????????
    lims = lims * (ub - lb) + lb

    # ??????critical values
    med_fraud = np.median(mae_fraud)          #fraud???????????? --> ????????????????????????????????????
    per_test = np.percentile(mae_test, 75)    #test???75%????????? --> ????????????????????????????????????

    # ??????critical values????????????
    lim = lims[np.where(lims < med_fraud)]
    lim = lim[np.where(lim > per_test)]

    threshold = per_test
    # ????????????
    if lim.size > 0:
        threshold = lim.min()
        print('??????????????????????????????', lim.min())
    else:
        threshold = per_test
        if per_test > med_fraud:
            result_path = result_path[:-4] + ".need_check.csv"
            print('??????????????????:?????????????????????????????????75%????????????????????????')
        else:
            result_path = result_path[:-4] + ".75.csv"
            print('????????????????????????????????????75%???????????????????????????', per_test)

    # ?????????????????????
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

def _web_preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df['abroad'] = raw_data['abroad'].apply(lambda x:1 if x else 0)

    # ??????????????? ???
    df['MM'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[1]))
    df['dd'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[2]))

    # ??????????????? ??? ???
    df['hh'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[0]))
    df['mm'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[1]))
    df['ss'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[2]))

    # IP?????????
    df["ip1"] = raw_data['IP'].apply(lambda x:int(x.split(".")[0]))
    df["ip2"] = raw_data['IP'].apply(lambda x:int(x.split(".")[1]))
    df["ip3"] = raw_data['IP'].apply(lambda x:int(x.split(".")[2]))
    df["ip4"] = raw_data['IP'].apply(lambda x:int(x.split(".")[3]))


    for name, col in raw_data.iteritems():
        if name in ["abroad", "yyyyMMdd", "hhmmss", "IP","domainName", "result", "locale", "timestamp", "dateTime", "requestVar", "voice", "pref", "ids", "rev", "creationDate"]:
            continue  #???????????????????????????
        if col.isnull().all(): # ??????????????????0
            df[name] = 0
        if name in ["fid","flag","read","fn","group","name","type","saveSentCopy","content","subject","value","length","lable","operand","id5","id4","id3","id2","id1","resultVar","resultOthers","cnt","status","email","imapFolder","sentTInfo","ips","uuid","folderName","fileName","shareLink"]:
            df[name] = [0 if item else 1 for item in col.isnull()]  #?????????????????????01????????????????????????1???nan????????????0
        elif col.dtype == object:  # ????????????????????????????????????????????????
            if name in ["account","country", "province", "city", "operator", "netAffiliation", "institute", "osLanguage", "os", "friendlyName", "requestURL","fromAddress","attachId","keywords","id","requestReadReceipt","bcc","cc","to","from","mode","mid","msid","deleted","returnInfo","isHtml","action","target","field"]:
                enc = preprocessing.LabelEncoder()
                df[name] = enc.fit_transform(col.apply(str))
        else:
            df[name] = col

    # nan?????????0
    df.fillna(0, inplace=True)
    return df



def _mta_preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df['abroad'] = raw_data['abroad'].apply(lambda x:1 if x else 0)

    # ??????????????? ???
    df['MM'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[1]))
    df['dd'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[2]))

    # ??????????????? ??? ???
    df['hh'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[0]))
    df['mm'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[1]))
    df['ss'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[2]))

    # IP?????????
    df["ip1"] = raw_data['IP'].apply(lambda x:int(x.split(".")[0]))
    df["ip2"] = raw_data['IP'].apply(lambda x:int(x.split(".")[1]))
    df["ip3"] = raw_data['IP'].apply(lambda x:int(x.split(".")[2]))
    df["ip4"] = raw_data['IP'].apply(lambda x:int(x.split(".")[3]))


    for name, col in raw_data.iteritems():
        if name in ["abroad", "yyyyMMdd", "hhmmss", "IP"]:
            continue
        if col.isnull().all(): # ??????????????????0
            df[name] = 0
        elif col.dtype == object:  # ????????????????????????????????????????????????
            if name in ["rcpt", "errinfo", "province", "city", "operator", "netAffiliation", "institute", "account", "country", "SenderEmail", "fromDN"]:
                enc = preprocessing.LabelEncoder()
                df[name] = enc.fit_transform(col.apply(str))
        else:
            df[name] = col
    
    # nan?????????0
    df.fillna(0, inplace=True)
    return df

def _imap_preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df['abroad'] = raw_data['abroad'].apply(lambda x:1 if x else 0)

    # ??????????????? ???
    df['MM'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[1]))
    df['dd'] = raw_data['yyyyMMdd'].apply(lambda x:int(x.split("-")[2]))

    # ??????????????? ??? ???
    df['hh'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[0]))
    df['mm'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[1]))
    df['ss'] = raw_data['hhmmss'].apply(lambda x:int(x.split(":")[2]))

    # IP?????????
    df["ip1"] = raw_data['IP'].apply(lambda x:int(x.split(".")[0]))
    df["ip2"] = raw_data['IP'].apply(lambda x:int(x.split(".")[1]))
    df["ip3"] = raw_data['IP'].apply(lambda x:int(x.split(".")[2]))
    df["ip4"] = raw_data['IP'].apply(lambda x:int(x.split(".")[3]))


    for name, col in raw_data.iteritems():
        if name in ["abroad", "yyyyMMdd", "hhmmss", "IP"]:
            continue
        if col.isnull().all(): # ?????????????????????
            continue
        elif col.dtype == object:  # ????????????????????????????????????????????????
            if name in ["province", "city", "operator", "netAffiliation", "institute", "account", "country", "keywordBehind"]:
                enc = preprocessing.LabelEncoder()
                df[name] = enc.fit_transform(col.apply(str))
        else:
            df[name] = col
    
    # nan?????????0
    df.fillna(0, inplace=True)
    return df

def _pop_preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    #??????IP???Q???????????????
    raw_data.dropna(axis=0, how="any", subset=["IP", "processQ"], inplace=True)
    raw_data.reset_index(drop=True, inplace=True)
    df = pd.DataFrame()
    
    df['abroad'] = raw_data['abroad'].apply(lambda x:1 if x else 0)

    # ??????????????? ???
    df['MM'] = raw_data['create_Date'].apply(lambda x:int(x.split("-")[1]))
    df['dd'] = raw_data['create_Date'].apply(lambda x:int(x.split("-")[2]))

    # ??????????????? ??? ???
    df['hh'] = raw_data['time'].apply(lambda x:int(x.split(":")[0]))
    df['mm'] = raw_data['time'].apply(lambda x:int(x.split(":")[1]))
    df['ss'] = raw_data['time'].apply(lambda x:int(x.split(":")[2]))

    # IP?????????
    raw_data["IP"].fillna("0.0.0.0", inplace=True)
    df["ip1"] = raw_data['IP'].apply(lambda x:int(x.split(".")[0]))
    df["ip2"] = raw_data['IP'].apply(lambda x:int(x.split(".")[1]))
    df["ip3"] = raw_data['IP'].apply(lambda x:int(x.split(".")[2]))
    df["ip4"] = raw_data['IP'].apply(lambda x:int(x.split(".")[3]))


    for name, col in raw_data.iteritems():
        if name in ["abroad", "create_Date", "time", "IP"]:
            continue
        if col.isnull().all(): # ?????????????????????
            continue
        elif col.dtype == object:  # ????????????????????????????????????????????????
            if name in ["province", "city", "operator", "netAffiliation", "institute", "account", "country", "keywords", "mBoxID", "mid", "currentFlow"]:
                enc = preprocessing.LabelEncoder()
                df[name] = enc.fit_transform(col.apply(str))
        else:
            df[name] = col
    
    # nan?????????0
    df.fillna(0, inplace=True)
    return df


def web_predict(account_list, result_path, data_path):
    _predict(account_list=account_list,
             result_path=result_path,
             data_path=data_path,
             preprocess_method=_web_preprocess,
             header=web_header,
             model_name="web.h1")


def mta_predict(account_list, result_path, data_path):
    _predict(account_list=account_list,
             result_path=result_path,
             data_path=data_path,
             preprocess_method=_mta_preprocess,
             header=mta_header,
             model_name="mta.h1")
    
def imap_predict(account_list, result_path, data_path):
   _predict(account_list=account_list,
             result_path=result_path,
             data_path=data_path,
             preprocess_method=_imap_preprocess,
             header=imap_header,
             model_name="imap.h1")
    
def pop_predict(account_list, result_path, data_path):
    
    _predict(account_list=account_list,
             result_path=result_path,
             data_path=data_path,
             preprocess_method=_pop_preprocess,
             header=pop_header,
             model_name="pop.h1")
    
    
if __name__ == "__main__":
    
    account_list = ['laisuomei2020@sibcb.ac.cn_2.csv', 'leiyb@itpcas.ac.cn_2.csv', 'maliang@ioz.ac.cn_2.csv']
    web_predict(
        account_list=account_list,
        result_path=r"C:\Users\lichangUCAS\Desktop\?????????",
        data_path=r"E:\sectry\web")
