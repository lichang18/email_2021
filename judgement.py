# 判断模型预测结果，以及提取其中的恶意ip

import os
from typing import List, Set
import pandas as pd


def judge_imap(data_path: str, fail_behavior: List = ['2']) -> Set:
    bad_ip = set()
    # for list1
    print("list1...")
    for file in os.listdir(os.path.join(data_path, "list1")):
        if file.endswith(".csv"):

            file_path = os.path.join(os.path.join(data_path, "list1"), file)
            df = pd.read_csv(file_path,index_col=None,header=0,encoding="utf-8")
            df["success"] = df["behavior"].apply(lambda x: x not in fail_behavior)

            for ip in df.loc[(df["success"] == False) & (df["predict"] == 1), "IP"].unique().tolist():
                if len(ip) > 4:
                    bad_ip.add(ip)
            for ip in df.loc[(df["abroad"] == True) & (df["predict"] == 1), "IP"].unique().tolist():
                if len(ip) > 4:
                    bad_ip.add(ip)

            mp = dict()
            for n, g in df.loc[df["success"] == True].groupby("account"):
                op = set(["教育网", "科技网"])
                op.add(g["operator"].value_counts().idxmax())
                mp[n] = {
                    "city": g["city"].value_counts().idxmax(),
                    "operator": op
                }
            
            df["judge"] = df["predict"]
            for i, line in df.iterrows():
                if line["predict"] == 0:
                    continue
                if line["success"] == False:
                    continue
                if line["abroad"] == True:
                    continue
                if line["city"] == mp[line["account"]]["city"]:
                    df.loc[i, "judge"] = 0
                    continue
                if line["operator"] in mp[line["account"]]["operator"]:
                    df.loc[i, "judge"] = 0
                    continue
                df.loc[i, "judge"] = 2
            del df["success"]
            
            df.to_csv(file_path, index=False,encoding="utf-8")
            del df
            
    # list2
    print("list2...")
    for file in os.listdir(os.path.join(data_path, "list2")):
        if file.endswith(".csv"):
            file_path = os.path.join(os.path.join(data_path, "list2"), file)
            df = pd.read_csv(file_path,index_col=None,header=0,encoding="utf-8")
            df["success"] = df["behavior"].apply(lambda x: x not in fail_behavior)

            for ip in df.loc[(df["success"] == False) & (df["predict"] == 1), "IP"].unique().tolist():
                if len(ip) > 4:
                    bad_ip.add(ip)
            for ip in df.loc[(df["abroad"] == True) & (df["predict"] == 1), "IP"].unique().tolist():
                if len(ip) > 4:
                    bad_ip.add(ip)

            mp = dict()
            for n, g in df.loc[df["success"] == True].groupby("account"):
                op = set(["教育网", "科技网"])
                op.add(g["operator"].value_counts().idxmax())
                mp[n] = {
                    "city": g["city"].value_counts().idxmax(),
                    "operator": op
                }
            
            df["judge"] = df["predict"]
            for i, line in df.iterrows():
                if line["predict"] == 0:
                    continue
                if line["success"] == False:
                    continue
                if line["abroad"] == True:
                    continue
                if line["city"] == mp[line["account"]]["city"]:
                    df.loc[i, "judge"] = 0
                    continue
                if line["operator"] in mp[line["account"]]["operator"]:
                    df.loc[i, "judge"] = 0
                    continue
                df.loc[i, "judge"] = 2
            del df["success"]
            
            df.to_csv(file_path, index=False,encoding="utf-8")
            del df
    
    # list3
    print("list3...")
    for file in os.listdir(os.path.join(data_path, "list3")):
        if file.endswith(".csv"):
            file_path = os.path.join(os.path.join(data_path, "list3"), file)
            df = pd.read_csv(file_path,index_col=None,header=0,encoding="utf-8")
            df["success"] = df["behavior"].apply(lambda x: x not in fail_behavior)

            for ip in df.loc[(df["success"] == False) & (df["predict"] == 1), "IP"].unique().tolist():
                if len(ip) > 4:
                    bad_ip.add(ip)
            df["judge"] = df["predict"]
            for i, line in df.iterrows():
                if line["predict"] == 0:
                    continue
                if line["success"] == False:
                    continue
                df.loc[i, "judge"] = 2
            del df["success"]
            
            df.to_csv(file_path, index=False,encoding="utf-8")
            del df
    
    # list4
    print("list4...")
    for file in os.listdir(os.path.join(data_path, "list4")):
        if file.endswith(".csv"):
            file_path = os.path.join(os.path.join(data_path, "list4"), file)
            df = pd.read_csv(file_path,index_col=None,header=0,encoding="utf-8")

            for ip in df.loc[df["predict"] == 1, "IP"].unique().tolist():
                if len(ip) > 4:
                    bad_ip.add(ip)
            
    return bad_ip

def judge_statistics(data_path: str):
    dirs = ["list1", "list2", "list3"]
    res = []
    for dir in dirs:
        for file in os.listdir(os.path.join(data_path, dir)):
            if file.endswith(".csv"):
                file_path = os.path.join(os.path.join(data_path, dir), file)
                df = pd.read_csv(file_path,index_col=None,header=0,encoding="utf-8")
                res.append({
                    "name": file_path,
                    "1-1": len(df.loc[(df["predict"] == 1) & (df["judge"] == 1)]),
                    "1-0": len(df.loc[(df["predict"] == 1) & (df["judge"] == 0)]),
                    "sum": len(df),
                    "pred1": len(df.loc[df["predict"] == 1]),
                    "label1": len(df.loc[df["Label"] == 1]),
                })
    return pd.DataFrame.from_records(res)
            

if __name__ == "__main__":
    
    data_path = "../../result/imap_with_cls_thd"
    
    fail_behaavior = ['2']
    
    mali_ip_set = judge_imap(data_path=data_path)
    
    with open("../../ip/imap0.txt", "w", encoding="utf-8") as f:
        for item in mali_ip_set:
            f.write(item + "\n")
    
    judge_statistics(data_path).to_csv("../../result/some_tmp/imap.csv", index=False)