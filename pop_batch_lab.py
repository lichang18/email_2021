from all_in_one import pop_predict, pop_header
from classify_user import pop_classify
import os
import pandas as pd

data_path = "./data/pop"
result_path = "./result/pop_with_cls_thd"
if not os.path.exists(result_path):
    os.mkdir(result_path)
    
list1, list2, list3, list4 = pop_classify(filepath=data_path)

def sort_file(file_list):
    files = dict()
    for file in file_list:
        df = pd.read_csv(os.path.join(data_path, file), encoding="utf-8", header=None, sep=";", names=pop_header, index_col=False)
        files[file] = len(df.loc[df["Label"]==1])
        
    sorted_files = sorted(files.items(), key=lambda d:d[1])
    return sorted_files

def run_detector(sorted_file_list: list, list_name: str):
    write_path = os.path.join(result_path, list_name)
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    i = 0
    while len(sorted_file_list) > 0:
        account_list = sorted_file_list[0:3]
        sorted_file_list = sorted_file_list[3:]
        if len(sorted_file_list) > 0:
            account_list += sorted_file_list[-3:]
            sorted_file_list = sorted_file_list[:-3]
        print("{}.csv".format(i))
        pop_predict(account_list=account_list, result_path=os.path.join(write_path, "{}.csv".format(i)), data_path=data_path)
        
        i += 1
        
run_detector([item[0] for item in sort_file(list1)], "list1")

run_detector([item[0] for item in sort_file(list2)], "list2")

run_detector([item[0] for item in sort_file(list3)], "list3")

run_detector([item[0] for item in sort_file(list4)], "list4")