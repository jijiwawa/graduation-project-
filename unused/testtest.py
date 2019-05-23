import numpy as np
import re
import os
# 将亚马逊数据转换成数字
def Transform_amzon_To_digital(file):
    user_name = []
    item_name = []
    num_user = 0
    num_item = 0
    with open(file, "r") as f:
        line = f.readline()
        while line != None and line != "":
            pattern = r'[,|\s|:]+'
            arr = re.split(pattern, line)
            userId, itemId, rating = str(arr[0]), str(arr[1]), str(arr[2])
            if userId not in user_name:
                user_name.append(userId)
                num_user+=1
                print('user:',num_user)
            if itemId not in item_name:
                item_name.append(itemId)
                num_item+=1
                print('item:',num_item)
            line = f.readline()
            if num_user >500 and num_item>1000:
                break
    return user_name,item_name

if __name__ == '__main__':
    user_list,item_list = Transform_amzon_To_digital(os.getcwd()+'\\prepare_datasets\\ratings_Amazon_Instant_Video.csv')
    print(user_list)
    print(len(user_list))
    print(len(item_list))

    f1 = open(os.getcwd()+'\\prepare_datasets\\ratings_Amazon_Instant_Video.csv', 'r+')
    f2 = open(os.getcwd()+'\\prepare_datasets\\Amazon_Instant_Video_new.csv', 'w+').read()

    user_id,item_id = 0,0
    for name in user_list:
        f2=f2.replace(name,str(user_id))
        user_id+=1
    for name_item in item_list:
        f2 = f2.replace(name_item, str(item_id))
        item_id += 1
    f1.close()
    f2.close()