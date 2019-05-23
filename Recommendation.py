# -*- coding: utf-8 -*-
import scipy.sparse as sp
import numpy as np
import math
import time
import re
import os
import random


class Recommendation():

    # 每个file 均为路径
    def __init__(self, file, train_file, test_file):
        # 获取用户数和物品数量
        self.num_users, self.num_items, self.num_rating = self.Count_Num_Of_UserAndItem(file)
        print('用户数：', self.num_users)
        print('物品数：', self.num_items)
        print('评分数：', self.num_rating)
        # 构建训练评分矩阵
        self.trainMatrix = self.Transform_csv_To_RatingMatrix(train_file)
        # 构建测试评分矩阵用于后期评估运算
        self.testMatrix = self.Transform_csv_To_RatingMatrix(test_file)
        # 评分矩阵参数
        self.ratingMax, self.ratingMim, self.num_scale = 5, 1, 5
        # 生成每个用户评分了的物品集合 {user:[item1,item2]}
        self.coItemDict = self.Generate_ratingItemDict_ForEachUser(self.trainMatrix)
        print('用户评分了的物品集合：')
        print(self.coItemDict)

        # 每个用户评分均值
        # 生成每个用户的评分列表 {用户：{1.0，3.0}，用户2：{2.0.4.0}}
        self.user_rating_dict = self.Generate_User_Rating_Dict(train_file)
        print(self.user_rating_dict)

        # 生成用户平均评分
        self.user_ave_rating_dict = self.Get_Average_UserRating(self.user_rating_dict)
        print('用户平均评分集合：')
        print(self.user_ave_rating_dict)

    # 统计总样本中用户数和物品数
    def Count_Num_Of_UserAndItem(self, ratedfile):
        num_users, num_items = 0, 0
        count = 0
        with open(ratedfile, "r") as f:
            line = f.readline()
            while line != None and line != "":
                # [\s,:]
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                userId, itemId = int(float(arr[0])), int(float(arr[1]))
                num_users = max(num_users, userId)
                num_items = max(num_items, itemId)
                count += 1
                line = f.readline()
        return num_users + 1, num_items + 1, count

    def Transform_list_To_RatingMatrix(self, train_list):
        Matrix = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        for line in train_list:
            userId, itemId, rating = int(float(line[0])), int(float(line[1])), int(float(line[2]))
            Matrix[userId, itemId] = rating
        return Matrix

    # 将训练样本转换成评分矩阵
    def Transform_csv_To_RatingMatrix(self, file):
        ratingMatrix = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                ratingMatrix[userId, itemId] = rating
                line = f.readline()
        return ratingMatrix

    # 分割数据集为测试集和训练集并写入train 和 test文件
    def SplitData(self, file, M, k, seed):
        train, test = [], []
        random.seed(seed)
        # 以列表的形式存
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                if random.randint(0, M) == k:
                    test.append([arr[0], arr[1], arr[2]])
                else:
                    train.append([arr[0], arr[1], arr[2]])
                line = f.readline()
        return train, test

    # preditmatrix 是一个array[][]
    # testmatrix 是一个dokmatirx
    # 从评分矩阵中生成用户的评分了的物品列表 用户/物品
    def Generate_ratingItemDict_ForEachUser(self, trainMatrix):
        ratingItemList = dict()
        for userid in range(self.num_users):
            ratingItemList[userid] = set()
        for (userid, itemid) in trainMatrix.keys():
            ratingItemList[userid].add(itemid)
        return ratingItemList

    # 从csv中生成物品的评分字典  {物品id:{评分}}
    def Generate_ratingDict_ForEachItem(self, file):
        itemRatingDict = dict()
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                if itemId not in itemRatingDict.keys():
                    itemRatingDict[itemId] = []
                itemRatingDict[itemId].append(rating)
                line = f.readline()
        return itemRatingDict

    # 从评分矩阵中生成用户的评分了的物品列表 用户/物品
    def Generate_ratingItemDict_ForEachUser(self, trainMatrix):
        ratingItemList = dict()
        for (userid, itemid) in trainMatrix.keys():
            if userid not in ratingItemList.keys():
                ratingItemList[userid] = set()
            ratingItemList[userid].add(itemid)
        return ratingItemList

    # 从训练矩阵中得出
    def Generate_User_Rating_Dict(self, trainfile):
        user_rating_dict = dict()
        with open(trainfile, "r") as f:
            line = f.readline()
            while line != None and line != "":
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                if userId not in user_rating_dict.keys():
                    user_rating_dict[userId] = []
                user_rating_dict[userId].append(rating)
                line = f.readline()
        return user_rating_dict

    # user_rating_dict 求每个用户的平均评分
    def Get_Average_UserRating(self, user_rating_dict):
        user_ave_rating = dict()
        for userid in range(self.num_users):
            if userid in user_rating_dict.keys():
                user_ave_rating[userid] = np.sum(user_rating_dict[userid]) / len(user_rating_dict[userid])
            else:
                user_ave_rating[userid] = 2.5
        return user_ave_rating


def SplitData_To_TrainandTest(datafile, M, k, seed):
    random.seed(seed)
    # 以文件的形式存储
    trainfile_path = os.getcwd() + '\\prepare_datasets\\' + os.path.basename(datafile) + '_train.csv'
    if os.path.exists(trainfile_path):
        os.remove(trainfile_path)
    f_train = open(trainfile_path, 'a')
    testfile_path = os.getcwd() + '\\prepare_datasets\\' + os.path.basename(datafile) + '_test.csv'
    if os.path.exists(testfile_path):
        os.remove(testfile_path)
    f_test = open(testfile_path, 'a')
    with open(datafile, "r") as f:
        line = f.readline()
        while line != None and line != "":
            if random.randint(0, M) == k:
                f_test.write(line)
            else:
                f_train.write(line)
            line = f.readline()
    f_train.close()
    f_test.close()


# 选择数据集
def get_dataset_path(num_dataset):
    # test数据集合
    Hybird = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv'
    Hybird_train = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_train.csv'
    Hybird_test = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_test.csv'
    Hybird_ = [Hybird, Hybird_train, Hybird_test]
    # ml-100-400
    ml_100_400 = os.getcwd() + '\\prepare_datasets\\m1-100-400.csv'
    ml_100_400_train = os.getcwd() + '\\prepare_datasets\\m1-100-400.csv_train.csv'
    ml_100_400_test = os.getcwd() + '\\prepare_datasets\\m1-100-400.csv_test.csv'
    ml_100_400_ = [ml_100_400, ml_100_400_train, ml_100_400_test]
    # m1-100k
    ml_100k = os.getcwd() + '\\prepare_datasets\\m1-100k.csv'
    ml_100k_train = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_train.csv'
    ml_100k_test = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_test.csv'
    ml_100k_ = [ml_100k, ml_100k_train, ml_100k_test]
    # m1-1m
    ml_1m = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_train = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_test = os.getcwd() + '\\prepare_datasets\\ml-1m.test.rating'
    ml_1m_ = [ml_1m, ml_1m_train, ml_1m_test]
    # pcc_data
    pcc = os.getcwd() + '\\prepare_datasets\\PCC_traindata.csv'
    pcc_train = os.getcwd() + '\\prepare_datasets\\PCC_traindata.csv'
    pcc_test = os.getcwd() + '\\prepare_datasets\\PCC_traindata.csv'
    pcc_data = [pcc, pcc_train, pcc_test]
    # ml_200_1000
    ml_200_1000 = os.getcwd() + '\\prepare_datasets\\m1-200-1000.csv'
    ml_200_1000_train = os.getcwd() + '\\prepare_datasets\\m1-200-1000.csv_train.csv'
    ml_200_1000_test = os.getcwd() + '\\prepare_datasets\\m1-200-1000.csv_test.csv'
    ml_200_1000_ = [ml_200_1000, ml_200_1000_train, ml_200_1000_test]
    # test_99_400
    test_99_400 = os.getcwd() + '\\prepare_datasets\\test_99_400.base'
    test_99_400_train = os.getcwd() + '\\prepare_datasets\\test_99_400.base_train.csv'
    test_99_400_test = os.getcwd() + '\\prepare_datasets\\test_99_400.base_test.csv'
    test_99_400_ = [test_99_400, test_99_400_train, test_99_400_test]
    # ml_200_120
    ml_200_120 = os.getcwd() + '\\prepare_datasets\\m1-200-120.csv'
    ml_200_120_train = os.getcwd() + '\\prepare_datasets\\m1-200-120.csv_train.csv'
    ml_200_120_test = os.getcwd() + '\\prepare_datasets\\m1-200-120.csv_test.csv'
    ml_200_120_ = [ml_200_120, ml_200_120_train, ml_200_120_test]
    numbers = {
        0: Hybird_,
        1: ml_100_400_,
        2: ml_100k_,
        3: ml_1m_,
        4: pcc_data,
        5: ml_200_1000_,
        6: test_99_400_,
        7: ml_200_120_
    }
    return numbers.get(num_dataset)

def Generate_non_user_item(file):
    userlist,itemlist=[],[]
    with open(file, "r") as f:
        line = f.readline()
        while line != None and line != "":
            pattern = r'[,|\s|:]+'
            arr = re.split(pattern, line)
            userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
            if userId not in userlist:
                userlist.append(userId)
            if itemId not in itemlist:
                itemlist.append(itemId)
            line = f.readline()
    return userlist,itemlist

if __name__ == '__main__':
    s = 1
    # SplitData_To_TrainandTest(os.getcwd() + '\\prepare_datasets\\m1-200-120.csv', 10, 8, 1)
    userlist,itemlist = Generate_non_user_item(os.getcwd() + '\\prepare_datasets\\m1-200-120.csv_train.csv')
    print('user')
    for i in range(200):
        if i not in userlist:
            print(i)
    print('item')
    for i in range(120):
        if i not in itemlist:
            print(i)