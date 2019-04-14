# -*- coding: utf-8 -*-
import scipy.sparse as sp
import numpy as np
import math
import time
import re
import os
import random


class Recommendation():

    def __init__(self, file):
        # 获取用户数和物品数量
        self.num_users, self.num_items = self.Count_Num_Of_UserAndItem(file)
        # 构建训练评分矩阵
        self.trainMatrix = self.Transform_csv_To_RatingMatrix(
            os.getcwd() + '\\prepare_datasets\\' + os.path.basename(file) + '_train.csv')
        # 构建测试评分矩阵用于后期评估运算
        self.testMatrix = self.Transform_csv_To_RatingMatrix(
            os.getcwd() + '\\prepare_datasets\\' + os.path.basename(file) + '_test.csv')
        print('success Transform_csv_To_RatingMatrix!!!')
        self.ratingMax, self.ratingMim, self.num_scale = 5, 1, 5

    # 将csv转换称列表的形式
    def loadcsv_to_list(self, file):
        data = []
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                # user,item,rating=int(str.strip(arr[0])),int(str.strip(arr[1])),int(str.strip(arr[2]))
                user, item, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                data.append([user, item, rating])
                line = f.readline()
        return data

    # 统计总样本中用户数和物品数
    def Count_Num_Of_UserAndItem(self, ratedfile):
        num_users, num_items = 0, 0
        with open(ratedfile, "r") as f:
            line = f.readline()
            while line != None and line != "":
                # [\s,:]
                pattern = r'[,|\s|:]+'
                arr = re.split(pattern, line)
                userId, itemId = int(float(arr[0])), int(float(arr[1]))
                num_users = max(num_users, userId)
                num_items = max(num_items, itemId)
                line = f.readline()
        return num_users + 1, num_items + 1

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

    def Evaluate_MAE(self, preditmatrix, testmatrix):
        matrix_sub = sp.dok_matrix.copy(testmatrix)
        userid1 = -1
        m = 0
        n = np.zeros(self.num_items)
        for (userid, itemid) in testmatrix.keys():
            matrix_sub[userid, itemid] = math.fabs(matrix_sub[userid, itemid] - preditmatrix[userid][itemid])
            if userid != userid1:
                m += 1
            n[userid] += 1
            userid1 = userid
        sum = 0
        sum_each_row = np.sum(matrix_sub.toarray(), axis=1)
        for i in range(0, self.num_items):
            if n[i] != 0:
                sum += sum_each_row[i] / n[i]
        MAE = sum / m
        return MAE


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


if __name__ == '__main__':
    csv_path = os.getcwd()+ '\\prepare_datasets\\Hybird_data.csv'
    csv_path1 = 'E:\\0学业\\毕设\\useful_dataset\\m-100k\\m1-100k.csv'
    train_path = os.getcwd() + '\\prepare_datasets\\' + os.path.basename(csv_path)+'_train.csv'
    test_path = os.getcwd() + '\\prepare_datasets\\' + os.path.basename(csv_path)+'_test.csv'

    count_train = len(open(train_path, 'rU').readlines())
    count_test = len(open(test_path, 'rU').readlines())
    print(count_train)
    print(count_test)
    # SplitData_To_TrainandTest(csv_path, 10, 2, 1)
    # paths = [csv_path3]
    # for path in paths:
    #     print('数据集：%s信息如下：' % (path))
    #     Rc = Recommendation(path)
    #     print(Rc.num_users)
    #     print(Rc.num_items)
    #     print(sp.dok_matrix.count_nonzero(Rc.trainMatrix))
    #     print(Rc.trainMatrix)
    #     print('--------')
    #     print(Rc.testMatrix)
    #     print(Rc.ratingMax)
    #     print(Rc.ratingMim)
    #     testmatrix = sp.dok_matrix((5, 6), dtype=np.float32)
    #     testmatrix[0, 0] = 2
    #     testmatrix[1, 1] = 4
    #     testmatrix[2, 1] = 2
    #     testmatrix[2, 2] = 3
    #
    #     testmatrix[4, 2] = 2
    #     print(Rc.Evaluate_MAE(
    #         [[3, 0, 4, 0, 0, 0], [0, 5, 0, 5, 5, 3], [3, 4, 4, 3, 4, 3], [1, 2, 1, 0, 2, 0], [1, 0, 5, 0, 0, 0]],
    #         testmatrix))
