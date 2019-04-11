import scipy.sparse as sp
import numpy as np
import math
import random


class Recommendation():

    def __init__(self, file):
        self.ratingMatrix = self.Transform_csv_To_RatingMatrix(file)
        # print(sp.dok_matrix.count_nonzero(self.ratingMatrix))
        print('success Transform_csv_To_RatingMatrix!!!')
        # self.trainMatrix, self.testMatrix = self.SplitData()
        self.num_users, self.num_items = self.ratingMatrix.shape
        self.ratingMax = self.ratingMatrix.toarray().max()
        self.ratingMim = self.ratingMatrix.toarray().min()
        self.num_scale = 5 # 1,2,3,4,5

    # 将csv转换称列表的形式
    def loadcsv_to_list(self,file):
        data = []
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                # user,item,rating=int(str.strip(arr[0])),int(str.strip(arr[1])),int(str.strip(arr[2]))
                user, item, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                data.append([user, item, rating])
                line = f.readline()
        return data

    # 将csv导入转换称评分矩阵
    def Transform_csv_To_RatingMatrix(self, file):
        num_users, num_items = 0, 0
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                userId, itemId = int(float(arr[0])), int(float(arr[1]))
                num_users = max(num_users, userId)
                num_items = max(num_items, itemId)
                line = f.readline()
        ratingMatrix = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                ratingMatrix[userId, itemId] = rating
                line = f.readline()
        return ratingMatrix

    # 分割数据集为测试集和训练集
    def SplitData(self, data, M, k, seed):
        test = []
        train = []
        random.seed(seed)
        for userId, movieId, rating in data:
            if random.randint(0, M) == k:
                test.append([userId, movieId, rating])
            else:
                train.append([userId, movieId, rating])
        return train, test

    # 返回评分矩阵
    def GetRatingMatrix(self):
        return self.ratingMatrix



if __name__ == '__main__':
    print('1233')
    # csv_path = 'C:\\Users\\41885\\Desktop\\Recommendation\\prepare_datasets\\test.csv'
    # Rc = Recommendation(csv_path)
    #
    # print(Rc.num_users)
    # print(Rc.ratingMax)
    a = range(1,5)
    print(a)