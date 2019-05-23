# -*- coding: utf-8 -*-
import scipy.sparse as sp
import heapq
import numpy as np
import math
import time
import os
import re
from Recommendation import Recommendation, get_dataset_path
import multiprocessing


class PCC_Recommendation(Recommendation):
    def __init__(self, file, train_file, test_file):
        super(PCC_Recommendation, self).__init__(file, train_file, test_file)


        print('训练的评分矩阵：')
        print(self.trainMatrix.toarray())

        # 生成Pearson相关系数矩阵
        self.pccMatrix = self.Generate_pccMatrix()
        np.save(os.getcwd() + '\\out_file\\PCC\\PCC_pccMatrix_' + os.path.basename(
            train_file) + '_bingxing.npy', self.pccMatrix)
        print('Pearson相关系数矩阵：')
        print(self.pccMatrix)

        # 生成评分矩阵
        self.K = 4
        while self.K <= 20:
            self.predictMatrix = self.Generate_PredictRating_Matrix()
            np.save(os.getcwd() + '\\out_file\\PCC\\PCC_predictMatrix_' + str(self.K) + '_' + os.path.basename(
                train_file) + '_bingxing.npy', self.predictMatrix)
            print('预测评分矩阵：')
            print(self.predictMatrix)
            self.K+=4

    def Generate_pccMatrix(self):
        pccMatrix = np.zeros((self.num_users, self.num_users), dtype=np.float32)
        for u in range(self.num_users):
            for v in range(self.num_users):
                pccMatrix[u][v] = self.PCC(u, v)
        return pccMatrix

    def PCC(self, u, v):
        if u in self.coItemDict.keys():
            u_list = list(self.coItemDict[u])
        else:
            u_list = []
        if v in self.coItemDict.keys():
            v_list = list(self.coItemDict[v])
        else:
            v_list = []
        ave_u = self.user_ave_rating_dict[u]
        ave_v = self.user_ave_rating_dict[v]
        # 用户u和用户v共同评分了的物品
        u_v_ret_list = list((set(u_list).union(set(v_list))) ^ (set(u_list) ^ set(v_list)))
        if u_v_ret_list is None:
            print('用户%d和用户%d没有共同评分的物品', u, v)
            return 0
        else:
            up, down_left, down_right = 0, 0, 0
            try:
                for i in u_v_ret_list:
                    up += (self.trainMatrix[u, i] - ave_u) * (self.trainMatrix[v, i] - ave_v)
                    down_left += (self.trainMatrix[u, i] - ave_u) ** 2
                    down_right += (self.trainMatrix[v, i] - ave_v) ** 2
                if down_right == 0 or down_left == 0:
                    print('分母为0')
                else:
                    pccUV = up / (math.sqrt(down_left) * math.sqrt(down_right))
                return pccUV
            except:
                return 0



    # 预测评分
    def Generate_Predicti_User_U_OnItem_I(self, u, i):
        print("(%d,%d)用户对物品评分预测" % (u, i))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        if self.trainMatrix[u, i] == 0 or self.trainMatrix[u, i] == None:
            ave_u = self.user_ave_rating_dict[u]
            # 前k相似用户result
            # result = list(map(user_u_vertor.index, heapq.nlargest(self.K, user_u_vertor)))
            result = list(np.argsort(self.pccMatrix[u])[-self.K:])
            up_up = 0
            down_down = 0
            for v in result:
                down_down += math.fabs(self.pccMatrix[u][v])
                if v in self.coItemDict.keys():
                    v_list=self.coItemDict[v]
                else:
                    v_list=[]
                if i in v_list:
                    ave_v = sum(list(self.trainMatrix[v].toarray()[0])) / len(self.trainMatrix[v])
                    rvi = self.trainMatrix[v, i]
                    up_up += self.pccMatrix[u][v] * (rvi - ave_v)
            if down_down==0:
                return ave_u
            else:
                return ave_u + up_up / down_down
        else:
            print('用户%d对物品%d已有评分' % (u, i))
            # 返回-1 是为了方便后期计算命中率
            return -1

    def multi_Predict(self, args):
        return self.Generate_Predicti_User_U_OnItem_I(*args)

    # 生成预测评分矩阵
    def Generate_PredictRating_Matrix(self):
        predictMatrix = np.copy((self.trainMatrix.toarray()))
        for i in range(0, self.num_users):
            print("第%d个用户评分预测" % (i))
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            i1 = [i for x1 in range(0, self.num_items)]
            j = [x2 for x2 in range(0, self.num_items)]
            zip_args = list(zip(i1, j))
            # cores = multiprocessing.cpu_count()
            p = multiprocessing.Pool(processes=4)  # 64 服务器上
            predictMatrix[i1] = p.map(self.multi_Predict, zip_args)
            p.close()
            p.join()
            print("第第%d个用户评分预测结束" % (i))
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        return predictMatrix


if __name__ == '__main__':
    #         0: Hybird_,
    #         1: ml_100_400_,
    #         2: ml_100k_,
    #         3: ml_1m_,
    #         4: pcc_data,
    #         5:ml_200_1000_
    list_dataset = get_dataset_path(7)
    pcc_recommemdation = PCC_Recommendation(list_dataset[0], list_dataset[1], list_dataset[2])
