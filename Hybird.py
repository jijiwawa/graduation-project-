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


class SimilarityBaseRecommendation(Recommendation):

    def __init__(self, file, train_file, test_file):
        # 计算物品相似度用到的参数
        self.sigma = 0.0000099
        super(SimilarityBaseRecommendation, self).__init__(file, train_file, test_file)
        print('训练的评分矩阵：')
        print(self.trainMatrix.toarray())

        # 生成每个物品的评分集合 {item:[1.0,2.0]}
        self.itemRatingDict = self.Generate_ratingDict_ForEachItem(train_file)
        print('物品的评分集合：')
        print(self.itemRatingDict)
        # 生成物品相似度矩阵，用索引方便获取
        print('物品相似度矩阵：')
        # self.itemSimialrityMatrix_Sitem = self.Generate_ItemSimilarity_Matrix()
        # np.save(
        #     os.getcwd() + '\\out_file\\Hybird\\itemSimialrityMatrix_' + os.path.basename(train_file) + '_bingxing.npy',
        #     self.itemSimialrityMatrix_Sitem)
        self.itemSimialrityMatrix_Sitem = np.load(
            os.getcwd() + '\\out_file\\Hybird\\itemSimialrityMatrix_' + os.path.basename(train_file) + '_bingxing.npy')
        print(self.itemSimialrityMatrix_Sitem)
        # 生成用户相似度矩阵
        print('用户相似度矩阵：')
        # self.userSimilarityMatrix = self.Generate_UserSimilarity_Matrix()
        # np.save(
        #     os.getcwd() + '\\out_file\\Hybird\\userSimialrityMatrix_' + os.path.basename(train_file) + '_bingxing.npy',
        #     self.userSimilarityMatrix)
        self.userSimilarityMatrix = np.load(
            os.getcwd() + '\\out_file\\Hybird\\userSimialrityMatrix_' + os.path.basename(train_file) + '_bingxing.npy')
        print(self.userSimilarityMatrix)
        # 生成评分矩阵
        self.K = 4
        while self.K <= 20:
            self.predictMatrix = self.Generate_PredictRating_Matrix()
            np.save(os.getcwd() + '\\out_file\\Hybird\\predictMatrix_' + str(self.K) + '_' + os.path.basename(
                train_file) + '_bingxing.npy',
                    self.predictMatrix)
            print('预测评分矩阵：')
            print(self.predictMatrix)
            self.K += 4

    # The PSS model computes the user similarity value only based on the co-rated items
    # function S1 based on PSS model
    def S1(self, rui, rvj, i, j, rmed):
        Proximity = 1 - 1 / (1 + math.exp(-math.fabs(rui - rvj)))
        Significance = 1 / (1 + math.exp(-math.fabs(rui - rmed) * math.fabs(rvj - rmed)))
        avei = self.Get_AveOfList(self.itemRatingDict[i])
        avej = self.Get_AveOfList(self.itemRatingDict[j])
        Singularity = 1 - 1 / (1 + math.exp(-math.fabs((rui + rvj) - (avei + avej)) / 2))
        s1_result = Proximity * Significance * Singularity
        return s1_result

    # Sitem(i,j) is an item similarity measure
    def Sitem(self, i, j):
        # 只用计算一半
        if j <= i:
            Sitem = 1
        else:
            Sitem = 1 / (1 + self.Ds(i, j))
        return Sitem

    def Ds(self, i, j):
        Ds = (self.D(i, j) + self.D(j, i)) / 2
        return Ds

    def D(self, i, j):
        sum = 0
        for v in range(1, int(float(self.ratingMax + 1))):
            '''i物品中评分为v的概率'''
            try:
                pxi = self.itemRatingDict[i].count(v) / (len(self.itemRatingDict[i]))
                pxj = self.itemRatingDict[j].count(v) / (len(self.itemRatingDict[j]))
                piv = (self.sigma + pxi) / (1 + self.sigma * self.num_scale)
                pjv = (self.sigma + pxj) / (1 + self.sigma * self.num_scale)
                sum += piv * math.log2(piv / pjv)
            except:
                print('物品%d和%d有一个没有评分' % (i, j))
                break
        return sum

    # S1*Sitem 中间过渡函数
    def midfunciion(self, u, v):
        if u in self.coItemDict.keys():
            Iu = self.coItemDict[u]
        else:
            Iu = []
        if v in self.coItemDict.keys():
            Iv = self.coItemDict[v]
        else:
            Iv = []
        sum = 0
        if self.ratingMax > 1:
            med = (1 + self.ratingMax) / 2
        else:
            med = 0.5
        for i in Iu:
            rui = self.trainMatrix[u, i]
            for j in Iv:
                rvj = self.trainMatrix[v, j]
                # sum += self.Sitem(i, j) * self.S1(rui, rvj, i, j, med)
                if i <= j:
                    sum += self.itemSimialrityMatrix_Sitem[i][j] * self.S1(rui, rvj, i, j, med)
                else:
                    sum += self.itemSimialrityMatrix_Sitem[j][i] * self.S1(rui, rvj, i, j, med)
        return sum

    # S2 Function S2 can be seen as an asymmetric factor, which describes the asymmetry between user u and user v.
    def S2(self, u, v):
        if u in self.coItemDict.keys():
            a_list = list(self.coItemDict[u])
        else:
            a_list = []
        if v in self.coItemDict.keys():
            b_list = list(self.coItemDict[v])
        else:
            b_list = []
        ret_list = list((set(a_list).union(set(b_list))) ^ (set(a_list) ^ set(b_list)))
        if len(a_list) ==0:
            lenght=1
        else:
            lenght=len(a_list)
        s2_result = 1 / (1 + math.exp(-(len(ret_list)) / lenght))
        return s2_result

    # S3 Function S3 focuses on the rating preference of each user
    def S3(self, u, v):
        ave_u = self.user_ave_rating_dict[u]
        ave_v = self.user_ave_rating_dict[v]
        sum_u = 0
        sum_v = 0
        for rating in self.trainMatrix[u].keys():
            sum_u += math.pow((self.trainMatrix[u][rating] - ave_u), 2)
        if len(self.trainMatrix[u]) == 0:
            sd_u = 0
        else:
            sd_u = math.sqrt(sum_u / len(self.trainMatrix[u]))
        for rating in self.trainMatrix[v].keys():
            sum_v += math.pow((self.trainMatrix[v][rating] - ave_v), 2)
        if len(self.trainMatrix[v]) == 0:
            sd_v = 0
        else:
            sd_v = math.sqrt(sum_v / len(self.trainMatrix[v]))
        s3_result = 1 - 1 / (1 + math.exp(-math.fabs(ave_u - ave_v) * math.fabs(sd_u - sd_v)))
        return s3_result

    # S(u,v)
    def S(self, u, v):
        print("(%d,%d)用户之间相似度计算" % (u, v))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        if u == v:
            s_result = 1
        else:
            s_result = self.S2(u, v) * self.S3(u, v) * self.midfunciion(u, v)
        return s_result

    def multi_Sitem(self, args):
        return self.Sitem(*args)

    def multi_S(self, args):
        return self.S(*args)

    def multi_Predict(self, args):
        return self.Generate_Predicti_User_U_OnItem_I(*args)

    # 生成一个物品相似度矩阵
    def Generate_ItemSimilarity_Matrix(self):
        itemSimialrityMatrix_Sitem = np.ones((self.num_items, self.num_items), dtype=np.float32)
        for i in range(0, self.num_items):
            # item_i = [i for x2 in range(0, self.num_items)]
            # item_j = [x2 for x2 in range(0, self.num_items)]
            # zip_args = list(zip(item_i, item_j))
            # cores = multiprocessing.cpu_count()
            # p = multiprocessing.Pool(processes=4)
            # itemSimialrityMatrix_Sitem[i] = p.map(self.multi_Sitem, zip_args)
            # p.close()
            # p.join()
            for j in range(i + 1, self.num_items):
                itemSimialrityMatrix_Sitem[i][j] = self.Sitem(i, j)
        return itemSimialrityMatrix_Sitem

    # 生成用户相似度矩阵
    def Generate_UserSimilarity_Matrix(self):
        userSimilarityMatrix = np.zeros((self.num_users, self.num_users), dtype=np.float32)
        for userId_i in range(0, self.num_users):
            print("第%d个用户与其他用户的相似度开始" % (userId_i))
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            userId_i1 = [userId_i for x2 in range(0, self.num_users)]
            userId_j = [x2 for x2 in range(0, self.num_users)]
            zip_args = list(zip(userId_i1, userId_j))
            cores = multiprocessing.cpu_count()
            p = multiprocessing.Pool(processes=4)  # 64 服务器上
            userSimilarityMatrix[userId_i] = p.map(self.multi_S, zip_args)
            p.close()
            p.join()
            print("第%d个用户与其他用户的相似度结束" % (userId_i))
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            # for userId_j in range(0, self.num_users):
            #     if userId_j != userId_i:
            #         print('computing user ' + str(userId_i) + ' and ' + str(userId_j))
            #         print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            #
            #         userSimilarityMatrix[userId_i][userId_j] = self.S(userId_i, userId_j)
            #         print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            #         print('success compute ' + str(userId_i) + ' and ' + str(userId_j))
        return userSimilarityMatrix

    # 预测评分
    def Generate_Predicti_User_U_OnItem_I(self, u, i):
        print("(%d,%d)用户对物品评分预测" % (u, i))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        if self.trainMatrix[u, i] == 0 or self.trainMatrix[u, i] == None:
            ave_u = self.user_ave_rating_dict[u]
            # user_u_vertor = list(self.userSimilarityMatrix[u])
            # 前k相似用户result
            # result = list(map(user_u_vertor.index, heapq.nlargest(self.K, user_u_vertor)))
            result = list(np.argsort(self.userSimilarityMatrix[u])[-self.K:])

            up_up = 0
            down_down = 0
            for v in result:
                down_down += math.fabs(self.userSimilarityMatrix[u][v])
                if v in self.coItemDict.keys():
                    v_list = self.coItemDict[v]
                else:
                    v_list = []
                if i in v_list:
                    ave_v = self.user_ave_rating_dict[v]
                    rvi = self.trainMatrix[v, i]
                    up_up += self.userSimilarityMatrix[u][v] * (rvi - ave_v)
            return ave_u + up_up / down_down
        else:
            print('用户%d对物品%d已有评分' % (u, i))
            # 返回-1 是为了方便后期计算命中率
            return -1

    # 生成预测评分矩阵
    def Generate_PredictRating_Matrix(self):
        predictMatrix = np.copy((self.trainMatrix.toarray()))
        # for i in range(0, self.num_users):
        #     print("第%d个用户评分预测" % (i))
        #     print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        #     for j in range(0, self.num_items):
        #         if predictMatrix[i][j] == 0 or predictMatrix[i][j] == None:
        #             predictMatrix[i][j] = self.Generate_Predicti_User_U_OnItem_I(i, j)
        for i in range(0, self.num_users):
            print("第%d个用户评分预测" % (i))
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            i1 = [i for x1 in range(0, self.num_items)]
            j = [x2 for x2 in range(0, self.num_items)]
            zip_args = list(zip(i1, j))
            cores = multiprocessing.cpu_count()
            p = multiprocessing.Pool(processes=4)  # 64 服务器上
            predictMatrix[i1] = p.map(self.multi_Predict, zip_args)
            p.close()
            p.join()
            print("第第%d个用户评分预测结束" % (i))
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        return predictMatrix

    # 求列表的中位数
    def Get_MedOfList(self, data):
        data.sort()
        half = len(data) // 2
        return (data[half] + data[~half]) / 2

    # 求列表的平均数
    def Get_AveOfList(self, data):
        sum = 0
        for i in range(len(data)):
            sum += data[i]
        return sum / len(data)

    # 测试用
    def testFunction(self):
        print(self.trainMatrix.toarray())
        # print(self.itemRatingDict)
        # print(self.coItemDict)
        # print(len(self.itemRatingDict[6]))
        # print(self.trainMatrix[0, 0])
        # print(self.coItemDict[1])
        # print(self.Get_AveOfList(self.itemRatingDict[3]))
        # print(self.Get_MedOfList(self.itemRatingDict[5]))
        # print(self.Get_MedOfList([2,4,4,3,3,1]))

    def show_all_paterner(self):
        print('评价物品表coItemDict')
        print(self.coItemDict)
        print('评分矩阵trainMatrix')
        print(self.trainMatrix)
        print('用户相似度矩阵userSimilarityMatrix')
        print(self.userSimilarityMatrix)
        print('用户评分矩阵trainMatrix')
        print(sp.dok_matrix(self.trainMatrix))
        print('预测评分矩阵result')
        print(self.Generate_PredictRating_Matrix())

    def test_else(self):
        print('相似度矩阵')
        print(self.userSimilarityMatrix)
        nums = list(self.userSimilarityMatrix[1])
        print(nums)
        result = list(map(nums.index, heapq.nlargest(2, nums)))
        result.sort()
        print(result)


if __name__ == '__main__':
    #         0: Hybird_,
    #         1: test_,
    #         2: ml_100k_,
    #         3: ml_1m_,
    #         4: pcc_data,
    #         5:ml_200_1000_
    list_dataset = get_dataset_path(7)

    # gogogogogogogogo______test
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # sbr = SimilarityBaseRecommendation(Hybird,Hybird,Hybird_test)
    sbr = SimilarityBaseRecommendation(list_dataset[0], list_dataset[1], list_dataset[2])
    # sbr = SimilarityBaseRecommendation(ml_100k, ml_100k_train, ml_100k_test)
    # sbr = SimilarityBaseRecommendation(ml_1m,ml_1m_train,ml_1m_test)
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
