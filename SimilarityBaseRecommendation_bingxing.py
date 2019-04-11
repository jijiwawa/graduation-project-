# -*- coding: utf-8 -*-
import sys
import scipy.sparse as sp
import heapq
import numpy as np
import math
import time
import os
from prepare_datasets.Recommendation import Recommendation
import multiprocessing

class SimilarityBaseRecommendation(Recommendation):

    def __init__(self, file):
        self.sigma = 0.0000099
        self.count = 0
        super(SimilarityBaseRecommendation, self).__init__(file)
        print(self.ratingMatrix.toarray())
        self.coItemDict = self.Generate_ratingItemDict_ForEachUser(self.ratingMatrix)
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        print('successful generate ratingItemDict!!')
        self.itemRatingDict = self.Generate_ratingDict_ForEachItem(file)
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        print('successful generate ratingDict_ForEachItem!!')
        # 通过S2函数构建物品相似度矩阵，用索引方便获取，顺便测试是否会加快算法进度
        self.itemSimialrityMatrix_Sitem = self.Generate_ItemSimilarity_Matrix()
        # print(self.itemSimialrityMatrix_Sitem)
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        print(self.itemSimialrityMatrix_Sitem)
        np.save(os.getcwd() + '\out_file\itemSimialrityMatrix_Sitem.npy', self.itemSimialrityMatrix_Sitem)
        print('successful generate itemSimialrityMatrix!!')
        self.userSimilarityMatrix = self.Generate_UserSimilarity_Matrix()
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        print('successful Generate_UserSimilarity_Matrix!!')

    # 从评分矩阵中生成用户的评分了的物品列表 用户/物品
    def Generate_ratingItemDict_ForEachUser(self, ratingMatrix):
        ratingItemList = dict()
        for (userid, itemid) in ratingMatrix.keys():
            if userid not in ratingItemList.keys():
                ratingItemList[userid] = set()
            ratingItemList[userid].add(itemid)
        return ratingItemList

    # 从csv中生成物品的评分字典  {物品id:{评分}}
    def Generate_ratingDict_ForEachItem(self, file):
        itemRatingDict = dict()
        with open(file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                if itemId not in itemRatingDict.keys():
                    itemRatingDict[itemId] = []
                itemRatingDict[itemId].append(rating)
                line = f.readline()
        return itemRatingDict

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
            # piv = self.itemRatingDict[i].count(v) / (len(self.itemRatingDict[i]))
            # pjv = self.itemRatingDict[j].count(v) / (len(self.itemRatingDict[j]))
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
        Iu = self.coItemDict[u]
        Iv = self.coItemDict[v]
        sum = 0
        if self.ratingMax > 1:
            med = (1 + self.ratingMax) / 2
        else:
            med = 0.5
        for i in Iu:
            rui = self.ratingMatrix[u, i]
            for j in Iv:
                rvj = self.ratingMatrix[v, j]
                # sum += self.Sitem(i, j) * self.S1(rui, rvj, i, j, med)
                if i <= j:
                    sum += self.itemSimialrityMatrix_Sitem[i][j] * self.S1(rui, rvj, i, j, med)
                else:
                    sum += self.itemSimialrityMatrix_Sitem[j][i] * self.S1(rui, rvj, i, j, med)
        return sum

    # S2 Function S2 can be seen as an asymmetric factor, which describes the asymmetry between user u and user v.
    def S2(self, u, v):
        a_list = list(self.coItemDict[u])
        b_list = list(self.coItemDict[v])
        ret_list = list((set(a_list).union(set(b_list))) ^ (set(a_list) ^ set(b_list)))
        s2_result = 1 / (1 + math.exp(-(len(ret_list)) / len(a_list)))
        return s2_result

    # S3 Function S3 focuses on the rating preference of each user
    def S3(self, u, v):
        ave_u = sum(list(self.ratingMatrix[u].toarray()[0])) / len(self.ratingMatrix[u])
        ave_v = sum(list(self.ratingMatrix[v].toarray()[0])) / len(self.ratingMatrix[v])
        sum_u = 0
        sum_v = 0
        for rating in self.ratingMatrix[u].keys():
            sum_u += math.pow((self.ratingMatrix[u][rating] - ave_u), 2)
        sd_u = math.sqrt(sum_u / len(self.ratingMatrix[u]))
        for rating in self.ratingMatrix[v].keys():
            sum_v += math.pow((self.ratingMatrix[v][rating] - ave_v), 2)
        sd_v = math.sqrt(sum_v / len(self.ratingMatrix[v]))
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

    def multi_S(self, args):
        return self.S(*args)

    def multi_Sitem(self, args):
        return self.Sitem(*args)

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
            p = multiprocessing.Pool(processes=4)
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
            np.save(os.getcwd() + '\\out_file\\userSimilarity[%d].npy' % (userId_i), userSimilarityMatrix[userId_i])
        return userSimilarityMatrix

    # 选择前K个相似用户
    # def Choose_First_K_n_Neighbor_For_U(self, u, K):
    #     self.userSimilarityMatrix[u - 1]

    # 预测评分
    def Generate_Predicti_User_U_OnItem_I(self, u, i, K):
        ave_u = sum(list(self.ratingMatrix[u].toarray()[0])) / len(self.ratingMatrix[u])
        user_u_vertor = list(self.userSimilarityMatrix[1])
        # 前k相似用户result
        result = list(map(user_u_vertor.index, heapq.nlargest(K, user_u_vertor)))
        up_up = 0
        down_down = 0
        for v in result:
            down_down += math.fabs(self.S(u, v + 1))
            if i in self.coItemDict[v + 1]:
                ave_v = sum(list(self.ratingMatrix[v].toarray()[0])) / len(self.ratingMatrix[v])
                rvi = self.ratingMatrix[v, i]
                up_up += self.S(u, v + 1) * (rvi - ave_v)
        return ave_u + up_up / down_down

    # 生成预测评分矩阵
    def Generate_PredictRating_Matrix(self):
        resultMatrix = (self.ratingMatrix.toarray())
        for i in range(0, self.num_users):
            for j in range(0, self.num_items):
                if resultMatrix[i][j] == 0 or resultMatrix[i][j] == None:
                    resultMatrix[i][j] = self.Generate_Predicti_User_U_OnItem_I(i + 1, j + 1, 2)
        return resultMatrix

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
        print(self.ratingMatrix.toarray())
        # print(self.itemRatingDict)
        # print(self.coItemDict)
        # print(len(self.itemRatingDict[6]))
        # print(self.ratingMatrix[0, 0])
        # print(self.coItemDict[1])
        # print(self.Get_AveOfList(self.itemRatingDict[3]))
        # print(self.Get_MedOfList(self.itemRatingDict[5]))
        # print(self.Get_MedOfList([2,4,4,3,3,1]))

    # 测试S1
    def test_S1(self):

        print(self.S1(3, 5, 1, 2, 3))

    # 测试S2
    def test_S2(self):
        print('测试函数S2:')
        a = math.exp(0) + 1
        print(1 / a)
        print(self.S2(1, 2))

    # 测试S3
    def test_S3(self):
        print('测试函数S3:')
        # print(sum(list(self.ratingMatrix[1].toarray()[0])))
        # print(len(self.ratingMatrix[1]))
        # print((self.ratingMatrix[1]))
        # for rating in self.ratingMatrix[1].keys():
        #     print(self.ratingMatrix[1][rating])
        # result = 1 - 1 / (1 + math.exp(-2.5 * 0.5))
        # print(result)
        result = 1 - 1 / (1 + math.exp(-0.25))
        print(result)
        print(self.S3(1, 2))

    # 测试Sitem
    def test_Sitem(self):
        print('测试函数Sitem:')
        # result = 0
        # for x in range(1,6):
        #     pxi = self.itemRatingDict[2].count(x) / (len(self.itemRatingDict[2]))
        #     pxj = self.itemRatingDict[4].count(x) / (len(self.itemRatingDict[4]))
        #     up = self.sigma+pxi
        #     down = self.sigma+pxj
        #     result += ((self.sigma+pxi)/(1+5*self.sigma))*math.log2(up/down)
        # print(result)

        # print(self.Sitem(1,2))
        print(self.Sitem(2, 4))
        print(self.Sitem(4, 2))

    def show_all_paterner(self):
        print('评价物品表coItemDict')
        print(self.coItemDict)
        print('评分矩阵ratingMatrix')
        print(self.ratingMatrix)
        print('用户相似度矩阵userSimilarityMatrix')
        print(self.userSimilarityMatrix)
        print('用户评分矩阵ratingMatrix')
        print(sp.dok_matrix(self.ratingMatrix))
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

    def testS(self):
        # self.S(2,4)
        a = list(self.ratingMatrix.toarray()[0])
        print(a)
        while 0 in a:
            a.remove(0)
        print(a)

    def test_sigma(self):
        print(self.Generate_UserSimilarity_Matrix())
        self.sigma = 0.00001
        print(self.sigma)
        print(self.Generate_UserSimilarity_Matrix())
        self.sigma = 0.000009
        for i in range(1, 10, 2):
            self.sigma = 0.000009
            self.sigma += i / 10000000
            print(self.sigma)
            print(self.Generate_UserSimilarity_Matrix())
        for i in range(1, 10, 2):
            self.sigma = 0.00001
            self.sigma += i / 1000000
            print(self.sigma)
            print(self.Generate_UserSimilarity_Matrix())


if __name__ == '__main__':
    csv_path = 'C:\\Users\\suxik\\Desktop\\text\\graduation-project-\\prepare_datasets\\test.csv'
    csv_path1 = 'C:\\Users\\suxik\Desktop\\text\graduation-project-\\prepare_datasets\\ml-1m.train.rating'
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

    sbr = SimilarityBaseRecommendation(csv_path1)
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # print(sbr.ratingMatrix)
    # print(sbr.userSimilarityMatrix)
    # sbr.test_sigma()
    '''不同sigma的值不同的用户相似度
    for i in range(1,10,1):
        sbr.sigma=i/10
        print(sbr.sigma)
        print(sbr.Generate_UserSimilarity_Matrix())
    '''
    # sbr.testS()
    # sbr.test_S1()
    print(sbr.userSimilarityMatrix)
    # sbr.test_S2()
    # sbr.test_S3()
    # sbr.test_Sitem()
    # sbr.test_else()
    # sbr.show_all_paterner()
    # print(sbr.ratingMatrix.toarray())