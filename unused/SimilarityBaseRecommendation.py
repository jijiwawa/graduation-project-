import scipy.sparse as sp
import heapq
import numpy as np
import math
import time
import os
from Recommendation import Recommendation


class SimilarityBaseRecommendation(Recommendation):

    def __init__(self, file, train_file, test_file):
        # 计算物品相似度用到的参数
        self.sigma = 0.0000099
        super(SimilarityBaseRecommendation, self).__init__(file, train_file, test_file)
        print('训练的评分矩阵：')
        print(self.trainMatrix.toarray())
        # 生成每个用户评分了的物品集合 {user:[item1,item2]}
        self.coItemDict = self.Generate_ratingItemDict_ForEachUser(self.trainMatrix)
        print('用户评分了的物品集合：')
        print(self.coItemDict)
        # 生成每个物品的评分集合 {item:[1.0,2.0]}
        self.itemRatingDict = self.Generate_ratingDict_ForEachItem(train_file)
        print('物品的评分集合：')
        print(self.itemRatingDict)
        # 生成物品相似度矩阵，用索引方便获取
        print('物品相似度矩阵：')
        # self.itemSimialrityMatrix_Sitem = self.Generate_ItemSimilarity_Matrix()
        # np.save(os.getcwd() + '\\out_file\\itemSimialrityMatrix_' + os.path.basename(train_file) + '_chuanxing.npy',
        #         self.itemSimialrityMatrix_Sitem)
        self.itemSimialrityMatrix_Sitem = np.load(
            os.getcwd() + '\\out_file\\itemSimialrityMatrix_' + os.path.basename(train_file) + '_chuanxing.npy')
        print(self.itemSimialrityMatrix_Sitem)
        # 生成用户相似度矩阵
        print('用户相似度矩阵：')
        # self.userSimilarityMatrix = self.Generate_UserSimilarity_Matrix()
        # np.save(os.getcwd() + '\\out_file\\userSimialrityMatrix_' + os.path.basename(train_file) + '_chuanxing.npy',
        #         self.userSimilarityMatrix)
        self.userSimilarityMatrix = np.load(
            os.getcwd() + '\\out_file\\userSimialrityMatrix_' + os.path.basename(train_file) + '_chuanxing.npy')
        print(self.userSimilarityMatrix)
        # 生成评分矩阵
        self.K = 10
        self.predictMatrix = self.Generate_PredictRating_Matrix()
        np.save(os.getcwd() + '\\out_file\\predictMatrix_' + str(self.K) + '_' + os.path.basename(
            train_file) + '_chuanxing.npy',
                self.predictMatrix)
        print('预测评分矩阵：')
        print(self.predictMatrix)


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
            for j in range(i+1, self.num_items):
                if i != j:
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
        a_list = list(self.coItemDict[u])
        b_list = list(self.coItemDict[v])
        ret_list = list((set(a_list).union(set(b_list))) ^ (set(a_list) ^ set(b_list)))
        s2_result = 1 / (1 + math.exp(-(len(ret_list)) / len(a_list)))
        return s2_result

    # S3 Function S3 focuses on the rating preference of each user
    def S3(self, u, v):
        ave_u = sum(list(self.trainMatrix[u].toarray()[0])) / len(self.trainMatrix[u])
        ave_v = sum(list(self.trainMatrix[v].toarray()[0])) / len(self.trainMatrix[v])
        sum_u = 0
        sum_v = 0
        for rating in self.trainMatrix[u].keys():
            sum_u += math.pow((self.trainMatrix[u][rating] - ave_u), 2)
        sd_u = math.sqrt(sum_u / len(self.trainMatrix[u]))
        for rating in self.trainMatrix[v].keys():
            sum_v += math.pow((self.trainMatrix[v][rating] - ave_v), 2)
        sd_v = math.sqrt(sum_v / len(self.trainMatrix[v]))
        s3_result = 1 - 1 / (1 + math.exp(-math.fabs(ave_u - ave_v) * math.fabs(sd_u - sd_v)))
        return s3_result

    # S(u,v)
    def S(self, u, v):
        print("(%d,%d)用户之间相似度计算" % (u,v))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        if u == v:
            s_result = 1
        else:
            s_result = self.S2(u, v) * self.S3(u, v) * self.midfunciion(u, v)
        return s_result

    # 生成用户相似度矩阵
    def Generate_UserSimilarity_Matrix(self):
        userSimilarityMatrix = np.zeros((self.num_users, self.num_users), dtype=np.float32)
        for userId_i in range(0, self.num_users):
            print("第%d个用户与其他用户的相似度开始" % (userId_i))
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            for userId_j in range(0, self.num_users):
                if userId_j != userId_i:
                    # print('computing user ' + str(userId_i) + ' and ' + str(userId_j))
                    # print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
                    userSimilarityMatrix[userId_i][userId_j] = self.S(userId_i, userId_j)
                    # print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
                    # print('success compute ' + str(userId_i) + ' and ' + str(userId_j))
            print("第%d个用户与其他用户的相似度结束" % (userId_i))
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        return userSimilarityMatrix

    # 选择前K个相似用户
    # def Choose_First_K_n_Neighbor_For_U(self, u, K):
    #     self.userSimilarityMatrix[u - 1]

    # 预测评分
    def Generate_Predicti_User_U_OnItem_I(self, u, i):
        print("(%d,%d)用户对物品评分预测" % (u, i))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        ave_u = sum(list(self.trainMatrix[u].toarray()[0])) / len(self.trainMatrix[u])
        user_u_vertor = list(self.userSimilarityMatrix[1])
        # 前k相似用户result
        result = list(map(user_u_vertor.index, heapq.nlargest(self.K, user_u_vertor)))
        up_up = 0
        down_down = 0
        for v in result:
            down_down += math.fabs(self.userSimilarityMatrix[u][v])
            if i in self.coItemDict[v]:
                ave_v = sum(list(self.trainMatrix[v].toarray()[0])) / len(self.trainMatrix[v])
                rvi = self.trainMatrix[v, i]
                up_up += self.userSimilarityMatrix[u][v] * (rvi - ave_v)
        return ave_u + up_up / down_down

    # 生成预测评分矩阵
    def Generate_PredictRating_Matrix(self):
        resultMatrix = (self.trainMatrix.toarray())
        for i in range(0, self.num_users):
            for j in range(0, self.num_items):
                if resultMatrix[i][j] == 0 or resultMatrix[i][j] == None:
                    resultMatrix[i][j] = self.Generate_Predicti_User_U_OnItem_I(i, j)
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
        print(self.trainMatrix.toarray())
        # print(self.itemRatingDict)
        # print(self.coItemDict)
        # print(len(self.itemRatingDict[6]))
        # print(self.trainMatrix[0, 0])
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
        # print(sum(list(self.trainMatrix[1].toarray()[0])))
        # print(len(self.trainMatrix[1]))
        # print((self.trainMatrix[1]))
        # for rating in self.trainMatrix[1].keys():
        #     print(self.trainMatrix[1][rating])
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

    def testS(self):
        # self.S(2,4)
        a = list(self.trainMatrix.toarray()[0])
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
        for i in range(1,10,2):
            self.sigma = 0.000009
            self.sigma += i/10000000
            print(self.sigma)
            print(self.Generate_UserSimilarity_Matrix())
        for i in range(1,10,2):
            self.sigma = 0.00001
            self.sigma += i/1000000
            print(self.sigma)
            print(self.Generate_UserSimilarity_Matrix())


if __name__ == '__main__':
    # test数据集合
    Hybird = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv'
    Hybird_train = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_train.csv'
    Hybird_test = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_test.csv'

    # test_99_400
    test = os.getcwd() + '\\prepare_datasets\\test_99_400.base'
    test_train = os.getcwd() + '\\prepare_datasets\\test_99_400.base_train.csv'
    test_test = os.getcwd() + '\\prepare_datasets\\test_99_400.base_test.csv'
    # m1-100k
    ml_100k = os.getcwd() + '\\prepare_datasets\\m1-100k.csv'
    ml_100k_train = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_train.csv'
    ml_100k_test = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_test.csv'

    # m1-1m
    ml_1m = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_train = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_test = os.getcwd() + '\\prepare_datasets\\ml-1m.test.rating'

    # gogogogogogogogo______test
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # sbr = SimilarityBaseRecommendation(Hybird,Hybird,Hybird_test)
    sbr = SimilarityBaseRecommendation(test, test_train, test_test)

    # sbr = SimilarityBaseRecommendation(ml_100k, ml_100k_train, ml_100k_test)
    # sbr = SimilarityBaseRecommendation(ml_1m,ml_1m_train,ml_1m_test)
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
