import scipy.sparse as sp
import heapq
import numpy as np
import math
import time
from prepare_datasets.Recommendation import Recommendation


class SimilarityBaseRecommendation(Recommendation):

    def __init__(self, file):
        super(SimilarityBaseRecommendation, self).__init__(file)
        self.coItemDict = self.Generate_ratingItemDict_ForEachUser(self.ratingMatrix)
        print('successful fenerate ratingItemDict!!')
        self.itemRatingDict = self.Generate_ratingDict_ForEachItem(file)
        print('successful fenerate ratingDict_ForEachItem!!')
        self.userSimilarityMatrix = self.Generate_UserSimilarity_Matrix()
        print('successful Generate_UserSimilarity_Matrix!!')


    # 从评分矩阵中生成用户的评分了的物品列表 用户/物品
    def Generate_ratingItemDict_ForEachUser(self, ratingMatrix):
        ratingItemList = dict()
        for (userid, itemid) in ratingMatrix.keys():
            userid += 1
            itemid += 1
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
                arr = line.split(",")
                userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
                if itemId not in itemRatingDict.keys():
                    itemRatingDict[itemId] = []
                itemRatingDict[itemId].append(rating)
                line = f.readline()
        return itemRatingDict

    # The PSS model computes the user similarity value only based on the co-rated items
    # function S1 based on PSS model
    def S1(self, rui, rvj, i, j ,rmed):
        Proximity = 1 - 1 / (1 + math.exp(-math.fabs(rui - rvj)))
        Significance = 1 / (1 + math.exp(-math.fabs(rui - rmed) * math.fabs(rvj - rmed)))
        avei = self.Get_AveOfList(self.itemRatingDict[i])
        avej = self.Get_AveOfList(self.itemRatingDict[j])
        Singularity = 1 - 1 / (1 + math.exp(-math.fabs((rui + rvj) - (avei + avej)) / 2))
        s1_result = Proximity * Significance * Singularity
        return s1_result

    # Sitem(i,j) is an item similarity measure
    def Sitem(self, i, j):
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
            pxi = self.itemRatingDict[i].count(v) / (len(self.itemRatingDict[i]))
            pxj = self.itemRatingDict[j].count(v) / (len(self.itemRatingDict[j]))
            piv = (self.sigma + pxi) / (1 + self.sigma * self.num_scale)
            pjv = (self.sigma + pxj) / (1 + self.sigma * self.num_scale)
            sum += piv * math.log2(piv / pjv)
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
            rui = self.ratingMatrix[u - 1, i - 1]
            for j in Iv:
                rvj = self.ratingMatrix[v - 1, j - 1]
                sum += self.Sitem(i, j) * self.S1(rui, rvj, i, j, med)
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
        ave_u = sum(list(self.ratingMatrix[u - 1].toarray()[0])) / len(self.ratingMatrix[u - 1])
        ave_v = sum(list(self.ratingMatrix[v - 1].toarray()[0])) / len(self.ratingMatrix[v - 1])
        sum_u = 0
        sum_v = 0
        for rating in self.ratingMatrix[u - 1].keys():
            sum_u += math.pow((self.ratingMatrix[u - 1][rating] - ave_u), 2)
        sd_u = math.sqrt(sum_u / len(self.ratingMatrix[u - 1]))
        for rating in self.ratingMatrix[v - 1].keys():
            sum_v += math.pow((self.ratingMatrix[v - 1][rating] - ave_v), 2)
        sd_v = math.sqrt(sum_v / len(self.ratingMatrix[v - 1]))
        s3_result = 1 - 1 / (1 + math.exp(-math.fabs(ave_u - ave_v) * math.fabs(sd_u - sd_v)))
        return s3_result

    # S(u,v)
    def S(self, u, v):
        s_result = self.S2(u, v) * self.S3(u, v) * self.midfunciion(u, v)
        return s_result

    # 生成用户相似度矩阵
    def Generate_UserSimilarity_Matrix(self):
        userSimilarityMatrix = np.zeros((self.num_users, self.num_users), dtype=np.float32)
        for userId_i in range(1, self.num_users + 1):
            for userId_j in range(1, self.num_users + 1):
                if userId_j != userId_i:
                    # print('computing user '+str(userId_i)+ ' and ' + str(userId_j))
                    userSimilarityMatrix[userId_i - 1][userId_j - 1] = self.S(userId_i, userId_j)
                    # print('success compute '+str(userId_i)+ ' and ' + str(userId_j))
        return userSimilarityMatrix

    # 选择前K个相似用户
    # def Choose_First_K_n_Neighbor_For_U(self, u, K):
    #     self.userSimilarityMatrix[u - 1]

    # 预测评分
    def Generate_Predicti_User_U_OnItem_I(self, u, i, K):
        ave_u = sum(list(self.ratingMatrix[u - 1].toarray()[0])) / len(self.ratingMatrix[u - 1])
        user_u_vertor = list(self.userSimilarityMatrix[1])
        # 前k相似用户result
        result = list(map(user_u_vertor.index, heapq.nlargest(K, user_u_vertor)))
        up_up = 0
        down_down = 0
        for v in result:
            down_down += math.fabs(self.S(u, v+1))
            if i in self.coItemDict[v+1]:
                ave_v = sum(list(self.ratingMatrix[v].toarray()[0])) / len(self.ratingMatrix[v])
                rvi = self.ratingMatrix[v, i - 1]
                up_up += self.S(u, v+1) * (rvi - ave_v)
        return ave_u + up_up / down_down

    # 生成预测评分矩阵
    def Generate_PredictRating_Matrix(self):
        resultMatrix = (self.ratingMatrix.toarray())
        for i in range(0,self.num_users):
            for j in range(0, self.num_items):
                if resultMatrix[i][j] == 0 or resultMatrix[i][j] ==None:
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

        print(self.S1(3,5,1,2,3))

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

        print(self.Sitem(1,2))
        print(self.Sitem(2,5))
        print(self.Sitem(4,6))


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
        a=list(self.ratingMatrix.toarray()[0])
        print(a)
        while 0 in a:
            a.remove(0)
        print(a)

    def test_sigma(self):
        self.sigma = 0.000011
        print('sigma = 0.000011')
        print(self.Generate_UserSimilarity_Matrix())
        self.sigma = 0.000015
        print('sigma = 0.000015')
        print(self.Generate_UserSimilarity_Matrix())
        self.sigma = 0.0005
        print('sigma = 0.0005')
        print(self.Generate_UserSimilarity_Matrix())
        # self.sigma = 0.001
        # print('sigma = 0.001')
        # print(self.Generate_UserSimilarity_Matrix())
        # self.sigma = 0.01
        # print('sigma = 0.01')
        # print(self.Generate_UserSimilarity_Matrix())
        # self.sigma = 0.1
        # print('sigma = 0.1')
        # print(self.Generate_UserSimilarity_Matrix())


if __name__ == '__main__':
    csv_path = 'C:\\Users\\41885\\Desktop\\Recommendation\\prepare_datasets\\test.csv'
    csv_path1 = 'C:\\Users\\41885\\Desktop\\Recommendation\\prepare_datasets\\ratings_del_firstrow.csv'
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

    sbr = SimilarityBaseRecommendation(csv_path)
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # print(sbr.ratingMatrix)
    # print(sbr.userSimilarityMatrix)
    sbr.test_sigma()
    '''不同sigma的值不同的用户相似度
    for i in range(1,10,1):
        sbr.sigma=i/10
        print(sbr.sigma)
        print(sbr.Generate_UserSimilarity_Matrix())
    '''
    # sbr.testS()
    # sbr.test_S1()

    # sbr.test_S2()
    # sbr.test_S3()
    # sbr.test_Sitem()
    # sbr.test_else()
    # sbr.show_all_paterner()
    # print(sbr.ratingMatrix.toarray())
