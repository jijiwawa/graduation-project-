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


def Evaluate_Precision_AND_Recall(preditmatrix, testmatrix,topk):
    IRup = dict()
    IRua = dict()
    topK = topk
    for (userid, itemid) in testmatrix.keys():
        if userid not in IRua.keys():
            user_u_vertor = np.array(list(preditmatrix[userid]))
            IRup[userid] = np.argsort(user_u_vertor)[-topK:]
            user_u_vertor_test = np.array(sp.dok_matrix.toarray(testmatrix)[userid])
            IRua[userid] = np.argsort(user_u_vertor_test)[-topK:]
    sum_precision, sum_recall, m = 0, 0, 0
    for userid in IRua.keys():
        m += 1
        IRup_userid = list(IRup[userid])
        IRua_userid = list(IRua[userid])
        ret_list = list((set(IRup_userid).union(set(IRua_userid))) ^ (set(IRup_userid) ^ set(IRua_userid)))
        sum_precision += len(ret_list) / len(IRup_userid)
        sum_recall += len(ret_list) / len(IRua_userid)
    Precision = sum_precision / m
    Recall = sum_recall / m
    return Precision, Recall


def Evaluate_F1_measure(Precision, Recall):
    return (2 * Precision * Recall) / (Precision + Recall)