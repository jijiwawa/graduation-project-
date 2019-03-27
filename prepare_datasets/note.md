#准备IuIv,在用户评分矩阵中找到每个用户评分的物品


#计算用户的相似度


# 从物品的评分矩阵itemRatingMatrix中计算得到物品i的平均评分average和中位数med
def Average(self, i):
    act_i = i - 1
    act_row = self.itemRatingMatrix.getrow(act_i).toarray()[0]
    sum, count = 0, 0
    for i in range(0,int(float(self.ratingMax+1))):
        if i == 0 or act_row[i] == 0:
            continue
        else:
            sum += i * act_row[i]
            count += act_row[i]
    if count == 0:
        return 0.0
    else:
        return sum/count
        
# 从csv中生成物品的评分字典  物品/评分  值为评分个数
def Generate_ratingItemMatrix_ForEachItem(self, file):
    itemRatingMatrix = sp.dok_matrix((self.num_items, int(float(self.ratingMax)) + 1), dtype=np.int)
    with open(file, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(",")
            userId, itemId, rating = int(float(arr[0])), int(float(arr[1])), int(float(arr[2]))
            if itemRatingMatrix[itemId - 1, int(float(rating))] == None or itemRatingMatrix[
                itemId - 1, int(float(rating))] == 0.0:
                itemRatingMatrix[itemId - 1, int(float(rating))] = 1.0
            else:
                itemRatingMatrix[itemId - 1, int(float(rating))] += 1.0
            line = f.readline()
    return itemRatingMatrix