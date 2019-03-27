
from prepare_datasets.before_recommendation import loadcsv_to_list

test_matrix = np.array([[3, 0, 4, 0, 0, 0],
                        [0, 5, 0, 3, 5, 3],
                        [3, 4, 4, 3, 4, 4],
                        [1, 2, 1, 0, 2, 0],
                        [1, 0, 5, 0, 0, 0]])

class Recommendation
# 计算用户uv的相似度
'''1.'''








def Compute_UserSimilarity_of_U_V():
    Iv = []
    Iu = []
    S = 0.0
    for i in Iu:
        for j in Iv:
            a = 1
    # 首先在
    return S







#
def Ds(i,j):
    return (D(i,j)+D(j,i))/2

def D(i,j):
    return 1

data_path = 'C:\\Users\\41885\\Desktop\\Recommendation\\prepare_datasets\\test.csv'

ratingMatrix = Transform_csv_To_RatingMatrix(data_path)
data = loadcsv_to_list(data_path)
length = len(data)
print(length)
print(Generate_ratingItemList_ForEachUser(ratingMatrix))
