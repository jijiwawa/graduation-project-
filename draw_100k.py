# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import mpl
from Recommendation import Recommendation, get_dataset_path

mpl.rcParams['font.sans-serif'] = ['SimHei']
import scipy.sparse as sp
import numpy as np


def Draw_MAE():
    names = ['32', '64', '128']
    x = range(len(names))
    PMF_result = [0.7441103941113266,0.7408713117797431,0.742699502798439]
    DMF_result = [0.7112464959106123,0.713069654674644,0.7119933714667165]

    plt.plot(x, PMF_result, marker='+', mec='b',linestyle='--', mfc='w', label=u'PMF')
    plt.plot(x, DMF_result, marker='x', linestyle='-',ms=10, label=u'DMF')
    for a, b in zip(x, PMF_result):
        b1 = format(b, '.4f')
        plt.text(a, b, b1, ha='center', fontsize=10)
    for a, b in zip(x, DMF_result):
        b1 = format(b, '.4f')
        plt.text(a, b, b1, ha='center', fontsize=10)
    plt.legend()  # 让图例生效
    plt.xticks(x, names, rotation=45)
    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"K-潜在特征latent-feature")  # X轴标签
    plt.ylabel("M A E")  # Y轴标签
    plt.title("PMF、DMF算法MAE比较")  # 标题
    plt.show()


def Draw_HR(topk):
    names = ['32', '64', '128']
    x = range(len(names))
    if topk == 5:
        PMF_result = [0.08552499498092753,0.08191126279863481,0.08753262397109014]
        DMF_result = [0.0335,0.0449,0.0425]
    elif topk == 10:
        PMF_result = [0.08552499498092753, 0.08191126279863481, 0.08753262397109014]
        DMF_result = [0.0335, 0.0449, 0.0425]
    plt.plot(x, PMF_result, marker='+', mec='b', mfc='w', linewidth=4,linestyle=':',label=u'PMF')
    plt.plot(x, DMF_result, marker='x',linestyle='-',linewidth=2, ms=10, label=u'DMF')

    for a, b in zip(x, PMF_result):
        b1 = format(b, '.4f')
        plt.text(a, b, b1, ha='center', fontsize=10)
    for a, b in zip(x, DMF_result):
        b1 = format(b, '.4f')
        plt.text(a, b, b1, ha='center', fontsize=10)
    plt.legend()  # 让图例生效
    plt.xticks(x, names, rotation=45)
    plt.margins(0.01)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"K-潜在特征latent-feature")  # X轴标签
    plt.ylabel("HR@"+str(topk))  # Y轴标签
    plt.title('命中率HR@'+str(topk)+'比较')  # 标题

    plt.show()


def Draw_ALL():
    result = [0.7926,0.7688,0.8098,0.8028]
    plt.figure(figsize=(9, 6))
    n = 4
    X = np.arange(n) + 1
    plt.bar(X, result, width=0.15, facecolor='yellowgreen', edgecolor='white')
    # 给图加text
    for x, y in zip(X, result):
        plt.text(x, y, '%.4f' % y, ha='center', va='bottom')
    name = [' ', 'PCC', 'Hybird', 'PMF', 'DMF']
    x = range(len(name))
    plt.xticks(x, name, rotation=10)
    plt.margins(0.1)
    plt.xlabel(u"不同算法")  # X轴标签
    plt.ylabel("MAE")  # Y轴标签
    plt.title(u"四种算法最优MAE比较")  # 标题
    plt.ylim(0.6, )
    plt.show()


def DrawData_Distribute(dataname, trainname, testname):
    rec = Recommendation(dataname, trainname, testname)
    alldata_matrix = rec.Transform_csv_To_RatingMatrix(dataname)
    num_user = alldata_matrix.shape[0]
    count_rating_list = []
    for i in range(num_user):
        count_rating_list.append(sp.dok_matrix.count_nonzero(alldata_matrix[i]))
    # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.figure(figsize=(8, 6), dpi=80)
    # 再创建一个规格为 1 x 1 的子图
    plt.subplot(1, 1, 1)
    plt.margins(0.5)
    # 柱子总数
    N = len(count_rating_list)
    # 包含每个柱子对应值的序列
    values = count_rating_list
    # 包含每个柱子下标的序列
    index = np.arange(N)
    # 柱子的宽度
    width = 0.01
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p2 = plt.bar(index, values, label="rainfall", color="#87CEFA")
    # 设置横轴标签
    plt.xlabel('UserID')
    # 设置纵轴标签
    plt.ylabel('NumOfRating')
    # 添加标题
    plt.title('mk-1m评分分布')
    print(count_rating_list)
    # 添加纵横轴的刻度
    plt.xticks(index[np.array(count_rating_list) > 60])
    # plt.xticks((index for index in range(0,num_user,10)),(i for i in range(0,num_user,10)))
    # 添加图例
    plt.legend((u"用户的评分数量",))

    plt.show()

def Draw_HR_ALL(topk):
    if topk == 5:
        result = [0.0473, 0.0769, 0.0473, 0.0532]
    elif topk == 10:
        result = [0.1005, 0.1124, 0.0769, 0.0710]

    plt.figure(figsize=(9, 6))
    n = 4
    X = np.arange(n) + 1
    plt.bar(X, result, width=0.15, facecolor='yellowgreen', edgecolor='white')
    # 给图加text
    for x, y in zip(X, result):
        plt.text(x, y, '%.4f' % y, ha='center', va='bottom')
    name = [' ', 'PCC', 'Hybird', 'PMF', 'DMF']
    x = range(len(name))
    plt.xticks(x, name, rotation=10)
    plt.margins(0.1)
    plt.xlabel(u"不同算法")  # X轴标签
    plt.ylabel('H R @ '+str(topk))  # Y轴标签
    plt.title(u'四种算法HR @'+str(topk)+'比较')  # 标题
    plt.show()

if __name__ == '__main__':
    #         0: Hybird_,
    #         1: test_,
    #         2: ml_100k_,
    #         3: ml_1m_,
    #         4: pcc_data,
    #         5:ml_200_1000_
    # list_dataset = get_dataset_path(1)
    # Draw_MAE()
    # Draw_ALL()
    Draw_HR(10)
    # Draw_HR_ALL(5)
    # Draw_HR_ALL(10)
    # DrawData_Distribute(list_dataset[0],list_dataset[1],list_dataset[2])
