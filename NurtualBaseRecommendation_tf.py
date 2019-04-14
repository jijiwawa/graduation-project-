import tensorflow as tf
from Recommendation import Recommendation

csv_path = 'C:\\Users\\suxik\\Desktop\\text\\graduation-project-\\prepare_datasets\\Hybird_data.csv'
# m1-1m
csv_path1 = 'C:\\Users\\suxik\Desktop\\text\graduation-project-\\prepare_datasets\\ml-1m.train.rating'
# m1-100k
csv_path2 = 'E:\\0学业\\毕设\\useful_dataset\\m-100k\\m1-100k.csv'
# 给出每个隐藏层输出的维度，即降维之后的横坐标
recommemdation = Recommendation(csv_path2)

layer_dimension =[128,64]  # 每层隐藏层的输出维度
batch_size = 256       # batch大小
learning_rate = 0.0001 # 学习率
negative_radio = 7     # 一个正样本，7个负样本

# 隐藏层层数N
N = len(layer_dimension)

# 定义权重参数Wh_user,Wh_item
Wh_user = tf.Variable()
Wh_item = tf.Variable()
for i in range(1,N+1):
    Wh_user = tf.Variable()
    Wh_item = tf.Variable()

# 定义输入输出
user_vertor = tf.placeholder(tf.float32,shape=(batch_size,recommemdation.num_users),name='user_vector-input')
item_vertor = tf.placeholder(tf.float32,shape=(batch_size,recommemdation.num_items),name='item_vertor-input')
y=tf.matmul(user_vertor,item_vertor)/(dimensional[-1]*dimensional[-1])

# 定义深度神经网络前向传播过程


# 定义损失函数和反向传播算法
cross_entropy =
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


# 加载训练集


# 训练神经网络
STEPS = 100
with tf.Session() as sess:
    # 参数初始化。

    # 迭代更新参数
    for i in range(STEPS):
        # 准备batch_size个训练数据。一般将所有训练数据随机打乱之后再选取可以得到
        current_X
        current_Y=
        sess.run(train_step,feed_dict={x:current_X,y:current_Y})




