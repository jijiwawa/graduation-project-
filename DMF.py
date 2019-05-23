import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.sparse as sp
import numpy as np
import os
import time
import evalute_tf
from Recommendation import Recommendation, get_dataset_path

# config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
config = tf.ConfigProto(allow_soft_placement=True)

SUMMARY_DIR = os.getcwd() + "\\path\\log"
TEST_DIR = os.getcwd() + "\\path\\test"
ALLDATARESULT_DIR = os.getcwd() + "\\path\\alldata"


# 获取训练样例
# 输入为训练的评分矩阵，以及负样本率。负样本个数=正样本数*neg_ratio
def get_train_instances(train, neg_ratio):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    num_items = train.shape[1]
    print(num_users)
    print(num_items)
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(train[u, i])
        # negative instances
        for t in range(neg_ratio):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    print('数据总量：%d' % len(user_input))
    print('加载数据完成！')
    return user_input, item_input, labels


def get_Userweight_variable(name, shape, regularizer):
    weights_user = tf.get_variable(name, shape,
                                   initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights_user))
    return weights_user


def get_Itemweight_variable(name, shape, regularizer):
    weights_item = tf.get_variable(name, shape,
                                   initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights_item))
    return weights_item


def variable_summaries(var, name):
    # 将生成监控信息的操作放到同一个命名空间下
    with tf.name_scope("summaries"):
        tf.summary.histogram(name, var)
        # 计算变量的平均值，并定义生成平均值信息日志的操作。记录变量平均值信息的日志标签名为'mean/'+name，其中mean为命名空间，/是
        # 命名空间的分隔符。name则给出了当前监控指标属于哪一个变量
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 计算变量的标准差，并定义生成其日志的操作
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


# 定义神经网络模型
def get_inference(user_vector, item_vector, rc, regularizer, keep_prob):
    layer_dimension = [32, 4]  # 每层隐藏层的输出维度

    # 定义深度神经网络前向传播过程
    with tf.name_scope('layer1_user'):
        with tf.name_scope('W1_user'):
            W1_user = get_Userweight_variable('W1_user', [rc.num_items, layer_dimension[0]], regularizer)
            # variable_summaries(W1_user, 'layer1_user/W1_user')
        with tf.name_scope('user_vector1'):
            layer1_user = tf.nn.relu(tf.matmul(user_vector, W1_user))
            layer1_user = tf.nn.dropout(layer1_user, keep_prob)

    with tf.name_scope('layer2_user'):
        with tf.name_scope('W2_user'):
            W2_user = get_Userweight_variable('W2_user', [layer_dimension[0], layer_dimension[1]], regularizer)
            # variable_summaries(W2_user, 'layer2_user/W2_user')

        with tf.name_scope('biases1_user'):
            biases1_user = tf.Variable(tf.constant(0.0, shape=[layer_dimension[1]]))
            # variable_summaries(biases1_user, 'layer2_user/biases1_user')

        with tf.name_scope('user_vector2'):
            layer2_user = tf.nn.relu(tf.matmul(layer1_user, W2_user) + biases1_user)
            layer2_user = tf.nn.dropout(layer2_user, keep_prob)

    with tf.name_scope('layer1_item'):
        with tf.name_scope('W1_item'):
            W1_item = get_Itemweight_variable('W1_item', [rc.num_users, layer_dimension[0]], regularizer)
            # variable_summaries(W1_item, 'layer1_item/W1_item')
        with tf.name_scope('item_vector1'):
            layer1_item = tf.nn.relu(tf.matmul(item_vector, W1_item))
            layer1_item = tf.nn.dropout(layer1_item, keep_prob)

    with tf.name_scope('layer2_item'):
        with tf.name_scope('W2_item'):
            W2_item = get_Itemweight_variable('W2_item', [layer_dimension[0], layer_dimension[1]], regularizer)
            # variable_summaries(W2_item, 'layer2_item/W2_item')
        with tf.name_scope('biases1_item'):
            biases1_item = tf.Variable(tf.constant(0.0, shape=[layer_dimension[1]]))
            # variable_summaries(biases1_item, 'layer2_item/biases1_item')

        with tf.name_scope('item_vector2'):
            layer2_item = tf.nn.relu(tf.matmul(layer1_item, W2_item) + biases1_item)
            layer2_item = tf.nn.dropout(layer2_item, keep_prob)

        # layer2_user = tf.nn.relu(tf.matmul(layer1_user, W2_user))
        # layer2_item = tf.nn.relu(tf.matmul(layer1_item, W2_item))

    # 结果
    y1 = tf.matmul(layer2_user, tf.transpose(layer2_item)) / (layer_dimension[-1] * layer_dimension[-1])
    y2 = tf.diag_part(y1)
    y = tf.clip_by_value(y2, 1.0e-6, 5.0)
    return y


if __name__ == '__main__':
    # 0: Hybird_,
    # 1: ml_100_400_,
    # 2: ml_100k_,
    # 3: ml_1m_,
    # 4: pcc_data,
    # 5: ml_200_1000_,
    # 6: test_99_400_
    list_dataset = get_dataset_path(1)
    recommemdation = Recommendation(list_dataset[0], list_dataset[1], list_dataset[2])
    modelsavepath = os.getcwd() + '\\model\\' \
                    + os.path.basename(list_dataset[0]) + '_model.ckpt'

    num_users = recommemdation.num_users
    num_items = recommemdation.num_items


    batch_size = 256  # batch大小
    # learning_rate = 0.0005  # 学习率
    learning_rate = 0.0001  # 学习率
    maxR = recommemdation.ratingMax  # 最大评分
    negative_radio = 0  # 一个正样本，negative_radio个负样本
    # layer_dimension = [32, 4]  # 每层隐藏层的输出维度,其实没什么用这里设置，get_inference
    STEPS = 2000  # 设定训练轮次。

    # 隐藏层层数N
    N = 2

    # 定义输入输出
    embedding = a = np.asarray(recommemdation.trainMatrix.toarray())
    keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('input_vector_user'):
        user_index = tf.placeholder(dtype='int32', shape=(None), name='user_index-input')
        user_vector = tf.nn.embedding_lookup(embedding, user_index)
    with tf.name_scope('input_vector_item'):
        item_index = tf.placeholder(dtype='int32', shape=(None), name='item_index-input')
        item_vector = tf.nn.embedding_lookup(np.transpose(embedding), item_index)
    with tf.name_scope('y_labels'):
        y_ = tf.placeholder(tf.float32, shape=(None), name='y-input')
    regularizer = slim.l2_regularizer(0.0001)
    y = get_inference(user_vector, item_vector, recommemdation, regularizer, keep_prob)

    # 定义损失函数和反向传播算法 y_ 标准答案 y 预测答案
    with tf.name_scope('cross_entropy'):
        # ce
        # cross_entropy = y_ * tf.log(y) + (1 - y_) * tf.log(1 - tf.where(tf.equal(y,1),1+1.0e-6,y))
        loss = tf.reduce_sum(tf.square(y-y_)) + tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('n_cross_entropy'):
        # nce 论文提出的损失函数
        loss1 = -tf.reduce_sum(y_ / maxR * tf.log(y) + (1 - y_ / maxR) * tf.log(1 - y)) + tf.add_n(
            tf.get_collection('losses'))
        tf.summary.scalar('n_cross_entropy', loss1)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.abs(y - y_))
        tf.summary.scalar('accuracy', accuracy)

    # 将所有日志写入文件
    merged = tf.summary.merge_all()

    print('加载训练集。。。。')

    # 加载训练集,获取输入输出
    user_input, item_input, labels = get_train_instances(recommemdation.trainMatrix, negative_radio)
    # 加载测试集,获取输入输出
    user_input_test, item_input_test, labels_test = evalute_tf.get_test_instances(recommemdation.testMatrix)

    # 训练神经网络
    dataset_size = len(user_input)
    print('开始训练。。。。')
    with tf.Session(config=config) as sess:
        # 参数初始化。
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 初始化写文件的writer，并将当前计算图写入日志
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        summary_writer_test = tf.summary.FileWriter(TEST_DIR)

        # 声明Saver类用于保存模型
        saver = tf.train.Saver()
        # sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]})
        # 迭代更新参数
        print('开始迭代')
        acc_min = 5
        for i in range(STEPS):
            k = 0
            end = 0
            sum_ = 0
            # 记录每batchsize的精度
            accuracy_list, n_cross_entropy_list = [], []
            # 将训练集数据跑一次
            while end < dataset_size:
                # 每次选取batch_size个样本进行训练
                start = (k * batch_size) % dataset_size
                end = min(start + batch_size, dataset_size)
                sess.run(train_step, feed_dict={user_index: np.array(user_input)[start:end],
                                                item_index: np.array(item_input)[start:end],
                                                y_: np.array(labels)[start:end], keep_prob: 0.9})
                accuracy_record, n_cross_entropy_score = sess.run([accuracy, loss], feed_dict={
                    user_index: np.array(user_input)[start:end],
                    item_index: np.array(item_input)[start:end],
                    y_: np.array(labels)[start:end], keep_prob: 1})
                accuracy_list.append(accuracy_record)
                n_cross_entropy_list.append(n_cross_entropy_score)

                k += 1
            # print('iter_num:%d', i)
            # print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            # sess.run(train_step, feed_dict={user_index: np.array(user_input), item_index: np.array(item_input),
            #                                 y_: np.array(labels)})
            # 每隔一段时间计算在所有数据上的交叉熵并输出。
            if i % 5 == 0:
                acc_all_data = np.mean(accuracy_list)
                n_cross_entropy_all_data = np.mean(n_cross_entropy_list)

                summary, _ = sess.run([merged, train_step], feed_dict={user_index: np.array(user_input)[start:end],
                                                                       item_index: np.array(item_input)[start:end],
                                                                       y_: np.array(labels)[start:end], keep_prob: 1})
                summary_test, acc_test = sess.run([merged, accuracy],
                                                  feed_dict={user_index: user_input_test,
                                                             item_index: item_input_test,
                                                             y_: labels_test, keep_prob: 1})
                print(
                    "After %d training step(s),(loss ,acc) on all data is (%g, %g) ,acc on test data is %g" % (
                        i, n_cross_entropy_all_data, acc_all_data, acc_test))
                if acc_test < acc_min:
                    saver.save(sess, modelsavepath)
                    acc_min = acc_test
                summary_writer.add_summary(summary, i)
                summary_writer_test.add_summary(summary_test, i)
        print('acc_test_min:',acc_min)
        summary_writer.close()
        summary_writer_test.close()
