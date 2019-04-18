# coding=utf-8
import tensorflow as tf
import numpy as np

# 定义一个未知变量input_ids用于存储索引
input_ids = tf.placeholder(dtype=tf.int32, shape=[None])

# 定义一个已知变量embedding，是一个5*5的对角矩阵
# embedding = tf.Variable(np.identity(5, dtype=np.int32))

# 或者随机一个矩阵
embedding = a = np.asarray([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]])

# 根据input_ids中的id，查找embedding中对应的元素
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# print(embedding.eval())
print(sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]}))

