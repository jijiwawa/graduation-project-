import tensorflow as tf
'''
# 定义一个简单的计算图，实现向量加,法的操作
input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")
'''
# 改进
# 将输入定义放入各自的命名空间中，从而使得TensorBoard可以根据命名空间来整理可视化效
# 果图上的节点。
with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

# 生成一个写日志的writer，并将当前的TensorFlow计算图写入日志。
# TensorFlow提供了多种写日志文件的API，在11.3节中介绍。
writer = tf.summary.FileWriter("/path1/to/log11.2.1", tf.get_default_graph())
writer.close()
