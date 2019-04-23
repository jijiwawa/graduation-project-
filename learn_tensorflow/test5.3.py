import tensorflow as tf


# 在名字为foo的命名空间内创建名字为v的变量：
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# 因为在命名空间foo中已经存在名字为v的变量，所以以下代码将会报错：
# Variable foo/v already exists, disallowed.Did you mean to set  reuse=True
# in VarScope?
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])

# 在生成上下文管理器时，将参数reuse设置成True。这样tf.get_variable函数将直接获取
# 已经声明的变量。
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v == v1)  # 将输出True，代表v，v1代表的是相同的TensorFlow中变量。

# 将参数reuse设置为True时，tf.variable_scope将只能获取已经创建过的变量。因为在
# 命名空间bar中还没有创建变量v，所以以下代码将会报错：
# Variable bar/v does not exist, disallowed,Did you mean to set reuse=None
# in VarScope?
with tf.variable_scope("bar", reuse=True):
    v = tf.get_variable("v", [1])
