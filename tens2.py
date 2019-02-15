import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorboard as tb


# # 创建一个常量Operation
# hw = tf.constant("Hello World!")
# # 启动会话
# sess = tf.Session()
# # 运行Graph（计算图）
# print(sess.run(hw))
# # 关闭会话
# sess.close()
#
#
#
# input = [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# print(tf.shape(input, name=None))
#
#
# diagonal = [1, 2, 3, 4]
# print(tf.diag(diagonal, name=None))