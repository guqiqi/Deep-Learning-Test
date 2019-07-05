import tensorflow as tf
import numpy as np
from skimage.measure import compare_ssim

import deepfool


# 模型本身
def inference(input_tensor):
    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    # 第一层：卷积层，过滤器的尺寸为5×5，深度为6,不使用全0补充，步长为1。
    # 尺寸变化：32×32×1->28×28×6
    with tf.variable_scope('layer1-conv1', reuse=tf.AUTO_REUSE):
        conv1_weights = tf.get_variable('weight', [5, 5, 1, 6], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [6], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    # 尺寸变化：28×28×6->14×14×6
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第三层：卷积层，过滤器的尺寸为5×5，深度为16,不使用全0补充，步长为1。
    # 尺寸变化：14×14×6->10×10×16
    with tf.variable_scope('layer3-conv2', reuse=tf.AUTO_REUSE):
        conv2_weights = tf.get_variable('weight', [5, 5, 6, 16],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    # 尺寸变化：10×10×6->5×5×16
    with tf.variable_scope('layer4-pool2', reuse=tf.AUTO_REUSE):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为5×5×16的矩阵，然而第五层全连接层需要的输入格式
    # 为向量，所以我们需要把代表每张图片的尺寸为5×5×16的矩阵拉直成一个长度为5×5×16的向量。
    # 举例说，每次训练64张图片，那么第四层池化层的输出的size为(64,5,5,16),拉直为向量，nodes=5×5×16=400,尺寸size变为(64,400)
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])

    # 第五层：全连接层，nodes=5×5×16=400，400->120的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×400->64×120
    # 训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题。
    # 这和模型越简单越不容易过拟合思想一致，和正则化限制权重的大小，使得模型不能任意拟合训练数据中的随机噪声，以此达到避免过拟合思想一致。
    # 本文最后训练时没有采用dropout，dropout项传入参数设置成了False，因为训练和测试写在了一起没有分离，不过大家可以尝试。
    with tf.variable_scope('layer5-fc1', reuse=tf.AUTO_REUSE):
        fc1_weights = tf.get_variable('weight', [nodes, 120], initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [120], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

    # 第六层：全连接层，120->84的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×120->64×84
    with tf.variable_scope('layer6-fc2', reuse=tf.AUTO_REUSE):
        fc2_weights = tf.get_variable('weight', [120, 84], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [84], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

    # 第七层：全连接层（近似表示），84->10的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×10。最后，64×10的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
    # 即得到最后的分类结果。
    with tf.variable_scope('layer7-fc3', reuse=tf.AUTO_REUSE):
        fc3_weights = tf.get_variable('weight', [84, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias', [10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit


# 获得正确率
def evaluate(x_data, y_data, num):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,
                           [num, 28, 28, 1],
                           name="x-input")
        y_ = tf.placeholder(tf.float32, shape=(num, 10), name="y-input")
        validate_feed = {
            x: np.reshape(x_data, [num, 28, 28, 1]),
            y_: np.reshape(y_data, [num, 10])
        }

        y = inference(x)

        correct_prediction = tf.equal(tf.cast(tf.arg_max(y, 1), tf.int64), tf.cast(tf.arg_max(y_, 1), tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # variables_aves = tf.train.ExponentialMovingAverage(p_train.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variables_aves.variables_to_restore()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("model_lenet5")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("validation accuracy = %g." % (accuracy_score))

            else:
                print("Error")


def make_deep_fool(x_data, epochs=20, eta=0.01, batch_size=64):
    """
    Generate Deep fool by running env.x_jsma.
    """
    tf.reset_default_graph()
    with tf.Graph().as_default():
        print('\nMaking adversarials via deep fool')

        n_sample = x_data.shape[0]
        n_batch = int((n_sample + batch_size - 1) / batch_size)
        x_adv = np.empty_like(x_data)

        x = tf.placeholder(tf.float32, (None, 28, 28, 1), name='x')
        epoch = tf.placeholder_with_default(10, shape=(), name='epochs')

        deep_fool_model = deepfool.deepfool(inference, x, eta=eta, epochs=epoch)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("model_lenet5")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                for batch in range(n_batch):
                    start = batch * batch_size
                    end = min(n_sample, start + batch_size)
                    feed_dict = {
                        x: x_data[start:end],
                        epoch: epochs
                    }
                    adv = sess.run(deep_fool_model, feed_dict=feed_dict)
                    x_adv[start:end] = adv
    print('over')

    return x_adv


def aiTest(images, shape=(1000, 28, 28, 1)):
    images = images / 256
    result = make_deep_fool(images)
    return result * 256


def get_ssim(img1, img2):
    m = 0.0
    for i in range(1000):
        (score, diff) = compare_ssim(img1[i][:, :, 0], img2[i][:, :, 0], full=True)
        diff = (diff * 255).astype("uint8")
        m = m + score
        # print("SSIM: {}".format(score))
    return m / 1000
