import numpy as np

import tensorflow as tf

from attacks import jsma

def inference(input_tensor):
    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    # 第一层：卷积层，过滤器的尺寸为5×5，深度为6,不使用全0补充，步长为1。
    # 尺寸变化：32×32×1->28×28×6
    with tf.variable_scope('layer1-conv1'):
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
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [5, 5, 6, 16],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [16], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    # 尺寸变化：10×10×6->5×5×16
    with tf.variable_scope('layer4-pool2'):
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
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, 120], initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [120], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

    # 第六层：全连接层，120->84的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×120->64×84
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [120, 84], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [84], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

    # 第七层：全连接层（近似表示），84->10的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×10。最后，64×10的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
    # 即得到最后的分类结果。
    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable('weight', [84, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias', [10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit


def evaluate(sess, x_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = x_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc = 0, 0

    logits = inference(x_data)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_data)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_data)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [loss, accuracy],
            feed_dict={x_data[start:end],
                       y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def make_jsma(sess, env, x_data, epochs=0.2, eps=1.0, batch_size=128):
    """
    Generate JSMA by running env.x_jsma.
    """
    print('\nMaking adversarials via JSMA')

    n_sample = x_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(x_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {
            env.x: x_data[start:end],
            env.target: np.random.choice(n_classes),
            env.adv_epochs: epochs,
            env.adv_eps: eps}
        adv = sess.run(env.x_jsma, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv


# def aiTest(images, shape=(1000, 28, 28, 1)):
#     sess = tf.InteractiveSession()
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#
#     # env.saver.restore(sess, 'model/{}'.format('mnist'))
#     env.saver.restore(sess, 'model_lenet5/{}'.format("lenet"))
#
#     print('\nGenerating adversarial data')
#
#     X_adv = make_jsma(sess, env, images, epochs=100, eps=1.0)
#     print(X_adv.shape)
#
#     print('\nEvaluating on adversarial data')
#
#     evaluate(sess, env, X_adv, y_test)
#
#     return X_adv


img_size = 28
img_chan = 1
n_classes = 10

print('\nLoading FASHION MNIST')

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_test = np.reshape(x_test, [-1, img_size, img_size, img_chan])
x_test = x_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_test = to_categorical(y_test)

X_test = x_test[0:1000, :]
y_test = y_test[0:1000]

# x_gengrate = aiTest(X_test)
#
# from skimage.measure import compare_ssim
#
# m = 0.0
# for i in range(1000):
#     (score, diff) = compare_ssim(X_test[i][:, :, 0], x_gengrate[i][:, :, 0], full=True)
#     diff = (diff * 255).astype("uint8")
#     m = m + score
#     # print("SSIM: {}".format(score))
#
# print(m / 1000)

# 创建Session会话
with tf.Session() as sess:
    # 初始化所有变量(权值，偏置等)
    new_saver = tf.train.import_meta_graph('model_lenet5/lenet.meta')

    saver = tf.train.Saver()
    saver.restore(sess, 'model_lenet5/model_lenet5/{}'.format('lenet'))
    evaluate(sess, x_test, y_test)
    # 将所有样本训练10次，每次训练中以64个为一组训练完所有样本。
    # train_num可以设置大一些。
#     train_num = 1
#     batch_size = 64
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())
# logits = inference(x_test)
# saver = tf.train.Saver()
# saver.restore(sess, 'model_lenet5/model_lenet5/{}'.format('lenet'))
# evaluate(sess, x_test, y_test)