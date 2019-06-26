import numpy as np

import tensorflow as tf

import jsma


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy:
    pass


def get_env():
    env = Dummy()
    with tf.variable_scope('model'):
        env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                               name='x')
        env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
        env.training = tf.placeholder_with_default(False, (), name='mode')

        env.ybar, logits = model(env.x, logits=True, training=env.training)

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                           logits=logits)
            env.loss = tf.reduce_mean(xent, name='loss')

        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer()
            env.train_op = optimizer.minimize(env.loss)

        env.saver = tf.train.Saver()

    with tf.variable_scope('model', reuse=True):
        env.target = tf.placeholder(tf.int32, (), name='target')
        env.adv_epochs = tf.placeholder_with_default(20, shape=(), name='epochs')
        env.adv_eps = tf.placeholder_with_default(0.2, shape=(), name='eps')
        env.x_jsma = jsma.jsma(model, env.x, env.target, eps=env.adv_eps,
                          epochs=env.adv_epochs)

    return env


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def make_jsma(sess, env, X_data, epochs=0.2, eps=1.0, batch_size=128):
    """
    Generate JSMA by running env.x_jsma.
    """
    print('\nMaking adversarials via JSMA')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {
            env.x: X_data[start:end],
            env.target: np.random.choice(n_classes),
            env.adv_epochs: epochs,
            env.adv_eps: eps}
        adv = sess.run(env.x_jsma, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv


def aiTest(images, shape=(1000, 28, 28, 1)):
    env = get_env()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    env.saver.restore(sess, 'model/{}'.format('mnist'))

    print('\nGenerating adversarial data')

    X_adv = make_jsma(sess, env, images, epochs=100, eps=1.0)
    print(X_adv.shape)

    print('\nEvaluating on adversarial data')

    evaluate(sess, env, X_adv, y_test)

    return X_adv


img_size = 28
img_chan = 1
n_classes = 10

print('\nLoading FASHION MNIST')

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_test = to_categorical(y_test)

X_test = X_test[0:1000, :]
y_test = y_test[0:1000]

x_gengrate = aiTest(X_test)

from skimage.measure import compare_ssim
m = 0.0
for i in range(1000):
    (score, diff) = compare_ssim(X_test[i][:, :, 0], x_gengrate[i][:, :, 0], full=True)
    diff = (diff * 255).astype("uint8")
    m = m + score
    # print("SSIM: {}".format(score))

print(m/1000)

