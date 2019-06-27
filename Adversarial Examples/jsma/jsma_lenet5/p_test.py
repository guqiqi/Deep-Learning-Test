import tensorflow as tf
from keras.datasets import fashion_mnist
import numpy as np

from jsma_lenet5 import Lenet


def evaluate(x_test, y_test):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,
                           [5000, 28, 28, 1],
                           name="x-input")
        y_ = tf.placeholder(tf.float32, shape=(5000, 10), name="y-input")
        validate_feed = {
            x: np.reshape(x_test, [5000, 28, 28, 1]),
            y_: np.reshape(y_test, [5000, 10])
        }

        y = Lenet.inference(x)

        correct_prediction = tf.equal(tf.cast(tf.arg_max(y, 1), tf.int64), tf.cast(tf.arg_max(y_, 1), tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # variables_aves = tf.train.ExponentialMovingAverage(p_train.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variables_aves.variables_to_restore()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("model_lenet5")
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print(global_step)

                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training steps, validation accuracy = %g." % (global_step, accuracy_score))

            else:
                print("Error")


def main(argv=None):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_test = np.reshape(X_test, [-1, 28, 28, 1])
    X_test = X_test.astype(np.float32) / 255

    to_categorical = tf.keras.utils.to_categorical
    y_test = to_categorical(y_test)

    X_test = X_test[0:5000]
    y_test = y_test[0:5000]

    evaluate(X_test, y_test)


if __name__ == '__main__':
    tf.app.run()
