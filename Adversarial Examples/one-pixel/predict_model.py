from keras.datasets import fashion_mnist
from keras.engine.saving import load_model
import keras

model = load_model('./../../model/LeNet.h5')
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 输入数据为 mnist 数据集
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

result = model.predict(x_test)
print(result)
