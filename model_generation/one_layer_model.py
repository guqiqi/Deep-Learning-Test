import keras
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten
from keras import Sequential

# 模型
cnn1 = Sequential()

cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

cnn1.add(MaxPooling2D(pool_size=(2, 2)))

cnn1.add(Dropout(0.2))

cnn1.add(Flatten())

cnn1.add(Dense(128, activation='relu'))

cnn1.add(Dropout(0.5))
cnn1.add(Dense(10, activation='softmax'))

cnn1.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])

# 数据
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 输入数据为 mnist 数据集
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 训练
cnn1.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))

score = cnn1.evaluate(x_test, y_test)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

#  正确率 0.9185
cnn1.save("./../model/cnn1.h5")
