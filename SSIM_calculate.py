from skimage.measure import compare_ssim
from keras.datasets import fashion_mnist

# 数据
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

grayA = x_train[0]
grayB = x_train[1]
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))