import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# 注意编码方式
def readnpy(file):
    NPY = np.load(file, allow_pickle=True, encoding="latin1")

    print("------data-------")
    print(NPY)
    print("------type-------")
    print(type(NPY))
    print("------shape-------")
    print('ndarray的形状: ', NPY.shape)
    print('ndarray的维度: ', NPY.ndim)
    print('ndarray的元素数量: ', NPY.size)
    print('ndarray中的数据类型: ', NPY.dtype)

def readout(file):
    out = cv2.imread(file, -1)
    # out = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
    print("------data-------")
    print(out)
    print("------type-------")
    print(type(out))
    print("------shape-------")
    print('ndarray的形状: ', out.shape)
    print('ndarray中的数据类型: ', out.dtype)
    print('ndarray的维度: ', out.ndim)
    print('ndarray的元素数量: ', out.size)
    plt.subplot()
    plt.imshow(out, cmap=plt.cm.gray)
    plt.show()

    return out


if __name__ == "__main__":
    # file = "Z_test.npy"
    # readnpy(file)

    # png = "./data/deep22/000000mask.png"
    png = "./data/mask5back/train/000000.png"
    im = readout(png)
    #
    # tiff = "./Z_train/5backtiff/000000.tiff"
    # im = readout(tiff)