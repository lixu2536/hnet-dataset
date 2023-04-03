import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os
from PIL import Image
import os.path
import glob


file_dir = "./"  # npy文件路径
dest_dir = "./Z_test/"  # 图片文件存储的路径


def npy_png(file_dir, dest_dir):
    file = file_dir + 'Z_test.npy'  # .npy文件名
    con_arr = np.load(file)
    print(type(con_arr))
    print(con_arr.shape)
    print(con_arr)
    count = 0  # 序号，用作设置文件名
    for con in con_arr:
        arr = con

        # 本人使用的数据集存储长为784的矩阵，即数据集图片为28*28像素
        arr = np.reshape(arr, (480, 640))
        # 括号内数值根据相应的数据集或需求修改
        # im = Image.fromarray(arr)

        # im = im.convert('L')  # 转为灰度图
        ###### im = cv2.normalize(arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.normalize为对图像进行指定范围的归一化变换；输入，输出，下界，上界，计算方式，数据类型
        # 32F为32位浮点数，8U为无符号8位

        im = np.asarray(arr, np.float32)
        # cv2.imwrite(dest_dir + '{:06d}'.format(count) + ".tiff", im, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
        cv2.imwrite(dest_dir + '{:06d}'.format(count) + ".png", im, (cv2.IMWRITE_PNG_COMPRESSION, 9))
        # 设定图片文件名为6位，如_000100.png
        '''
        path=dest_dir + "_" + '{:06d}'.format(count) + ".png"
        image = Image.open(path)
        image = image.convert("RGB")　# 转换为RGB
        os.remove(path)
        image.save(path)　# RGB图片替换此灰度图
        '''
        count = count + 1


# 从npy中读取单张图片，并保存为pil-image格式
def npytest(input):
    # file = np.load(input)
    # labels = sorted(list(file[0]))
    # labels_dir = input
    con_arr = np.load(input)
    print(type(con_arr))
    print(con_arr.shape)
    print(con_arr)
    count = 0  # 序号，用作设置文件名

    for con in con_arr:

        arr = con
        arr = np.reshape(arr, (480, 640))
        l1 = np.asarray(arr, np.float32)
        # 读取单张图片，以float32 类型
        l1 = Image.fromarray(l1)  # 将cv格式转为pil格式，用于后续的transform

        count = count + 1


def npy_read_one(input, num):
    con_arr = np.load(input)
    one = con_arr[0]
    one = one[:, :, num]
    print(type(one))
    print(one.shape)
    print(one)
    im = cv2.normalize(one, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite('npy_read_one{}.png'.format(num), im, (cv2.IMWRITE_PNG_COMPRESSION, 9))


# 灰度图转为深度伪彩色图
def convertPNG(pngfile, outdir):
    # 读取灰度图
    im_depth = cv2.imread(pngfile)
    # 转换成伪彩色（之前必须是8位图片）
    # 这里有个alpha值，深度图转换伪彩色图的scale可以通过alpha的数值调整，我设置为1，感觉对比度大一些
    # im_color = cv2.applyColorMap(cv2.convertScaleAbs(im_depth, alpha=1), cv2.COLORMAP_JET)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    # applyColorMap对彩色或深度图进行伪彩色处理，输入需要是8位无符号：convertScaleAbs处理；指定参考色彩的列表
    # print(im_color.shape)
    # 转成png
    im = Image.fromarray(im_color)
    # 对图像进行指定模式的输出
    # print(im.shape)
    # 保存图片
    im.save(os.path.join(outdir, os.path.basename(pngfile)))


if __name__ == "__main__":
    # npy_png(file_dir, dest_dir)

    # for pngfile in glob.glob(dest_dir + "*.png"):
    #     convertPNG(pngfile, dest_dir + "ColorMap/")

    # npytest('Z_test.npy')
    npy_read_one('X_test_double_fringe.npy', 0)
    npy_read_one('X_test_double_fringe.npy', 1)