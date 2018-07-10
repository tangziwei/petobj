import imreconstr
import cv2
from matplotlib import pyplot as plt
import numpy as np
#读入当前当前路路径下的一张图片
img=cv2.imread('lena.jpg',0)

#调用radon函数模拟投影过程 pj为返回的投影图像 pj_theta为返回的投影角度
pj,pj_theta=imreconstr.radon(img)


#pj = np.loadtxt('radon.txt')
#pj_theta = np.linspace(0 , 180 , 512 ,endpoint=True ,dtype=np.float32)

#调用反投影函数重建图像:
bj=imreconstr.iradon(pj,pj_theta)

#调用函数滤波
bj_f=imreconstr.BPF(bj,'ramp')

#图像归一化
img2=imreconstr.imgnomal(bj)
img3=imreconstr.imgnomal(bj_f)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img2, cmap = 'gray')
plt.title(' Direct BP'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img3,cmap= 'gray')
plt.title('Result in BPF'), plt.xticks([]), plt.yticks([])
plt.show()