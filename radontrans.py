#!/usr/bin/python
import cv2
import numpy as np
import radon
import pdb
#此函数读入当前路径下的一张名叫lena.jpg的图片,并且做雷登变换

img=cv2.imread('lena.jpg',0)


pj,pj_theta=radon.randon(img)

np.savetxt("radon.txt", pj)


cv2.imshow('res',pj)
cv2.waitKey(0)
cv2.destroyAllWindows()
