#!/usr/bin/python
# Filename: simplestclass.py
import cv2
import numpy as np
import pdb
import math
def randon(img):
    '''输入参数为一副图像 numpy array类型 返回函数的 radon变换阵,投影角度 '''


    #初始化数组用于储存结果
    radon_mat = np.zeros((img.shape[0],img.shape[1]),np.float32)

    #原图f(x,y),投影图像,p(theta,xr)
    theta = np.linspace(0 , 180 ,img.shape[0],np.float32)
    diagonal = np.sqrt(img.shape[0]**2+img.shape[1]**2)
    xr = np.linspace(-diagonal/2,diagonal/2,img.shape[1],np.float32)
    #从0度到180度对原图像积分,间隔就是原来的间隔
    for theta_s in range(len(theta)):
        for xr_s in range(len(xr)):
            #求积分的值  即P(theta=theta, xr=xr)
            for sumx in range(-int(diagonal/2),int(diagonal/2)):
                theta_rad = np.pi*theta[theta_s]/180
                x=int(img.shape[0]/2-xr[xr_s]*math.sin(theta_rad)+sumx*math.cos(theta_rad))
                y=int(img.shape[1]/2+xr[xr_s]*math.cos(theta_rad)+sumx*math.sin(theta_rad))
                if(x>=0 and x<img.shape[0] and y>=0 and y<img.shape[1]):
                    radon_mat[xr_s, theta_s]=radon_mat[theta_s,xr_s]+img[int(x),int(y)]
    return radon_mat,theta