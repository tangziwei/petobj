#!/usr/bin/python
import numpy as np
import scipy
from scipy.interpolate import interp1d



def iradon(radon_mat,theta):

    x_len = radon_mat.shape[0]
    y_len = radon_mat.shape[1]

    pdb.set_trace()

    # 构造x坐标矩阵
    x = np.linspace(-x_len/2,  x_len/2,x_len,endpoint=True,dtype=np.float32)
    x = np.array((x,))
    x = x.repeat(x_len,axis=0)

    # 构造y坐标矩阵
    y = np.linspace( y_len/2, -y_len/2,y_len,endpoint=True,dtype=np.float32)
    y = np.array((y,))
    y = y.repeat(y_len, axis=0)
    y = y.transpose()

    # 初始化输出矩阵
    img=np.zeros((x_len,y_len),np.float32)

    # 生成距离xr向量
    dig = np.sqrt(x_len ** 2 + y_len ** 2)
    xr = np.linspace(-dig / 2, dig / 2, num=y_len, endpoint=True)

    # theta转为弧度制
    sin = np.sin(theta * np.pi / 180)
    cos = np.cos(theta * np.pi / 180)

    for i in range(x_len):
        # 计算插值函数
        f=interp1d(xr,radon_mat[:,i])#默认是线性插值

        # 计算每一个点的位置
        t = x*sin[i]+y*cos[i]

        # 叠加
        img = img+f(t)

    return img