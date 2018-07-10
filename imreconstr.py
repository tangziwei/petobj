import numpy as np
from scipy.interpolate import interp1d


def imgnomal(img):
    """很多情况下得到的灰度点并不在0~255之间
    调用此函数江最大值映射为255,最小值映射为0以实现正常显示"""
    maxp = img.max()
    minp = img.min()
    px = 255 * (img - minp) / (maxp - minp)
    return np.uint8(px)

def radon(img):
    '''输入参数为一副图像 numpy array类型 返回函数的 radon变换阵,以及投影角度
       此函数用与模拟投影的过程,因为是用循环写的,处理时间比较长'''


    #初始化数组用于储存结果
    radon_mat = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)

    #原图f(x,y),投影图像,p(theta,xr)
    theta = np.linspace(0 , 180 ,img.shape[0],dtype=np.float32)
    diagonal = np.sqrt(img.shape[0]**2+img.shape[1]**2)
    xr = np.linspace(-diagonal/2,diagonal/2,img.shape[1],dtype=np.float32)
    #从0度到180度对原图像积分,间隔就是原来的间隔
    for theta_s in range(len(theta)):
        for xr_s in range(len(xr)):
            #求积分的值  即P(theta=theta, xr=xr)
            for sumx in range(-int(diagonal/2),int(diagonal/2)):
                theta_rad = np.pi*theta[theta_s]/180
                x=int(img.shape[0]/2-xr[xr_s]*np.sin(theta_rad)+sumx*np.cos(theta_rad))
                y=int(img.shape[1]/2+xr[xr_s]*np.cos(theta_rad)+sumx*np.sin(theta_rad))
                if(x>=0 and x<img.shape[0] and y>=0 and y<img.shape[1]):
                    radon_mat[xr_s, theta_s]=radon_mat[theta_s,xr_s]+img[int(x),int(y)]
    return radon_mat,theta


def iradon(radon_mat,theta):

    x_len = radon_mat.shape[0]
    y_len = radon_mat.shape[1]


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

    # 生成距离xr向量 dig 对角线长
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

def BPF( img , methord ):
    '''此函数返回频域滤波的结果 输入参数为img 与滤波方法
    可用的滤波器暂时只有ramp'''
    if(methord=='ramp'):
        #变换到频率域并平移
        P = np.fft.fft2(img)
        Pshift = np.fft.fftshift(P)
        #构造ramp滤波器
        ramp_filter = np.zeros((P.shape[0], P.shape[1]), dtype=np.float32)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
        #构造过程待优化
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                ramp_filter[i, j] = (i - P.shape[0] / 2 + 0.5) ** 2 + (j - P.shape[1] / 2 + 0.5) ** 2
        ramp_filter = np.sqrt(ramp_filter) / 360
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        "滤波 平移 反变换 取绝对值"
        fp_shift = Pshift * ramp_filter
        fp_ishift = np.fft.ifftshift(fp_shift)
        img_back = np.fft.ifft2(fp_ishift)
        img_back = np.abs(img_back)
        return img_back
    elif(methord=='harmming'):
        #待扩展
        return img
    #如果找不到对应的滤波器,就返回原图
    return img