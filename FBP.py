import numpy as np

def FBP(img,filiter='ramp'):
    xrlen = img.shape[0]
    thetalen = img.shape[1]
    if(filiter=='ramp'):
        sgn_neg = -np.ones( int(xrlen / 2) , dtype=np.float32)
        sgn_pos =  np.ones( int(xrlen / 2) , dtype=np.float32)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #生成sgn滤波器
        if(xrlen%2==0):
            sgn=np.hstack((sgn_neg,sgn_pos))
        else:
            sgn=np.hstack((sgn_neg,[0],sgn_pos))
        sgn = sgn.repeat(thetalen)
        sgn = sgn.reshape(xrlen,thetalen)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #生成ramp函数
        ramp = np.linspace(-xrlen/2,xrlen/2,xrlen,endpoint=True,dtype=np.float32)
        ramp = ramp.repeat(thetalen)
        ramp = ramp.reshape(xrlen,thetalen)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #对原图像做xr方向上的傅里叶变换得到P并且平移
        P = np.fft.fft(img,axis=0)
        P = np.fft.fftshift(P,axes=0)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #滤波 反变换
        PF = P*sgn*ramp
        PF = np.fft.ifftshift(PF,axes=0)
        pf = np.fft.ifft(PF,axis=0)
        return np.abs(pf)                     




                       

               
               


