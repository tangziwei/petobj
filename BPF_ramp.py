import numpy as np
import cv2

import pdb


p = np.loadtxt( 'iradon.txt' )
P = np.fft.fft2(p)
Pshift = np.fft.fftshift(P)

ramp_filter=np.zeros((P.shape[0],P.shape[1]),dtype=np.float32)

for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        ramp_filter[i,j]=(i-P.shape[0]/2+0.5)**2+(j-P.shape[1]/2+0.5)**2
pdb.set_trace()
ramp_filter=np.sqrt(ramp_filter)/360


fp_shift=Pshift*ramp_filter

fp_ishift = np.fft.ifftshift(fp_shift)
img_back = np.fft.ifft2(fp_ishift)
img_back = np.abs(img_back)

np.savetxt("BFPramp.txt", img_back)