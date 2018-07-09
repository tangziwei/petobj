import numpy as np
import cv2
from scipy.interpolate import interp1d
import iradon

a = np.loadtxt('radon.txt')
a=a.transpose()

theta = np.linspace(0 , 180 , 512 ,endpoint=True ,dtype=np.float32)
print(theta)

result=iradon.iradon(a,theta)

np.savetxt("iradon.txt", result)