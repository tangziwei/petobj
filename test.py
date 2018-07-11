import numpy as np
import cv2
from scipy.interpolate import interp1d
import iradon

def make_rampfunc(x0,y0):
    return lambda x,y: np.sqrt((x-x0)**2+(y-y0)**2)

x=1
y=2
ramp=make_rampfunc(x,y)

print(ramp(-2,6))