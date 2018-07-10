import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb


p = np.loadtxt( 'BFPramp.txt' )
maxp=p.max()
minp=p.min()
px=255*(p-minp)/(maxp-minp)


cv2.imshow('img',np.uint8(px))
cv2.waitKey(0)
cv2.destroyAllWindows()