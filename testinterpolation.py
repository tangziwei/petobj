from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

data_x = np.linspace(0,10,num=10,endpoint=True)
data_y = np.sin(data_x)

f = interp1d(data_x,data_y)

new_x = np.linspace(0,10,num=100,endpoint=True)

new_y = f(new_x)

plt.plot(data_x, data_y, 'o', new_x, new_y, '-')
plt.legend(['data', 'linear'], loc='best')
plt.show()