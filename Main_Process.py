import math
import numpy as np
from scipy import interpolate
import matplotlib.pylab as plt
from Servants import Random_Walk_final_pos
from Servants import Random_Walk_full_series
from Servants import Series_sort


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''
Series=[]
for i in range(10000):
    Series.append(Random_Walk_final_pos(100))
x,y=Series_sort(Series)
plt.bar(x,y)
'''


plt.show()