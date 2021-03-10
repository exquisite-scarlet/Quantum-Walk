import math
import numpy as np
from scipy import interpolate
import matplotlib.pylab as plt
from matplotlib.pylab import mpl
import random
#随机行走的最终位置
def Random_Walk_final_pos(Total_Steps):
    pos=0
    for i in range(1,Total_Steps):
        if np.random.rand(1)>=0.5:
            pos+=1
        else:
            pos-=1
    return pos
#随机行走位置序列
def Random_Walk_full_series(Total_Steps):
    Series=[0]
    for i in range(1,Total_Steps):
        if np.random.rand(1)>=0.5:
            Series.append(Series[i-1]+1)
        else:
            Series.append(Series[i-1]-1)
    return Series
#位置序列统计处理
def Series_sort(Series):
    Series_1=np.arange(min(Series),max(Series)+1)
    Series_2=np.zeros(np.size(Series_1))
    for i in range(np.size(Series_1)):
        Series_2[i]=Series.count(Series_1[i])
    return Series_1,Series_2    


#coin比特
class coin_bit:
    def __init__(self,initial_coin,initial_magn):
        self.coin,self.magn=initial_coin,initial_magn

#pos比特
class pos_bit:
    def __init__(self,initial_pos):
        self.pos=initial_pos

#dyadic比特
class dyadic_bit:
    def __init__(self,coin_bit,pos_bit):
        self.coin=coin_bit.coin
        self.pos=pos_bit.pos
        self.magn=coin_bit.magn
    def Hadamard(self):
        if self.coin==0:
            return dyadic_bit(coin_bit(0,1/np.sqrt(2)*self.magn),pos_bit(self.pos+1)),\
                   dyadic_bit(coin_bit(1,1/np.sqrt(2)*self.magn),pos_bit(self.pos-1))
        else:
            return dyadic_bit(coin_bit(0,1/np.sqrt(2)*self.magn),pos_bit(self.pos+1)),\
                   dyadic_bit(coin_bit(1,-1/np.sqrt(2)*self.magn),pos_bit(self.pos-1))