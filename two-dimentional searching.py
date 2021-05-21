from Zhihu import Two_Walk,possible
import numpy as np
from scipy import interpolate
import matplotlib.pylab as plt
import scipy.linalg
from scipy.linalg import kron
from PIL import Image
import os
from mpl_toolkits.mplot3d import Axes3D


#define some states and gates
spin_up = np.array([[1, 0]]).T
spin_down = np.array([[0, 1]]).T
bit = [spin_up, spin_down]
I = np.matrix("1 0; 0 1")
X = np.matrix("0 1; 1 0")
Y = np.matrix("0 -1j; 1j 0")
Z = np.matrix("1 0; 0 -1")
H = np.matrix("1 1; 1 -1") / np.sqrt(2)
CNOT = np.matrix("1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0")
SWAP = np.matrix("1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1")
gates = {'I':I,  'X':X, 'Y':Y, 'Z':Z, 'H':H, 'CNOT':CNOT, 'SWAP':SWAP}

#N for Walking times
N=60
#picture lim for y
lim=0.2
#showing interval(ms)
timess=100
#definition about the searching targets and basis
n=40

solution=[[20,20]]

X=range(n)
Y=range(n)
Coin=[]
initial=np.array([1,1,1,1]).T

# 1 for left, 2 for right, 3 for up, 4 for down
for i in X:
    Coin.append([])
    for j in Y:
        Coin[i].append(initial)

Coin=(1/np.sqrt(4*n*n))*np.array(Coin)

#operaters

G=(1/2)*np.array([[-1,1,1,1],[1,-1,1,1],[1,1,-1,1],[1,1,1,-1]])
I=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

result=[]
result.append(possible(Coin))
for kk in range(N):
    # 1 for left, 2 for right, 3 for up, 4 for down
    for i in range(len(Coin)):
        for j in range(len(Coin[0])):
            flag=0
            for solu in solution:
                if solu[0]==i and solu[1]==j:
                    flag=1
                    break
                else:
                    flag=0
            if flag==1:
                Coin[i][j]=np.dot(-I,Coin[i][j])
            else:
                Coin[i][j]=np.dot(G,Coin[i][j])

    Coin=Two_Walk(Coin)
    result.append(possible(Coin))


#print(result)
#drawing the GIF
os.chdir('D:\\Python\\Ultra Project\\Quantum Walk Grover\\pictures')

x,y=np.meshgrid(X,Y)

for i in range(len(result)):
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x,y,result[i],cmap='Spectral')
    #plt.zlim(0,lim)
    plt.title(str(i+1)+' steps')
    #fig.colorbar()
    ax.set_zlim(0,lim)
    plt.savefig(str(i+1)+' steps')
    plt.close()
    print('#',end='')

im=Image.open(str(1)+' steps.png')
images=[]
for i in range(N-1):
    images.append(Image.open(str(2+i)+' steps.png'))
im.save('animation.gif',save_all=True,append_images=images,loop=1,duration=1,comment=b'aaabb')
