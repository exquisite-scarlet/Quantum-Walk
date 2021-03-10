import math
import numpy as np
from scipy import interpolate
import matplotlib.pylab as plt
from scipy.linalg import kron
import Zhihu
from Zhihu import basis,hilbert_space,Hadamard,wave_func,project,decompose,print_wf,get_directions
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import os
from PIL import Image

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

N=10
lim=0.2
XX=500

solution=['0100010000','0100000000','0100000000','0100000100','0000100000','0000000001','0000010000','0000000000','0100000000','0000000100','0000100000','0000000001']
solution=['0100010000','0100000000','0000100000','0000000001']

s001 = basis('0000000000')
IIX = kron(H, kron(H, kron(H, kron(H, kron(H, kron(H, kron(H, kron(H, kron(H, H)))))))))
A0,T0=print_wf(IIX * s001)

def uncoupled(A0,T0,nbit):
    D=get_directions(nbit)
    C=[]
    result=[]
    for i in range(len(D)):
        C.append(0)
        for j in range(len(T0)):
            if T0[j]==D[i]:
                C[i]=(A0[j]/A0[0])**2
        result.append((1/(1+C[i]**2)))
        result.append((C[i]**2/(1+C[i]**2)))
    return np.array(result)

A0=np.matrix(A0)
A0=A0.T
nbit=len(solution[0])
nspace=2**nbit
BITS=[]
result=[]
B=[0]*nspace
for i in range(nspace):
    B[i]=A0[i,0]**2
BITS.append(uncoupled(A0,T0,nbit))
result.append(B)





for kk in range(N):
    result.append([])
    #print('this is '+str(kk)+' range:')
    for i in range(nspace):
        for j in solution:
            if T0[i]==j:
                A0[i,0]=-A0[i,0]
                #print('hello')
    
    #print('oracle:!!')
    #print(A0.T)
    A0=IIX*A0
    #print('Hadamard:!!')
    #print(A0.T)
    A0=-A0
    A0[0,0]=-A0[0,0]
    #print('flip:!!')
    #print(A0.T)
    A0=IIX*A0
    for i in range(nspace):
        result[kk+1].append(A0[i,0]**2)
    BITS.append(uncoupled(A0,T0,nbit).reshape(2*nbit))
    print(BITS[kk+1])


    
    #print('Hadamard:!!')
    #print(A0.T)

#drawing the GIF
os.chdir('D:\\Python\\Ultra Project\\Grover\\pictures')

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}
for i in range(len(result)):
    plt.figure()
    plt.plot(range(nspace),result[i])
    plt.ylim(0,lim)    
    #plt.xlabel(BITS[i],font1)
    plt.xlabel('states')
    plt.ylabel('Probability')
    plt.title(str(i+1)+' steps')
    plt.legend('probability')
    plt.savefig(str(i+1)+' steps')
    print('*',end='')
    plt.close()

im=Image.open(str(1)+' steps.png')
images=[]
for i in range(N-1):
    images.append(Image.open(str(2+i)+' steps.png'))
im.save('animation.gif',save_all=True,append_images=images,loop=1,duration=XX,comment=b'aaabb')
