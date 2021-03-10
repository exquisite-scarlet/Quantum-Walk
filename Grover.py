import math
import numpy as np
from scipy import interpolate
from scipy.linalg import kron
from Zhihu import basis,hilbert_space,Hadamard,wave_func,project,decompose,print_wf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

#N for Grover times
N=10
#picture lim for y
lim=0.2
#showing interval
timess=500
#definition about the searching targets and basis
solution=['0100010000','0100000000','0100000100','0000100000','0000000001','0100000000''0000100000','0000000001']
s001 = basis('0000000000')

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

#genearate Hilbert space and operaters. HN for Hadamard, ON for Oracle
HN=H    
ON=I
nbit=len(solution[0])
nspace=2**nbit
for i in range(N-1):
    HN = kron(H, HN)
    ON = kron(I, ON)
A0,T0=print_wf(HN * s001)
A0=np.matrix(A0)
A0=A0.T

for i in range(nspace):
    for j in solution:
            if T0[i]==j:
                ON[i,i]=-ON[i,i]
print(ON)
result=[]
B=[0]*nspace
for i in range(nspace):
    B[i]=A0[i,0]**2
result.append(B)

#implement grover
for kk in range(N):
    result.append([])
    #implement the Oracle
    A0=ON*A0
    #R2:Hadamard, flap, Hadamard
    A0=HN*A0
    A0=-A0
    A0[0,0]=-A0[0,0]
    A0=HN*A0
    #save the possibility
    for i in range(nspace):
        result[kk+1].append(A0[i,0]**2)

#drawing the GIF
fig, ax = plt.subplots()
def update(n):
    plt.title("z=%d"%(n))
    ax.cla()
    ax.set_xlim(0,nspace)
    ax.set_ylim(0,lim)
    return ax.plot(range(nspace), result[n])
def init():
    ax.set_xlim(0,nspace)
    ax.set_ylim(0,lim)
    return ax.plot(range(nspace), result[0])
ani = FuncAnimation(fig, update, frames=N,init_func=init, blit=False,interval=timess,repeat=True)
plt.show()
