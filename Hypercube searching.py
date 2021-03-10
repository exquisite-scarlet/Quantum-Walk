from Zhihu import basis,hilbert_space,wave_func,print_wf,get_directions,exclusive,Walk,get_Prob,normalize
import numpy as np
from scipy import interpolate
import matplotlib.pylab as plt
import scipy.linalg
from scipy.linalg import kron
from PIL import Image
import os

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
N=1000
#picture lim for y
lim=0.18
#showing interval(ms)
timess=40
#definition about the searching targets and basis
solution=['10010100','01000100','01011110','10100000','11000000']
s001 = basis('00000000')
nbit=len(solution[0])
nspace=2**nbit

#genearate Hilbert space
H_single_coin=H
for i in range(nbit-1):
    H_single_coin=kron(H, H_single_coin)
A0,T0=print_wf(H_single_coin*s001)

#geneate initial_states
Coin=[]
for i in range(nspace):
    Coin.append([])
    for j in range(nbit):
        Coin[i].append(1/np.sqrt(nbit*nspace)) 

Directions=get_directions(nbit)
Quantum=[Coin,Directions,T0]
print(Quantum)
results=[]
results.append(get_Prob(Quantum))

#genearate operaters.

Coin_single=np.matrix(Coin[0])
S_c=Coin_single.T*Coin_single
I_single=np.identity(nbit)
C_0=-I_single+2*S_c
C_1=-I_single
Oracle=np.matrix([0])
flag=0

for i in range(len(T0)):
    for j in solution:
        if j==T0[i]:
            flag=0
            break
        else:
            flag=1
    if flag==0:
        Oracle=scipy.linalg.block_diag(Oracle,C_1)
    else:
        Oracle=scipy.linalg.block_diag(Oracle,C_0)

Oracle=[line[1:] for line in Oracle[1:]]
Oracle=np.matrix(Oracle)
print(Oracle)


#Grover for Coins
for kk in range(N):
    Coin_state=[]
    for i in Quantum[0]:
        Coin_state.extend(i)
    Coin_state=np.matrix(Coin_state).T
    Coin_state=Oracle*Coin_state
    Coin_state=Coin_state.T
    Coin=np.array(Coin_state)[0]
    Coin=Coin.reshape(nspace,nbit)
    Quantum=[Coin,Directions,T0]
    #print(Quantum)
    Quantum=Walk(Quantum)
    Quantum=normalize(Quantum)
    results.append(get_Prob(Quantum))
    print('#',end="")

#print(results)
#drawing the GIF
os.chdir('D:\\Python\\Ultra Project\\Quantum Walk Grover\\pictures')

for i in range(len(results)):
    plt.figure()
    plt.plot(range(nspace),results[i])
    plt.ylim(0,lim)    
    plt.xlabel('Quantum States')
    plt.ylabel('Probability')
    plt.title(str(i+1)+' steps')
    plt.savefig(str(i+1)+' steps')
    print('*',end='')
    plt.close()

im=Image.open(str(1)+' steps.png')
images=[]
for i in range(250):
    images.append(Image.open(str(2+4*i)+' steps.png'))
im.save('animation.gif',save_all=True,append_images=images,loop=1,duration=1,comment=b'aaabb')
