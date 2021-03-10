import math
import numpy as np
from scipy import interpolate
import matplotlib.pylab as plt
from scipy.linalg import kron
import markdown as md

spin_up = np.array([[1, 0]]).T
spin_down = np.array([[0, 1]]).T
# bit[0] = |0>, bit[1] = |1>
bit = [spin_up, spin_down]

def basis(string='00010'):
    '''string: the qubits sequence'''
    res = np.array([[1]])
    # 从最后一位开始往前数，做直积
    for idx in string[::-1]:
        res = kron(bit[int(idx)], res)    
    return np.matrix(res)

def exclusive(str1,str2):
    A=eval('0b'+str1)
    B=eval('0b'+str2)
    C=A^B
    D=bin(C).replace('0b','')
    n=len(str1)-len(D)
    return '0'*n+D

def hilbert_space(nbit=5):
    nspace = 2**nbit
    for i in range(nspace):
        #bin(7) = 0b100
        binary = bin(i)[2:]
        nzeros = nbit - len(binary)
        yield '0'*nzeros + binary 

def directions(nbit):
    for i in range(nbit):
        n_former_zeros=nbit-1-i
        n_last_zeros=i
        yield '0'*n_former_zeros+'1'+'0'*n_last_zeros

def get_directions(nbit):
    res=[]
    for i in directions(nbit):
        res.append(i)
    return res

def Hadamard(A=[0,1]):
    a=(1/np.sqrt(2))*(A[0]+A[1])
    b=(1/np.sqrt(2))*(A[0]-A[1])
    return [a,b]

def wave_func(coef=[], seqs=[]):
    '''返回由振幅和几个Qubit序列表示的叠加态波函数，
       sum_i coef_i |psi_i> '''
    res = 0
    for i, a in enumerate(coef):
        res += a * basis(seqs[i])
    return np.matrix(res)

def project(wave_func, direction):
    '''<Psi | phi_i> to get the amplitude '''
    return wave_func.H * direction

def decompose(wave_func):
    '''将叠加态波函数分解'''
    nbit = int(np.log2(len(wave_func)))
    amplitudes = []
    direct_str = []
    for seq in hilbert_space(nbit):
        direct = basis(seq)
        amp = project(wave_func, direct).A1[0]
        if np.linalg.norm(amp) != 0:
            amplitudes.append(amp)
            direct_str.append(seq)
    return amplitudes, direct_str


def print_wf(wf):
    coef, seqs = decompose(wf)
    '''
    latex = ""
    for i, seq in enumerate(seqs):
        latex += r"%s$|%s\rangle$"%(coef[i], seq)
        if i != len(seqs) - 1:
            latex += "+"
            '''
    print('the coef does:')
    print(coef)
    print('the seqs does:')
    print(seqs)
    print('----------------------------------------------------------------------------------------------------------')
    return [coef,seqs]

def Walk(Quantum):
    C=Quantum[0]
    D=Quantum[1]
    T=Quantum[2]
    Coin=[]
    for sacred in C:
        Coin.append([i*0 for i in sacred])
    for j in range(len(C)):
        for i in range(len(C[0])):
            walk_state=exclusive(D[i],T[j])
            for k in range(len(T)):
                if T[k]==walk_state:
                    Coin[k][i]+=C[j][i]
    return [Coin,D,T]

def get_Prob(Quantum):
    Coin=Quantum[0]
    Prob=[]
    for i in Coin:
        possib=0
        for j in i:
            possib+=j**2
        Prob.append(possib)
    return Prob

def normalize(Quantum):
    Coin=Quantum[0]
    New_Coin=[]

    p=sum(get_Prob(Quantum))
    for i in Coin:
        New_Coin.append(i/np.sqrt(p))

    return [New_Coin,Quantum[1],Quantum[2]]

def Two_Walk(Coin):
    # 1 for left, 2 for right, 3 for up, 4 for down
    sgn = lambda x: 1 if x > 0 else  0
    L=len(Coin[0])
    New_coin=0*Coin
    for i in range(len(Coin)):
        for j in range(len(Coin[0])):
            New_coin[i][j+1-L*(sgn(j+2-L))][0]=Coin[i][j][0]
            New_coin[i][j-1+L*(sgn(1-j))][1]=Coin[i][j][1]
            New_coin[i+1-L*(sgn(i+2-L))][j][2]=Coin[i][j][2]
            New_coin[i-1+L*(sgn(1-i))][j][3]=Coin[i][j][3]
    return np.array(New_coin)

def possible(Coin):
    L=len(Coin)
    Prob=np.zeros([L,L])
    for i in range(L):
        #print('hello!')
        for j in range(L):
            #print('why?')
            for kkk in Coin[i][j]:
                #print(kkk,Prob[i][j])
                Prob[i][j]=Prob[i][j]+kkk**2
    Prob=Prob/(sum(sum(Prob)))
    Prob=(1/(L**2))-Prob
    return np.array(Prob)