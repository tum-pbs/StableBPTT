import numpy as np
from qmc_solver import *


def generate_data(N,Nx,weighting = '1-1'):
    es1 = eigenstate(1, Nx).reshape((1, -1))
    es2 = eigenstate(2, Nx).reshape((1, -1))

    if weighting == '1-1':
        w1,w2 = 1,1
    elif weighting == '3-1':
        w1,w2 = 3,1
    elif weighting == '7-1':
        w1,w2 = 7,1
    elif weighting == '15-1':
        w1,w2 = 15,1

    c1 = np.random.normal(size=(N,1)).astype(np.float32)*w1
    c2 = np.random.normal(size=(N,1)).astype(np.float32)*w2
    p1 = np.random.uniform(0,2*np.pi,size=(N,1)).astype(np.float32)
    p2 = np.random.uniform(0,2*np.pi,size=(N,1)).astype(np.float32)
    norm = np.sqrt(c1**2+c2**2)
    f1 = c1 *np.exp(1.j*p1)/norm
    f2 = c2 *np.exp(1.j*p2)/ norm

    states = tf.tensordot(f1,es1,axes=(1,0))+tf.tensordot(f2,es2,axes=(1,0))
    states = to_real(states)
    n2 = N//2
    train = states[:n2]
    test= states[n2:]
    return train,test
