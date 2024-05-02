import numpy as np

def generate_data(Nd, n_dr,n_ev):
    N=2*Nd
    np.random.seed(42)
    states = np.zeros((N,n_dr+n_ev,2,2)).astype(np.float32)

    def rectangle(N,xmin,xmax,ymin,ymax):
        x = np.random.rand(N,).astype(np.float32)
        y = np.random.rand(N,).astype(np.float32)
        x = (xmax-xmin) * x + xmin
        y = (ymax-ymin) * y + ymin
        return np.stack([x,y],axis=-1)
    
    def annulus(N,rmin,rmax):
        s1 = np.random.rand(N,).astype(np.float32)
        s2 = np.random.rand(N,).astype(np.float32)
        phi = 2 * np.pi * s1
        r =(rmax-rmin) * s2 + rmin
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.stack([x,y],axis=-1)


    # batch object x/v spatial

    #drivers
    for k in range(n_dr):
        states[:,k,0,:] = annulus(N,3,4)
    
    #evaders
    com = annulus(N,1,2)
    for k in range(n_ev):
        states[:,n_dr+k,0,:] = com+rectangle(N,-0.5,0.5,-0.5,0.5)

    return states[:Nd],states[Nd:]
