import tensorflow as tf
from qmc_solver import to_complex,eigenstate
import numpy as np
stop = tf.stop_gradient

class LOSS_FORMULATION():
    def __init__(self,network,solver,Nt,Nx,target_state,gradient_flow,loss_type):
        self.solver = solver
        self.network = network
        self.Nt = Nt
        self.Nx = Nx
        self.target_state = target_state
        self.loss_type = loss_type
        if type(gradient_flow) is list:
            self.gradient_flow = gradient_flow
        else:
            self.gradient_flow = [gradient_flow] * Nt
        self.compute_loss = self.construct_compute_loss()

    def physics_flow(self,x0,gradient_flow):
        if gradient_flow in ['F', 'P']:#['full', 'physics']
            x0b = x0 
        elif gradient_flow in ['N','S']:#['network','stop']
            x0b = stop(x0) 
        return x0b

    def control_flow(self,x0,gradient_flow):
        if gradient_flow in ['F', 'N']:
            c1 = self.network(x0)
        elif gradient_flow in ['P','S']:
            c1 = self.network(stop(x0))
        return c1

    def time_evolution(self,x0):
        xs = [x0]
        for n in range(self.Nt):
            cn = self.control_flow(xs[-1],self.gradient_flow[n])
            xnb = self.physics_flow(xs[-1],self.gradient_flow[n])
            xn = self.solver(xnb,cn)
            xs.append(xn)
        xs = tf.stack(xs,axis=1)
        return xs
    
    def ip_loss(self,ar,br):
        a = to_complex(ar)
        b = to_complex(br)
        ip_batch_time = tf.reduce_sum(tf.math.conj(a)*b,axis=-1)
        loss_batch_time = 1-tf.abs(ip_batch_time)**2
        loss_batch = tf.reduce_sum(loss_batch_time,axis=-1)
        loss = tf.reduce_mean(loss_batch)
        return loss
    
    def construct_compute_loss(self):
        psi2 = eigenstate(self.target_state, self.Nx).reshape((1, -1))
        es2 = np.stack([psi2, psi2 * 0], axis=-1)
        
        if self.loss_type=="CONTINUOUS":
            @tf.function
            def qm_compute_loss(x0):
                xs = self.time_evolution(x0)
                xp = xs[:,1:,:,:]
                l = self.ip_loss(xp,es2)
                return l
            
            return qm_compute_loss
        
        elif self.loss_type=="FINAL":
            @tf.function
            def qm_compute_loss_final(x0):
                xs = self.time_evolution(x0)
                xp = xs[:,-1:,:,:]
                l = self.ip_loss(xp,es2)
                return l
        
            return qm_compute_loss_final
    


    


