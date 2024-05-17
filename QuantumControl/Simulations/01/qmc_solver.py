import tensorflow as tf
from tensorflow.python.ops import gen_array_ops
import numpy as np


def normalize_probability(psi):
    prob = np.abs(psi) ** 2
    total_prob = np.sum(prob)
    res = psi / (total_prob) ** 0.5
    return res


@tf.function
def to_real(a_c):
    a_r = tf.stack([tf.math.real(a_c), tf.math.imag(a_c)], axis=-1)
    return a_r


@tf.function
def to_complex(a_r):
    k = tf.cast(a_r, tf.complex64)
    a_c = k[..., 0] + 1.0j * k[..., 1]
    return a_c


def eigenstate(n, L):
    points = np.arange(L + 2, dtype=np.complex64)
    k_n = np.pi * n / (L + 1)
    wave = np.sin(k_n * points)  
    wave = normalize_probability(wave)
    return wave[1:-1]

@tf.function
def ip_loss(ar,br):
    a = to_complex(ar)
    b = to_complex(br)
    ip_batch = tf.reduce_sum(tf.math.conj(a)*b,axis=1)
    loss_dp = 1-tf.abs(ip_batch)**2
    loss = tf.reduce_sum(loss_dp)
    return loss


class QMC:
    def __init__(self, dt, Nx):
        self.Nx = Nx
        self.dt = dt
        self.xmin = 0
        self.xmax = 2
        self.L = self.xmax - self.xmin
        self.dx = self.L / (self.Nx + 1)  
        self.step = self.construct_solver()

    def construct_solver(self):

        xmin = self.xmin
        xmax = self.xmax
        Nx = self.Nx
        dx = self.dx
        dt = self.dt
        @tf.function
        def qm_step_batch(psi_batch, control_batch):

            psi_batch = to_complex(psi_batch)

            control=tf.reshape(control_batch,(-1,1))

            therange=tf.reshape(tf.range(xmin, xmax, delta=dx, dtype=tf.float32, name='range')[1:],(1,-1))
            pot_batch=0.5j * dt * tf.cast(tf.tensordot(control,therange,axes=(1,0)),tf.complex64)
            "(batch,spatial)"

            batch_size =  psi_batch.shape[0]
            spatial_size = psi_batch.shape[1]

            alpha_batch = 1.j*(0.5 * dt * tf.ones((batch_size,spatial_size), dtype=tf.complex64) / dx / dx)
            gamma_batch = tf.ones((batch_size,spatial_size), dtype=tf.complex64) - 1.j * dt / dx / dx
            eta_batch = tf.ones((batch_size,spatial_size), dtype=tf.complex64) + 1.j * dt / dx / dx


            U_2_diag = gamma_batch - pot_batch
            U_2_subdiag = alpha_batch
            U_2_stack = tf.stack([U_2_subdiag,U_2_diag,U_2_subdiag],axis=1)
            U_2_batch = gen_array_ops.matrix_diag_v2(U_2_stack, k=(-1, 1), num_rows=-1, num_cols=-1, padding_value=0)

            U_1_diag = eta_batch + pot_batch
            U_1_subdiag = - alpha_batch
            U_1_stack = tf.stack([U_1_subdiag,U_1_diag,U_1_subdiag],axis=1)
            U_1_batch = gen_array_ops.matrix_diag_v2(U_1_stack, k=(-1, 1), num_rows=-1, num_cols=-1, padding_value=0)

            psi_batch_1 = tf.expand_dims(psi_batch,-1)

            b_batch = tf.tensordot(U_2_batch, psi_batch_1,axes=(2,1)) 

            b_batch1 = tf.transpose(b_batch,perm=(1,3,0,2)) 
            b_batch2 = tf.linalg.diag_part(b_batch1) 
            b_batch3 = tf.transpose(b_batch2,perm=(2,0,1)) 

            phi_t_batch = tf.linalg.solve(U_1_batch, b_batch3)[:,:,0]

            phi_t_batch = to_real(phi_t_batch)
            return phi_t_batch

        return qm_step_batch



