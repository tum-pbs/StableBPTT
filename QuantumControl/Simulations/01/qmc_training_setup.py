from qmc_solver import *
from qmc_loss_formulation import *
import numpy as np
import time 

def combine_grads(grad_i_base,grad_i_comp):
    sign_base = tf.math.sign(grad_i_base)
    sign_comp = tf.math.sign(grad_i_comp)
    cond = sign_base==sign_comp
    return tf.where(cond,grad_i_base,0)

def combine_grad_lists(grad_base,grad_comp):
    result = []
    for i in range(len(grad_base)):
        combined_i = combine_grads(grad_base[i],grad_comp[i])
        result.append(combined_i)
    return result
    
@tf.function
def get_grad_size(grad):
    grad_size = tf.constant(0.0, dtype=tf.float32)
    for weight in grad:
        grad_size += tf.reduce_sum(weight**2)
    return grad_size

def flatten(tensor_list):
    flat_list = []
    for tensor in tensor_list:
        flat_list.append(tf.reshape(tensor,(-1,)))
    return tf.concat(flat_list,axis=0)

def norm(v):
    return tf.reduce_sum(v*v)**0.5

def ip(v,w):
    return tf.reduce_sum(v*w)

@tf.function
def cosine_similarity(grad1,grad2):
    v = flatten(grad1)
    w = flatten(grad2)
    return ip(v,w)/(norm(v)*norm(w))



class TRAINING_SETUP():
    def __init__(self,data,controller,simulator,Nt,Nx,target_state,loss_type,gfm,opt_p):
        
        gf_dict = {
            'F':[1,'F'],
            'P':[1,'P'],
            'S':[1,'S'],
            'C':[2,'P','F'],
        }
        gf_p = gf_dict[gfm]
        
        
        self.dataloader,self.train_data,self.test_data=data
        self.controller = controller

        self.nb = self.dataloader.cardinality()
        self.nep = opt_p['ep']

        opt_dict = {
            'ADAM':tf.keras.optimizers.Adam,
            'SGD':tf.keras.optimizers.SGD,
            'SGDMOM080':lambda lr, **kwargs: tf.keras.optimizers.SGD(lr, momentum=0.8, **kwargs),
            'SGDMOM090':lambda lr, **kwargs: tf.keras.optimizers.SGD(lr, momentum=0.9, **kwargs),
            'SGDMOM095':lambda lr, **kwargs: tf.keras.optimizers.SGD(lr, momentum=0.95, **kwargs),
            'ADAGRAD':tf.keras.optimizers.Adagrad,
            'ADADELTA':tf.keras.optimizers.Adadelta,
            'RMSPROP':tf.keras.optimizers.RMSprop
        }
        opt = opt_dict[opt_p['opt']]

        clip_dict = {
            'NONE': opt(opt_p['lr']),
            'VALUE':opt(opt_p['lr'], clipvalue=opt_p['cnum']),
            'NORM':opt(opt_p['lr'],global_clipnorm=opt_p['cnum'])
        }
        self.optimizer = clip_dict[opt_p['cmod']]

        self.results = []

        if gf_p[0]==1:
            self.LF1=LOSS_FORMULATION(controller,simulator,Nt,Nx,target_state,gf_p[1],loss_type)
            self.update = self.update_1bp

        if gf_p[0]==2:
            self.LF1=LOSS_FORMULATION(controller,simulator,Nt,Nx,target_state,gf_p[1],loss_type)
            self.LF2=LOSS_FORMULATION(controller,simulator,Nt,Nx,target_state,gf_p[2],loss_type)
            self.update = self.update_2bp

    @tf.function
    def update_1bp(self,batch_states):
        with tf.GradientTape() as tape:
            loss = self.LF1.compute_loss(batch_states)
        grad = tape.gradient(loss,self.controller.variables)

        self.optimizer.apply_gradients(zip(grad, self.controller.variables))
        return loss,[grad]
    
    @tf.function
    def update_2bp(self,batch_states):
        with tf.GradientTape() as tape:
            loss = self.LF1.compute_loss(batch_states)
        grad_base = tape.gradient(loss,self.controller.variables)

        with tf.GradientTape() as tape:
            loss = self.LF2.compute_loss(batch_states)
        grad_comp = tape.gradient(loss,self.controller.variables)

        grad = combine_grad_lists(grad_base,grad_comp)

        self.optimizer.apply_gradients(zip(grad, self.controller.variables))
        return loss,[grad,grad_base,grad_comp]


    def mini_batch_update(self,ys):

        t0 = time.time()
        loss,grads = self.update(ys)
        t1 = time.time()-t0

        grad_size = get_grad_size(grads[0])

        if len(grads)>1:
            cossim = cosine_similarity(grads[1],grads[2])
        else:
            cossim = 0.0

        return loss, grad_size,cossim, t1

    def epoch_update(self, i):
        lb,gb,cb,tb = [],[],[],[]
        for j, states in enumerate(self.dataloader):
            loss, grad_size,cossim, t1 = self.mini_batch_update(states)
            lb.append(loss)
            gb.append(grad_size)
            cb.append(cossim)
            tb.append(t1)

        l = tf.reduce_mean(lb)
        g = tf.reduce_mean(gb)
        c = tf.reduce_mean(cb)
        t = tf.reduce_sum(tb)

        l_train = self.LF1.compute_loss(self.train_data)
        l_test = self.LF1.compute_loss(self.test_data)

        return l, l_train,l_test,g,c,t

    def run(self):
        for i in range(self.nep):
            l, l_train,l_test,g,c,t = self.epoch_update(i)
            tf.print('Epoch: ', i, ' Loss: ', l,
                     ' Grad :', g, ' Epoch Time :', t)
            self.results.append([l, l_train,l_test,g,c,t])
        return np.array(self.results)
