import tensorflow as tf

'''
x
0: batch
1: number 
2: x , v
3: x,y direction

dimensions: None,2,2,2
batch,number(de),x/v,dimension
'''

class DRIVER_EVADER_SYSTEM():
    def __init__(self,n_drivers,n_evaders,epsilon=0.1) -> None:
        self.n_drivers = n_drivers
        self.n_evaders = n_evaders
        self.epsilon = epsilon

    def set_evader_parameters(self, c_evade, c_fric, c_flock,d_flock):
        self.c_evade = c_evade
        self.c_fric_ev = c_fric
        self.c_flock = c_flock
        self.d_flock = d_flock

    def set_driver_parameters(self,c_drive, c_fric, c_spread):
        self.c_drive = c_drive
        self.c_spread = c_spread
        self.c_fric_dr = c_fric

    def construct_simulator(self,dt):

        def dist(x):
            return self.epsilon + tf.reduce_sum(x**2,-1)**0.5
        
        def get_orthog(x):
            x1,x2=tf.unstack(x,axis=-1)
            return tf.stack([-x2,x1],axis=-1)
        
        def center_diff(x_drivers,x_evaders):
            center = tf.expand_dims(tf.reduce_sum(x_evaders,axis=1),axis=1)
            return x_drivers-center
        
        ### CAUSES EVADERS TO RUN AWAY FROM DRIVERS

        def evading_function(d):
            p = -1.0 * d ** -2
            return p
        
        def F_evading(x_driver,x_evaders):
            relative_position = tf.expand_dims(x_driver,1) - x_evaders
            d = dist(relative_position) 
            F = tf.einsum('ij,ijk->ijk',evading_function(d),relative_position)
            return F

        def F_evading_total(x_drivers,x_evaders):
            F = 0
            for i in range(self.n_drivers):
                F += F_evading(x_drivers[:,i,:],x_evaders)
            return F
        
        ### CAUSES DRIVERS TO STAY CLOSE EVADERS BUT NOT TOO CLOSE
        
        def driving_function(d):
            r = 2 + 3 * d ** -2 - 4 *d **-4
            return -r
        
        def F_driving(x_drivers,x_evader):
            relative_position = x_drivers - tf.expand_dims(x_evader,1)
            d = dist(relative_position) 
            F = tf.einsum('ij,ijk->ijk',driving_function(d),relative_position)
            return F
        
        def F_driving_total(x_drivers,x_evaders):
            F = 0
            for i in range(self.n_evaders):
                F += F_driving(x_drivers,x_evaders[:,i,:])
            return F

        ### CAUSES DRIVERS TO SPREAD
        
        def spreading_function(d):
            r = d**-2
            return r
        
        def F_spreading(x_drivers,x_driver):
            relative_position = x_drivers - tf.expand_dims(x_driver,1)
            d = dist(relative_position) 
            F = tf.einsum('ij,ijk->ijk',spreading_function(d),relative_position)
            return F
        
        def tf_delete(array,entry):
            a = array[:,:entry,:]
            b = array[:,entry+1:,:]
            c = tf.concat([a,b],axis=1)
            return c
        
        def tf_insert(array,entry):
            a = array[:,:entry,:]
            b = array[:,entry:,:]
            c = tf.zeros(([b.shape[0],1,b.shape[2]]),dtype=tf.float32)
            d = tf.concat([a,c,b],axis=1)
            return d
        
        def F_spreading_total(x_drivers):
            F = 0
            for i in range(self.n_drivers):
                F_i = F_spreading(tf_delete(x_drivers,i),x_drivers[:,i,:])
                F += tf_insert(F_i,i)
            return F
        
        # CAUSES EVADERS TO STAY CLOSE TO EACH OTHER BUT NOT TOO CLOSE 

        def flock_function(d):
            p = (self.d_flock / d - 1)/d
            return p

        def F_flock(x_evaders,x_evader_i):
            relative_position = tf.expand_dims(x_evader_i,1) - x_evaders
            d = dist(relative_position) 
            F = tf.einsum('ij,ijk->ijk',flock_function(d),relative_position)
            return F

        def F_flock_total(x_evaders):
            F = 0
            for i in range(self.n_evaders):
                F_i = F_flock(tf_delete(x_evaders,i),x_evaders[:,i,:])
                F += tf_insert(F_i,i)
            return F

        def ode(u,c):

            # u 0 1 2 3
            #   batch, object, x/v, spatial
            # c 0 1(smaller,only drivers) 3
            x,v = tf.unstack(u,2,2) # 0 1 3
            x_drivers, x_evaders = tf.split(x,[self.n_drivers,self.n_evaders],1) # 0 3
            v_drivers, v_evaders = tf.split(v,[self.n_drivers,self.n_evaders],1)

            center_vec = center_diff(x_drivers,x_evaders)
            orthog_vec = get_orthog(center_vec)
            F_con = c[...,:1]* center_vec + c[...,1:]*orthog_vec

            F_ev        = self.c_evade * F_evading_total(x_drivers,x_evaders)
            F_fric_ev   = - self.c_fric_ev * v_evaders
            F_flock     = self.c_flock * F_flock_total(x_evaders)
            F_dr        = 0.0 
            F_fric_dr   = - self.c_fric_dr * v_drivers
            F_spread    = self.c_spread * F_spreading_total(x_drivers)


            x_drivers_dot = v_drivers
            x_evaders_dot = v_evaders
            v_drivers_dot = F_con + F_dr + F_fric_dr + F_spread 
            v_evaders_dot = F_ev + F_fric_ev + F_flock

            x_dot = tf.concat([x_drivers_dot,x_evaders_dot],axis=1)
            v_dot = tf.concat([v_drivers_dot,v_evaders_dot],axis=1)
            u_dot = tf.stack([x_dot,v_dot],2)
            return u_dot 

        @tf.function
        def euler_step(u,c):
            u_dot = ode(u,c)
            u_new = u + u_dot * dt
            return u_new

        return euler_step

    def construct_loss_function(self,target,n_dr):
        
        target = tf.reshape(tf.constant([0.0,0.0]),(1,1,1,2))

        @tf.function
        def de_loss(states):
            # batch time object x/v spatial
            x_evaders = states[:,:,self.n_drivers:,0,:] 
            diff = x_evaders - target
            l = tf.reduce_mean(diff**2)
            return l

        return de_loss