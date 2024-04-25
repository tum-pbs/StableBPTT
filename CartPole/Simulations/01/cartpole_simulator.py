### CARTPOLE ###

import tensorflow as tf

def build_cartpole_step(dt):
    g = 9.8
    m = 0.1 # mass pole
    M = 1.1 #total mass = mass pole + mass cart
    l = 0.5  #length
    ml = m * l 

    @tf.function
    def cartpole_step(state,force):
        """
        multiple poles
        state: batch, object i (cart, pole1,pole2,...), (x,x_dot)
        force: batch, 
            NOT batch,1
        """
        
        #unstack
        cart,poles = tf.split(state,[1,state.shape[1]-1],1)
        x,x_dot = tf.unstack(cart,axis=2)
        thetas,thetas_dot = tf.unstack(poles,axis=2)
        force = tf.expand_dims(force, axis=-1)

        # acceleration
        cos = tf.cos(thetas)
        sin = tf.sin(thetas)
        sintd2 = sin * thetas_dot**2
        A = (force + ml * sintd2) / M
        B = l * (4.0/3.0 - m * cos**2 / M)
        thetas_dot2 = (g * sin - cos * A) / B
        C = tf.reduce_sum(sintd2 - thetas_dot2 * cos,axis=1)
        C = tf.expand_dims(C,axis=-1)
        x_dot2 = (force + ml * C)/M

        # time step
        thetas_dot_new = thetas_dot + dt * thetas_dot2
        thetas_new = thetas + dt * thetas_dot_new
        x_dot_new = x_dot + dt * x_dot2
        x_new = x + dt * x_dot_new

        # stack
        cart_new = tf.stack([x_new,x_dot_new],axis=2)
        poles_new = tf.stack([thetas_new,thetas_dot_new],axis=2)
        state_new = tf.concat([cart_new, poles_new],axis=1)

        return state_new
    
    return cartpole_step



def build_cartpole_loss():

    @tf.function
    def cartpole_loss(states):

        loss_theta = 1-tf.reduce_mean(tf.cos(states[..., 1:,0]))

        return loss_theta
    
    return cartpole_loss


