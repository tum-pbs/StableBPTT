import tensorflow as tf

stop = tf.stop_gradient


class LOSS_FORMULATION():
    def __init__(self, simulator_time_step, controller, loss_function, 
                 Nt, loss_mode, regu_coef,gradient_flow):
        self.simulator_time_step = simulator_time_step
        self.controller = controller
        self.loss_function = loss_function
        self.Nt = Nt
        self.loss_mode = loss_mode
        self.gradient_flow = gradient_flow
        self.regu_coef = regu_coef

    def physics_flow(self,state):
        if self.gradient_flow in ['F', 'P']:#['full', 'physics']
            stateb = state
        elif self.gradient_flow in ['N','S']:#['network','stop']
            stateb = stop(state) 
        return stateb
    
    def control_flow(self, state):
        if self.gradient_flow in ['F', 'N']:
            control = self.controller(state)
        elif self.gradient_flow in ['P', 'S']:
            control = self.controller(stop(state))
        return control[:,0]

    @tf.function
    def time_evolution(self, initial_state):
        states = [initial_state]
        forces = []
        for n in range(1, self.Nt+1):
            control_n = self.control_flow(states[-1])
            state_nb = self.physics_flow(states[-1])
            state_n = self.simulator_time_step(state_nb, control_n)
            forces.append(control_n)
            states.append(state_n)
        states = tf.stack(states, axis=1)
        forces = tf.stack(forces, axis=1)
        return states, forces

    def build_loss(self,):

        if self.loss_mode == 'FINAL':
            def choose_states(states): return states[:, -1:, :]
        elif self.loss_mode == 'CONTINUOUS':
            def choose_states(states): return states[:, 1:, :]

        def full_loss(states, forces):
            chosen_states = choose_states(states)
            physics_loss = self.loss_function(chosen_states)
            regularization_loss = tf.reduce_mean(forces**2)
            loss = physics_loss + self.regu_coef * regularization_loss
            return loss, physics_loss

        @tf.function
        def compute_loss(initial_states):
            states, forces = self.time_evolution(initial_states)
            loss, physics_loss = full_loss(states, forces)
            return loss, physics_loss

        return compute_loss
