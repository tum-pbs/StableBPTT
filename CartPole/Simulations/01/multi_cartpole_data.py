import numpy as np


def generate_data(N,n_poles):
    # x,x_dot,theta,theta_dot

    train_states = np.random.rand(N,n_poles+1,2).astype(np.float32)
    train_states = train_states*2-1
    train_states[:, 1:, 0] = np.pi + train_states[:, 1:,0] * np.pi/6

    test_states = np.random.rand(N,n_poles+1,2).astype(np.float32)
    test_states = test_states*2-1
    test_states[:, 1:, 0] = np.pi + test_states[:, 1:,0] * np.pi/6

    return train_states, test_states
    



