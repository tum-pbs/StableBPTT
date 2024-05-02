import tensorflow as tf


def build_fully_connected_network(width, depth, use_bias, use_zero_initialization,n_dr,n_ev):

    activation = tf.keras.activations.tanh

    layers = []
    layers.append(tf.keras.layers.InputLayer(input_shape=(n_dr+n_ev,2,2)))
    layers.append(tf.keras.layers.Reshape((-1,)))
    for _ in range(depth):
        layers.append(tf.keras.layers.Dense(
            width, activation=activation, use_bias=use_bias))
    layers.append(tf.keras.layers.Dense(
        2*n_dr, activation='linear', use_bias=use_bias))
    layers.append(tf.keras.layers.Reshape((1,n_dr,2)))

    model = tf.keras.models.Sequential(layers)
    model.summary()

    if use_zero_initialization == True:
        if use_bias == True:
            last_weights_index = -2
        else:
            last_weights_index = -1

        weights = model.get_weights()
        weights[last_weights_index] = 0.0 * weights[last_weights_index]
        model.set_weights(weights)
    return model
