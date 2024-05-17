import tensorflow as tf

def get_cnn(Nx,non,bias,zinit=True):
    act = tf.keras.activations.tanh
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(Nx,2)),
        tf.keras.layers.Reshape((Nx,2,1)),
        tf.keras.layers.Conv2D(non,(3,2),2, activation=act),
        tf.keras.layers.Reshape((Nx//2-1,-1)),
        tf.keras.layers.Conv1D(non,3,2, activation=act),
        tf.keras.layers.Reshape((-1,)),
        tf.keras.layers.Dense(1, activation='linear',use_bias=bias)
    ])
    weights = model.get_weights()
    if zinit==True:
        w = weights[-1]
        wnew = w * 0.0
        weights[-1]=wnew
        model.set_weights(weights)
    init_weights = model.get_weights()
    return model


