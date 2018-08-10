"""
Inspired by:
https://github.com/XifengGuo/CapsNet-Keras/


"""
from keras import activations
from keras.layers import Dense
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


def dynamic_routing(routings, activation, u_hat):
    u_hat = K.permute_dimensions(u_hat, (0, 2, 1, 3))
    b = K.zeros_like(u_hat[:,:,:,0])
    assert routings > 0, 'The routings should be > 0.'
    for i in range(routings):
        Wv = softmax(b, 1)

        vj = K.batch_dot(Wv, u_hat, [2, 2])
        if i < routings - 1:
            vj = K.l2_normalize(vj)

            b = K.batch_dot(vj, u_hat, [2, 3])
    return activation(vj)


def frob_norm(u_hat):
    return K.sqrt(K.sum(u_hat*u_hat))

def opt_dynamic_routing(routings, activation, u_hat, weight):

    u_hat = K.permute_dimensions(u_hat, (0, 2, 1, 3))
    b = K.zeros_like(u_hat[:,:,:,0])
    a = 1

    for i in range(routings):
             # Instead of frob norm, use l2 regularized marginalizations
        oji = K.sum(K.l2_normalize(u_hat)) * u_hat
        if i < 1:
            cij = b
        sjh = K.batch_dot(cij, oji, [2, 2])
        sj = K.l2_normalize(sjh)
        cij = softmax(a**(-1) * K.batch_dot(sj, oji, [2, 3]), 1)


    return activation(sjh, epsilon=1)

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.activation = squash

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        self.W = self.add_weight(name='capsule_kernel', shape=(1, input_dim_capsule,
                                self.num_capsule * self.dim_capsule),
                                 initializer='glorot_uniform',
                                 trainable=True)
    def call(self, u):
        batch_size, input_num_capsule = K.shape(u)[0], K.shape(u)[1]
        # u_hat1 = K.
        u_hat = K.conv1d(u, self.W)
        u_hat = K.reshape(u_hat, (batch_size, input_num_capsule,
                            self.num_capsule, self.dim_capsule))
        # return dynamic_routing(self.routings, self.activation, u_hat)
        return opt_dynamic_routing(self.routings, self.activation, u_hat, self.W)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def squash(x, epsilon=0.5, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (epsilon + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)
