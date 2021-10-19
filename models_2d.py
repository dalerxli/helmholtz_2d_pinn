import tensorflow as tf
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Concatenate, Lambda
from keras.initializers import RandomUniform


# Mappings
def bessel_mapping(input_X, k, N, M):

    R = tf.norm(input_X, axis=1)
    R = tf.reshape(R, [-1, 1])
    T = tf.math.atan2(input_X[:, 1:2], input_X[:, 0:1])

    J = np.arange(M)
    SIN_j = tf.math.sin(np.pi * J / M)
    SIN_j = tf.reshape(SIN_j, [1, -1])
    SIN_j = tf.cast(SIN_j, tf.float32)
    kR_SIN_j = tf.linalg.matmul(a=k * R, b=SIN_j)
    X = tf.math.cos(kR_SIN_j)

    for n in range(1, N + 1):
        COS_n = tf.math.cos(n * T)
        SIN_n = tf.math.sin(n * T)
        nJ = n * J * np.pi / M
        nJ = tf.cast(nJ, tf.float32)
        X_ = tf.math.cos(nJ - kR_SIN_j)

        X = Concatenate()([X, X_, COS_n, SIN_n])

    # for n in range(-N, 0):
    #     nJ = n * J * np.pi / M
    #     nJ = tf.cast(nJ, tf.float32)
    #     X_ = tf.math.cos(nJ - kR_SIN_j)
    #
    #     X = Concatenate()([X, X_])

    return X


def fourier_feature_mapping(input_X, M=20, mean=0., stddev=1.):

    input_dim = input_X.shape[-1]
    B = tf.random.normal(shape=[M, input_dim], mean=mean,  stddev=stddev)
    B = tf.cast(B, tf.float32)
    BX = tf.linalg.matmul(a=input_X, b=B, transpose_b=True)
    FFX1 = tf.math.cos(2 * np.pi * BX)
    FFX2 = tf.math.sin(2 * np.pi * BX)

    X = Concatenate()([FFX1, FFX2])

    return X


# Models
def mapping_model(input_dim, output_dim, width, depth, k=1., N=2, M=20):

    # Initialize Sequential Model
    input_X = Input(shape=input_dim)

    X = bessel_mapping(input_X, k, N, M)

    new_width = X.shape[-1]

    w_0 = 2.
    limit_0 = 1. / (2 * new_width)
    initializer_first_layer = RandomUniform(minval=-w_0 * limit_0,
                                            maxval=+w_0 * limit_0)
    w_1 = 1.
    limit_1 = np.sqrt(6. / width) / w_1
    initializer = RandomUniform(minval=-limit_1, maxval=limit_1)

    act_fun = tf.math.sin

    X = Dense(width,
              activation=act_fun,
              use_bias=True,
              kernel_initializer=initializer_first_layer,
              bias_initializer=initializer)(X)

    for _ in range(depth - 1):
        X = Dense(units=width,
                  activation=act_fun,
                  use_bias=True,
                  kernel_initializer=initializer,
                  bias_initializer=initializer)(X)
        if w_1 != 1.:
            X = Lambda(lambda x: x * w_1)(X)

    # Output layer
    Y = Dense(units=output_dim,
              activation=None,
              use_bias=True,
              kernel_initializer=initializer,
              bias_initializer=initializer)(X)

    model = Model(inputs=input_X, outputs=Y)

    return model
