import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Loss functions
def loss_piston(k, uR, uI, uR_X, uI_X, uR_XX, uI_XX,
                uR_b, uI_b, uR_X_b, uI_X_b, X_b,
                uR_b0, uI_b0, X_b0,
                fR = 0, fI = 0,
                C0 = 1., C1 = 1., C2 = 0.1, C3 = 0.1, C4 = 0.1, C5 = 0.1):

    loss_int = C0 * tf.reduce_mean(tf.square((uR_XX[:, 0:1] + uR_XX[:, 1:2])/k**2 + uR - fR)) \
               + C1 * tf.reduce_mean(tf.square((uI_XX[:, 0:1] + uI_XX[:, 1:2])/k**2 + uI - fI))

    loss_b = C2 * tf.reduce_mean(tf.square((uR_X_b[:, 0:1] * X_b[:, 0:1] + uR_X_b[:, 1:2] * X_b[:, 1:2]) / k + uI_b)) \
             + C3 * tf.reduce_mean(tf.square((uI_X_b[:, 0:1] * X_b[:, 0:1] + uI_X_b[:, 1:2] * X_b[:, 1:2]) / k - uR_b))

    T = tf.math.atan2(X_b0[:, 1:2], X_b0[:, 0:1])
    BC = np.where(np.abs(T) < np.pi / 6, 1, 0)

    loss_b0 = C4 * tf.reduce_mean(tf.square(uR_b0 - BC)) \
              + C5 * tf.reduce_mean(tf.square(uI_b0))

    loss = loss_int + loss_b + loss_b0

    return loss_int, loss_b, loss_b0, loss


def loss_dipole(k, uR, uI, uR_X, uI_X, uR_XX, uI_XX,
                uR_b, uI_b, uR_X_b, uI_X_b, X_b,
                uR_b0, uI_b0, X_b0,
                fR = 0, fI = 0,
                C0 = 1., C1 = 1., C2 = 0.1, C3 = 0.1, C4 = 0.1, C5 = 0.1):

    loss_int = C0 * tf.reduce_mean(tf.square((uR_XX[:, 0:1] + uR_XX[:, 1:2])/k**2 + uR - fR)) \
               + C1 * tf.reduce_mean(tf.square((uI_XX[:, 0:1] + uI_XX[:, 1:2])/k**2 + uI - fI))

    loss_b = C2 * tf.reduce_mean(tf.square((uR_X_b[:, 0:1] * X_b[:, 0:1] + uR_X_b[:, 1:2] * X_b[:, 1:2]) / k + uI_b)) \
             + C3 * tf.reduce_mean(tf.square((uI_X_b[:, 0:1] * X_b[:, 0:1] + uI_X_b[:, 1:2] * X_b[:, 1:2]) / k - uR_b))

    R = np.sqrt(X_b0[:, 0:1] ** 2 + X_b0[:, 1:2] ** 2)
    COS = X_b0[:, 0:1] / R

    loss_b0 = C4 * tf.reduce_mean(tf.square(uR_b0 - COS)) \
              + C5 * tf.reduce_mean(tf.square(uI_b0))

    loss = loss_int + loss_b + loss_b0

    return loss_int, loss_b, loss_b0, loss


def loss_monopole(k, uR, uI, uR_X, uI_X, uR_XX, uI_XX,
                  uR_b, uI_b, uR_X_b, uI_X_b, X_b,
                  uR_b0, uI_b0, X_b0,
                  fR = 0, fI = 0,
                  C0 = 1., C1 = 1., C2 = 0.1, C3 = 0.1, C4 = 0.1, C5 = 0.1):

    loss_int = C0 * tf.reduce_mean(tf.square((uR_XX[:, 0:1] + uR_XX[:, 1:2])/k**2 + uR - fR)) \
             + C1 * tf.reduce_mean(tf.square((uI_XX[:, 0:1] + uI_XX[:, 1:2])/k**2 + uI - fI))

    loss_b = C2 * tf.reduce_mean(tf.square((uR_X_b[:, 0:1] * X_b[:, 0:1] + uR_X_b[:, 1:2] * X_b[:, 1:2]) / k + uI_b)) \
           + C3 * tf.reduce_mean(tf.square((uI_X_b[:, 0:1] * X_b[:, 0:1] + uI_X_b[:, 1:2] * X_b[:, 1:2]) / k - uR_b))

    loss_b0 = C4 * tf.reduce_mean(tf.square(uR_b0 - 1)) \
            + C5 * tf.reduce_mean(tf.square(uI_b0))

    loss = loss_int + loss_b + loss_b0

    return loss_int, loss_b, loss_b0, loss


# Uniformly distributed random domain
def get_training_points(N=10000, N_b=500, r0=0., plot_dataset=True):

    t = tf.reshape(tf.linspace(0., 2*np.pi, N_b), [N_b, 1])
    x1_b = tf.cos(t)
    x2_b = tf.sin(t)
    X_b = tf.concat([x1_b, x2_b], axis=1)

    if r0 > 0.:
        N_b0 = round(N_b * r0)
        t = tf.reshape(tf.linspace(0., 2*np.pi, N_b0), [N_b0, 1])
        x1_b0 = r0 * tf.cos(t)
        x2_b0 = r0 * tf.sin(t)
        X_b0 = tf.concat([x1_b0, x2_b0], axis=1)
    else:
        X_b0 = []

    N_ = round(N * 4 / np.pi * (1 - r0 ** 2))
    X = tf.random.uniform(shape=[N_, 2], minval=-1, maxval=1)
    R = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)

    mask = [all(tup) for tup in zip(R > r0, R < 1)]
    X = tf.boolean_mask(X, mask)

    if plot_dataset:
        # Plot Domain
        x1 = X[:, 0:1]
        x2 = X[:, 1:2]
        x1_b = X_b[:, 0:1]
        x2_b = X_b[:, 1:2]
        x1_b0 = X_b0[:, 0:1]
        x2_b0 = X_b0[:, 1:2]

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('Training points', fontsize=16)
        axs[0].scatter(x1, x2, 1)
        axs[0].set_title('Interior Domain Points', fontsize=14)
        axs[0].set_xlabel('x', fontsize=12)
        axs[0].set_ylabel('y', fontsize=12)
        axs[1].scatter(x1_b, x2_b, 0.1, 'r')
        axs[1].scatter(x1_b0, x2_b0, 0.1, 'r')
        axs[1].set_title('Boundary Points', fontsize=14)
        axs[1].set_xlabel('x', fontsize=12)
        axs[1].set_ylabel('y', fontsize=12)

    return X, X_b, X_b0
