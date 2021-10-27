import tensorflow as tf
from tensorflow import keras
import numpy as np
from models_2d import mapping_model
from loss_funs import loss_dipole, loss_piston, get_training_points
from make_plots import plot_results


def loss_piston_adapt(k, uR, uI, uR_X, uI_X, uR_XX, uI_XX,
                      uR_b, uI_b, uR_X_b, uI_X_b, X_b,
                      uR_b0, uI_b0, X_b0,
                      fR=0, fI=0):
    loss_int = tf.reduce_mean(tf.square((uR_XX[:, 0:1] + uR_XX[:, 1:2]) / k ** 2 + uR - fR)) \
               + tf.reduce_mean(tf.square((uI_XX[:, 0:1] + uI_XX[:, 1:2]) / k ** 2 + uI - fI))

    loss_b1_r = tf.reduce_mean(tf.square((uR_X_b[:, 0:1] * X_b[:, 0:1] + uR_X_b[:, 1:2] * X_b[:, 1:2]) / k + uI_b))
    loss_b1_i = tf.reduce_mean(tf.square((uI_X_b[:, 0:1] * X_b[:, 0:1] + uI_X_b[:, 1:2] * X_b[:, 1:2]) / k - uR_b))

    T = tf.math.atan2(X_b0[:, 1:2], X_b0[:, 0:1])
    BC = np.where(np.abs(T) < np.pi / 6, 1, 0)

    loss_b0_r = tf.reduce_mean(tf.square(uR_b0 - BC))
    loss_b0_i = tf.reduce_mean(tf.square(uI_b0))

    loss_b = [loss_b1_r, loss_b1_i, loss_b0_r, loss_b0_i]

    return loss_int, loss_b


# Model
input_dim = 2
output_dim = 2
width = 20
depth = 5
optimizer = keras.optimizers.Adam(learning_rate=3e-3)

TOL = 5e-5
max_epochs = 1000

K = 2 * np.pi * np.array([2])
r0 = 0.1
X, X_b, X_b0 = get_training_points(N=10000, N_b=1000, r0=r0)

PINN_list = []

for k in K:

    model_pinn = mapping_model(input_dim, output_dim, width, depth, k=k, N=6)

    print('------------------------------------')
    print('Wavenumber k = {} * pi'.format(round(k / np.pi)))
    print(' ')

    history_loss_int = np.array([])
    history_loss_b = np.array([])
    history_loss = np.array([])

    loss = TOL + 1
    epoch = 0
    lambda_b = []
    alpha = 0.8
    while loss > TOL and epoch < max_epochs:

        with tf.GradientTape(persistent=True) as g1:
            g1.watch(X)
            g1.watch(X_b)
            g1.watch(X_b0)
            with tf.GradientTape(persistent=True) as g2:
                g2.watch(X)
                g2.watch(X_b)
                g2.watch(X_b0)
                with tf.GradientTape(persistent=True) as g3:
                    g3.watch(X)
                    g3.watch(X_b)
                    g3.watch(X_b0)
                    u = model_pinn(X, training=True)
                    u_b = model_pinn(X_b, training=True)
                    u_b0 = model_pinn(X_b0, training=True)
                    uR = u[:, 0:1]
                    uI = u[:, 1:2]
                    uR_b = u_b[:, 0:1]
                    uI_b = u_b[:, 1:2]
                    uR_b0 = u_b0[:, 0:1]
                    uI_b0 = u_b0[:, 1:2]
                uR_X = g3.gradient(uR, X)
                uI_X = g3.gradient(uI, X)
                uR_X_b = g3.gradient(uR_b, X_b)
                uI_X_b = g3.gradient(uI_b, X_b)
                uR_x = uR_X[:, 0:1]
                uR_y = uR_X[:, 1:2]
                uI_x = uI_X[:, 0:1]
                uI_y = uI_X[:, 1:2]
            uR_xx = g2.gradient(uR_x, X)[:, 0:1]
            uR_yy = g2.gradient(uR_y, X)[:, 1:2]
            uR_XX = tf.concat([uR_xx, uR_yy], axis=1)
            uI_xx = g2.gradient(uI_x, X)[:, 0:1]
            uI_yy = g2.gradient(uI_y, X)[:, 1:2]
            uI_XX = tf.concat([uI_xx, uI_yy], axis=1)

            loss_int, loss_b = loss_piston_adapt(k, uR, uI, uR_X, uI_X, uR_XX, uI_XX,
                                                 uR_b, uI_b, uR_X_b, uI_X_b, X_b,
                                                 uR_b0, uI_b0, X_b0)

            if len(lambda_b) == 0:
                lambda_b = [0.1] * len(loss_b)

            with g1.stop_recording():
                grad_int = g1.gradient(loss_int, model_pinn.trainable_variables)
                max_grad_int_list = []
                for i in range(depth):
                    max_grad_int_list.append(tf.reduce_max(tf.abs(grad_int[2 * i])))
                max_grad_int = tf.reduce_max(tf.stack(max_grad_int_list))

            loss = loss_int
            for i in range(len(loss_b)):
                loss_b_i = loss_b[i]
                with g1.stop_recording():
                    grad_b_i = g1.gradient(loss_b_i, model_pinn.trainable_variables)
                    mean_grad_b_i_list = []
                    for j in range(depth):
                        mean_grad_b_i_list.append(tf.reduce_mean(tf.abs(grad_b_i[2 * j])))
                    mean_grad_b_i = tf.reduce_mean(tf.stack(mean_grad_b_i_list))
                    lambda_i_hat = max_grad_int / mean_grad_b_i

                lambda_b[i] = (1 - alpha) * lambda_b[i] + alpha * lambda_i_hat
                loss += lambda_b[i] * loss_b_i

        grads = g1.gradient(loss, model_pinn.trainable_variables)

        optimizer.apply_gradients(zip(grads, model_pinn.trainable_variables))

        history_loss_int = np.append(history_loss_int, loss_int)
        history_loss_b = np.append(history_loss_b, np.sum(loss_b))
        history_loss = np.append(history_loss, loss)

        epoch += 1
        if epoch % 10 == 0:
            loss_ = loss_int + np.sum(loss_b)
            print("Epoch: (%d), loss: (%e)" % (epoch, loss_))

    # Dictionary that stores the history and prediction of last training
    u = model_pinn(X)
    uR = u[:, 0]
    uI = u[:, 1]

    PINN_dict = {'k': k,
                 'X': X,
                 'r0': r0,
                 'uR': uR,
                 'uI': uI,
                 'history_loss': history_loss,
                 'history_loss_int': history_loss_int,
                 'history_loss_b': history_loss_b}

    PINN_list.append(PINN_dict)
    # Save model and history
    # model_pinn.save("model_dipole_k={}pi".format(round(k / np.pi)))

# np.save("PINN_list_piston_k", PINN_list)
plot_results(PINN_list)

