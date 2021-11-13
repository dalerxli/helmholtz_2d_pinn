# Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
from models_2d import mapping_model
from loss_funs import loss_dipole, loss_piston, get_training_points
from make_plots import plot_results

# Model
input_dim = 2
output_dim = 2
width = 20
depth = 5
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

TOL = 5e-5
max_epochs = 50

K = 2 * np.pi * np.array([2, 4])
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
    history_loss_b0 = np.array([])
    history_loss = np.array([])

    loss = TOL + 1
    epoch = 0
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

            loss_int, loss_b, loss_b0, loss = loss_piston(k, uR, uI, uR_X, uI_X, uR_XX, uI_XX,
                                                          uR_b, uI_b, uR_X_b, uI_X_b, X_b,
                                                          uR_b0, uI_b0, X_b0,
                                                          fR=0, fI=0,
                                                          C0=1., C1=1., C2=0.1, C3=0.1, C4=0.1, C5=0.1)

        grads = g1.gradient(loss, model_pinn.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_pinn.trainable_variables))

        history_loss_int = np.append(history_loss_int, loss_int.numpy())
        history_loss_b = np.append(history_loss_b, loss_b.numpy())
        history_loss_b0 = np.append(history_loss_b0, loss_b0.numpy())
        history_loss = np.append(history_loss, loss.numpy())

        epoch += 1
        if epoch % 10 == 0:
            print("Epoch: (%d), loss: (%e)" % (epoch, loss))

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
                 'history_loss_b': history_loss_b,
                 'history_loss_b0': history_loss_b0}

    PINN_list.append(PINN_dict)
    # Save model and history
    # model_pinn.save("model_dipole_k={}pi".format(round(k / np.pi)))

np.save("PINN_list_piston", PINN_list)
plot_results(PINN_list)
