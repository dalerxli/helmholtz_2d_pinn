import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
from scipy.special import hankel1 as H1
from scipy.special import h1vp as dH1_dz
from scipy.special import hankel2 as H2
from scipy.special import h2vp as dH2_dz


def get_dipole_solution(x1, x2, k, r0):

    r = np.sqrt(x1 ** 2 + x2 ** 2)
    cos_theta = tf.cast(x1 / r, dtype='complex64')

    A_1 = (dH2_dz(1, k) - 1j * H2(1, k)) / (H1(1, k * r0) * (dH2_dz(1, k) -
                                                             1j * H2(1, k)) - H2(1, k * r0) * (dH1_dz(1, k) - 1j * H1(1, k)))
    C_1 = (dH1_dz(1, k) - 1j * H1(1, k)) / (H1(1, k * r0) * (dH2_dz(1, k) -
                                                             1j * H2(1, k)) - H2(1, k * r0) * (dH1_dz(1, k) - 1j * H1(1, k)))
    U = (A_1 * H1(1, k * r) + C_1 * H2(1, k * r)) * cos_theta

    U = np.reshape(U, [-1])
    UR, UI = np.real(U), np.imag(U)
    U_abs = np.sqrt(UR ** 2 + UI ** 2)

    return UR, UI, U_abs


def get_piston_solution(x1, x2, k, r0):

    r = np.sqrt(x1 ** 2 + x2 ** 2)
    theta = np.arctan2(x2, x1)

    U = 0
    for n in range(0, 8):
        if n == 0:
            f_n = 1 / 6
        else:
            f_n = 2 * np.sin(n * np.pi / 6) / (np.pi * n)

        A_n = f_n * (dH2_dz(n, k) - 1j * H2(n, k)) / (H1(n, k * r0) * (dH2_dz(n,
                                                                              k) - 1j * H2(n, k)) - H2(n, k * r0) * (dH1_dz(n, k) - 1j * H1(n, k)))
        C_n = f_n * (dH1_dz(n, k) - 1j * H1(n, k)) / (H1(n, k * r0) * (dH2_dz(n,
                                                                              k) - 1j * H2(n, k)) - H2(n, k * r0) * (dH1_dz(n, k) - 1j * H1(n, k)))

        cos_n = np.cos(n * theta)
        U += (A_n * H1(n, k * r) + C_n * H2(n, k * r)) * cos_n

    U = np.reshape(U, [-1])
    UR, UI = np.real(U), np.imag(U)
    U_abs = np.sqrt(UR ** 2 + UI ** 2)

    return UR, UI, U_abs


def get_dipole_solution_infty(x1, x2, k, r0):

    r = np.sqrt(x1 ** 2 + x2 ** 2)
    cos_theta = tf.cast(x1 / r, dtype='complex64')
    U = H1(1, k * r) / H1(1, k * r0) * cos_theta
    U = np.reshape(U, [-1])
    UR, UI = np.real(U), np.imag(U)
    U_abs = np.sqrt(UR ** 2 + UI ** 2)

    return UR, UI, U_abs


def get_piston_solution_infty(x1, x2, k, r0):

    r = np.sqrt(x1 ** 2 + x2 ** 2)
    theta = np.arctan2(x2, x1)

    A_0 = 1 / 6
    U = H1(0, k * r) / H1(0, k * r0) * A_0

    for n in range(1, 20):
        A_n = 2 * np.sin(n * np.pi / 6) / (np.pi * n)
        cos_n = np.cos(n * theta)
        U += A_n * cos_n * H1(n, k * r) / H1(n, k * r0)

    U = np.reshape(U, [-1])
    UR, UI = np.real(U), np.imag(U)
    U_abs = np.sqrt(UR ** 2 + UI ** 2)

    return UR, UI, U_abs


# Function to plot solution
def plot_results(PINN_list):

    eR_rel = np.array([])
    eI_rel = np.array([])
    eR_inf = np.array([])
    eI_inf = np.array([])
    K = np.array([])

    for PINN_dict in PINN_list:

        k = PINN_dict['k']
        X = PINN_dict['X']
        r0 = PINN_dict['r0']
        uR = PINN_dict['uR']
        uI = PINN_dict['uI']
        loss = PINN_dict['history_loss']
        loss_int = PINN_dict['history_loss_int']
        loss_b = PINN_dict['history_loss_b']
        loss_b0 = PINN_dict['history_loss_b0']

        u_abs = np.sqrt(uR**2 + uI**2)
        x1 = X[:, 0]
        x2 = X[:, 1]

        UR, UI, U_abs = get_piston_solution(x1, x2, k, r0)

        x = np.asarray(x1)
        y = np.asarray(x2)

        triang = tri.Triangulation(x, y)
        triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                                 y[triang.triangles].mean(axis=1)) < r0)

        fig, axs = plt.subplots(3, 3, figsize=(10, 8))
        fig.suptitle('Loss history and solution for k = {} $\pi$'.format(
            round(k / np.pi)), fontsize=12)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        axs[0, 0].plot(loss_int, label='loss interior domain')
        axs[0, 0].set_yscale('log')
        axs[0, 1].plot(loss_b, label='loss external boundary')
        axs[0, 1].set_yscale('log')
        axs[0, 2].plot(loss_b0, label='loss inner boundary')
        axs[0, 2].set_yscale('log')

        axs[0, 0].set_title('Interior Domain Loss', fontsize=10)
        axs[0, 0].set_xlabel('Epoch', fontsize=10)
        axs[0, 0].set_ylabel('loss_int', fontsize=10)

        axs[0, 1].set_title('External Boundary Loss', fontsize=10)
        axs[0, 1].set_xlabel('Epoch', fontsize=10)
        axs[0, 1].set_ylabel('loss_b', fontsize=10)

        axs[0, 2].set_title('Inner Boundary Loss', fontsize=10)
        axs[0, 2].set_xlabel('Epoch', fontsize=10)
        axs[0, 2].set_ylabel('Loss_b0', fontsize=10)

        real = axs[1, 0].tricontourf(
            triang, uR, 20, cmap='seismic', levels=100)
        divider = make_axes_locatable(axs[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(real, cax=cax, orientation='vertical')

        im = axs[1, 1].tricontourf(triang, uI, 20, cmap='seismic', levels=100)
        divider = make_axes_locatable(axs[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        abs = axs[1, 2].tricontourf(
            triang, u_abs, 20, cmap='inferno', levels=100)
        divider = make_axes_locatable(axs[1, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(abs, cax=cax, orientation='vertical')

        axs[1, 0].set_title('Real part', fontsize=10)
        axs[1, 0].set_xlabel('x', fontsize=10)
        axs[1, 0].set_ylabel('y', fontsize=10)
        axs[1, 0].set_xlim(-1, 1)
        axs[1, 0].set_ylim(-1, 1)

        axs[1, 1].set_title('Imaginary part', fontsize=10)
        axs[1, 1].set_xlabel('x', fontsize=10)
        axs[1, 1].set_ylabel('y', fontsize=10)
        axs[1, 1].set_xlim(-1, 1)
        axs[1, 1].set_ylim(-1, 1)

        axs[1, 2].set_title('Absolute Value', fontsize=10)
        axs[1, 2].set_xlabel('x', fontsize=10)
        axs[1, 2].set_ylabel('y', fontsize=10)
        axs[1, 2].set_xlim(-1, 1)
        axs[1, 2].set_ylim(-1, 1)

        real_diff = axs[2, 0].tricontourf(
            triang, np.abs(uR - UR), 20, cmap='Reds', levels=100)
        divider = make_axes_locatable(axs[2, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(real_diff, cax=cax, orientation='vertical')

        im_diff = axs[2, 1].tricontourf(
            triang, np.abs(uI - UI), 20, cmap='Reds', levels=100)
        divider = make_axes_locatable(axs[2, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_diff, cax=cax, orientation='vertical')

        abs_diff = axs[2, 2].tricontourf(triang, np.abs(
            u_abs - U_abs), 20, cmap='Reds', levels=100)
        divider = make_axes_locatable(axs[2, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(abs_diff, cax=cax, orientation='vertical')

        axs[2, 0].set_title('Real part error', fontsize=10)
        axs[2, 0].set_xlabel('x', fontsize=10)
        axs[2, 0].set_ylabel('y', fontsize=10)
        axs[2, 0].set_xlim(-1, 1)
        axs[2, 0].set_ylim(-1, 1)

        axs[2, 1].set_title('Imaginary part error', fontsize=10)
        axs[2, 1].set_xlabel('x', fontsize=10)
        axs[2, 1].set_ylabel('y', fontsize=10)
        axs[2, 1].set_xlim(-1, 1)
        axs[2, 1].set_ylim(-1, 1)

        axs[2, 2].set_title('Absolute Value error', fontsize=10)
        axs[2, 2].set_xlabel('x', fontsize=10)
        axs[2, 2].set_ylabel('y', fontsize=10)
        axs[2, 2].set_xlim(-1, 1)
        axs[2, 2].set_ylim(-1, 1)

        eR = uR - UR
        eI = uI - UI

        eR_rel = np.append(eR_rel, 100 * tf.norm(eR) / tf.norm(UR))
        eI_rel = np.append(eI_rel, 100 * tf.norm(eI) / tf.norm(UI))

        eR_inf = np.append(eR_inf, tf.norm(eR, ord=np.inf))
        eI_inf = np.append(eI_inf, tf.norm(eI, ord=np.inf))

        K = np.append(K, k)
        plt.tight_layout()
        plt.savefig('dipole_solution_k={}pi'.format(round(k / np.pi)))

    # plt.show()

    # Plot error vs k
    K_ = np.round(K / (2 * np.pi))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('PINN solution error vs k', fontsize=14)
    fig.subplots_adjust(hspace=0.2)

    axs[0].plot(K_, eR_rel, 'o-', label='Real part')
    axs[0].plot(K_, eI_rel, 'o-', label='Imaginary part')
    axs[0].set_title('Relative $l_2$ error', fontsize=12)
    axs[0].set_xlabel('k/2$\pi$', fontsize=10)
    axs[0].set_ylabel('Error [%]', fontsize=10)

    axs[1].plot(K_, eR_inf, 'o-', label='Real part')
    axs[1].plot(K_, eI_inf, 'o-', label='Imaginary part')
    axs[1].set_title('$l_\infty$ error', fontsize=12)
    axs[1].set_xlabel('k/2$\pi$', fontsize=10)
    axs[1].set_ylabel('Error', fontsize=10)

    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')

    plt.savefig('dipole_error_vs_k')

    plt.show()


def plot_live_results(X, r0, u):

    # Make live plot
    x = np.asarray(X[:, 0])
    y = np.asarray(X[:, 1])
    triang = tri.Triangulation(x, y)
    triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                             y[triang.triangles].mean(axis=1)) < r0)

    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].set_title('Real part', fontsize=14)
    axs[0].set_xlabel('x', fontsize=12)
    axs[0].set_ylabel('y', fontsize=12)
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(-1, 1)

    axs[1].set_title('Imaginary part', fontsize=14)
    axs[1].set_xlabel('x', fontsize=12)
    axs[1].set_ylabel('y', fontsize=12)
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylim(-1, 1)

    uR = u[:, 0]
    uI = u[:, 1]
    axs[0].tricontourf(triang, uR, 20, cmap='inferno')
    axs[1].tricontourf(triang, uI, 20, cmap='inferno')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()
    plt.ioff()
