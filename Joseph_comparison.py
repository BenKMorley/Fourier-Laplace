import numpy
import matplotlib.pyplot as plt
from numpy.core.numeric import zeros_like
from numpy.fft import fftn, ifftn
from tqdm import tqdm
from multiprocessing import Pool

## General Parameters
no_samples = 500
L = 256
alpha = 0
beta = 0
gamma = 0.1
g = 0.1
eps = 0.1  # The standard deviation of the Gaussian noise term
q_fac = 2 * numpy.pi / L
show_to = L // 4
offsets = numpy.arange(0, 10)


## Define the expected q dependence function
def analytic(q0, q1, q2, g=g, alpha=alpha, beta=beta, gamma=gamma):
    q0_hat = 2 * numpy.sin(q0 / 2)
    q1_hat = 2 * numpy.sin(q1 / 2)
    q2_hat = 2 * numpy.sin(q2 / 2)

    q_sq = q0_hat ** 2 + q1_hat ** 2 + q2_hat ** 2

    return (q_sq / g ** 2) ** (3 / 2) * alpha \
        + beta * (q_sq / g ** 2) * (1 / 2) * \
        numpy.log(q_sq / g ** 2, out=numpy.zeros_like(q_sq), where=q_sq!=0) \
        + (gamma / g) * q_sq / g ** 2


def my_fft(f_x):
    x_s = numpy.arange(L)
    q_s = numpy.arange(L) * numpy.pi * 2 / L

    return numpy.sum(numpy.exp(-1j * numpy.outer(q_s, x_s)) * f_x, axis=1)


def my_ifft(x_s):
    return numpy.sum(numpy.exp(+1j * 2 * numpy.pi / L) * q_s)


def Laplace_Transform_1D(data, offset=1):
    L = data.shape[0]

    # For example if the offset is 2, x = L - 2 -> 0 -> -2; x = L - 1 -> 1 -> -1
    # all other x are unaffected
    x_s = (numpy.arange(L) + offset) % L - offset

    # Distribution symmetric in q --> -q analytically, so only use positive q
    p_s = numpy.arange(L) * q_fac

    comb = numpy.outer(p_s, x_s)

    prefactor = numpy.divide(1, 1 - numpy.exp(-p_s * L), out=numpy.zeros_like(p_s), where=p_s!=0)

    return prefactor * numpy.sum(data * numpy.exp(-comb), axis=1)


q0_s = numpy.arange(L).reshape((L, 1, 1)).repeat(L, axis=1).repeat(L, axis=2) * q_fac
q1_s = numpy.arange(L).reshape((1, L, 1)).repeat(L, axis=0).repeat(L, axis=2) * q_fac
q2_s = numpy.arange(L).reshape((1, 1, L)).repeat(L, axis=1).repeat(L, axis=0) * q_fac


def analysis(params):
    alpha, beta, gamma, eps = params

    data_q_3D = analytic(q0_s, q1_s, q2_s, alpha=alpha, beta=beta, gamma=gamma)

    data_q_FT = numpy.zeros((no_samples, L))
    data_q_LT = numpy.zeros((len(offsets), no_samples, L))

    for i in tqdm(range(no_samples)):
        noise = eps * numpy.random.randn(L, L, L)

        data_q = data_q_3D + noise

        data_x = fftn(data_q).real

        data_x_1D = numpy.mean(data_x, axis=(1, 2))

        data_q_FT[i] = fftn(data_x_1D).real
        # data_q_FT2 = my_fft(data_x_1D)

        for j, offset in enumerate(offsets):
            data_q_LT[j, i] = Laplace_Transform_1D(data_x_1D, offset=offset)

    q_s = numpy.arange(L) * numpy.pi * 2 / L

    av = numpy.mean(data_q_FT, axis=0)[:show_to]
    std = numpy.std(data_q_FT, axis=0)[:show_to]
    plt.fill_between(q_s[:show_to], av - std, av + std)
    plt.plot(q_s[:show_to], av)
    plt.xlabel('q')
    plt.title(f'Fourier Transform, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
    plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_FT.png', dpi=500)
    plt.clf()

    for j, offset in enumerate(offsets):
        av = numpy.mean(data_q_LT[j], axis=0)[:show_to]
        std = numpy.std(data_q_LT[j], axis=0)[:show_to]
        plt.fill_between(q_s[:show_to], av - std, av + std,
                        label=f'offset={offset}')
        plt.plot(q_s[:show_to], av)
        plt.xlabel('q')
        plt.title(f'Laplace Transform, offset={offset}, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
        plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_LT_offset{offset}.png', dpi=500)
        plt.clf()

    # Use my favourite offset of 1
    offset = 1
    av = numpy.mean(data_q_FT + data_q_LT[numpy.argwhere(offsets == offset)[0, 0]], axis=0)[:show_to]
    std = numpy.std(data_q_LT[numpy.argwhere(offsets == offset)[0, 0]] + data_q_FT, axis=0)[:show_to]
    plt.fill_between(q_s[:show_to], av - std, av + std)
    plt.plot(q_s[:show_to], av)
    plt.xlabel('q')
    plt.title(f'Fourier Transform + Laplace Transform, offset=1, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
    plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_sum_offset{offset}.png', dpi=500)
    plt.clf()


params = [[1, 0, 0, 0.1], [0, 1, 0, 0.1], [0, 0, 1, 0.1], [1, 1, 0.1, 0.1]]


p = Pool(4)
p.map(analysis, params)

# for param_set in params:
#     analysis(param_set)
