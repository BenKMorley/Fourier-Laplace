import numpy
import matplotlib.pyplot as plt
from numpy.core.numeric import zeros_like
from numpy.fft import fftn, ifftn
from tqdm import tqdm
from multiprocessing import Pool
import pickle

## General Parameters
no_samples = 500
L = 256
q_fac = 2 * numpy.pi / L
show_to = L // 4
offsets = numpy.arange(0, 10)
g = 0.1


## Define the expected q dependence function
def analytic(q_s, g, alpha, beta, gamma):
    q_hat = []
    L = q_s[0].shape[0]
    dim = len(q_s)
    q_sq = numpy.zeros((L, ) * dim)

    for d in range(dim):
        q_hat = 2 * numpy.sin(q_s[d] / 2)
        q_sq += q_hat ** 2

    return (q_sq / g ** 2) ** (3 / 2) * alpha \
        + beta * (q_sq / g ** 2) * (1 / 2) * \
        numpy.log(q_sq / g ** 2, out=numpy.zeros_like(q_sq), where=q_sq!=0) \
        + (gamma / g) * q_sq / g ** 2


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


def Laplace_Transform_ND(data, dim, offset):
    L = data.shape[0]
    p_s = numpy.arange(L) * q_fac
    assert len(offset) == dim

    # prefactor = numpy.divide(1, 1 - numpy.exp(-p_s * L), out=numpy.zeros_like(p_s), where=p_s!=0)

    result = numpy.zeros((L, ) * dim)

    for d in range(dim):
        x_s = (numpy.arange(L) + offset[d]) % L - offset[d]

        comb = numpy.outer(p_s, x_s)
        # factor = prefactor.reshape((L, 1)).repeat(L, axis=1) * numpy.exp(-comb)

        data = numpy.tensordot(data, numpy.exp(-comb), axes=(d, 1))

    return data


q0_s = numpy.arange(L).reshape((L, 1, 1)).repeat(L, axis=1).repeat(L, axis=2) * q_fac
q1_s = numpy.arange(L).reshape((1, L, 1)).repeat(L, axis=0).repeat(L, axis=2) * q_fac
q2_s = numpy.arange(L).reshape((1, 1, L)).repeat(L, axis=1).repeat(L, axis=0) * q_fac


def analysis(params):
    ## Analysis
    alpha, beta, gamma, eps = params

    data_q_3D = analytic((q0_s, q1_s, q2_s), g, alpha, beta, gamma)

    data_q_FT = numpy.zeros((no_samples, L, L, L))
    data_q_LT = numpy.zeros((len(offsets), no_samples, L, L, L))
    data_x_sum = numpy.zeros((len(offsets), no_samples, L), dtype=numpy.complex128)

    try:
        data_full = pickle.load(open(f"data/fake_data/alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_samples{no_samples}.pcl", "rb"))
        data_q_FT, data_q_LT, data_x_sum = data_full
        raise(Exception)

    except Exception:
        for i in tqdm(range(no_samples)):
            noise = eps * numpy.random.randn(L, L, L)

            data_q = data_q_3D + noise

            data_x = fftn(data_q).real

            data_x_1D = numpy.mean(data_x, axis=(1, 2))

            data_q_FT[i] = fftn(data_x_1D).real

            for j, offset in enumerate(offsets):
                data_q_LT[j, i] = Laplace_Transform_1D(data_x_1D, offset=offset)

                # Have a look in x-space of the IFT of FT + LT
                data_x_sum[j, i] = ifftn(data_q_FT[i] + data_q_LT[j, i])

        data_full = data_q_FT, data_q_LT, data_x_sum
        pickle.dump(data_full, open(f"data/fake_data/alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_samples{no_samples}.pcl", "wb"))

    q_s = numpy.arange(L) * numpy.pi * 2 / L

    ## Plotting
    av = numpy.mean(data_q_FT, axis=0)[1:show_to]
    std = numpy.std(data_q_FT, axis=0)[1:show_to]
    plt.fill_between(q_s[1:show_to] / g, av - std, av + std)
    plt.plot(q_s[1:show_to] / g, av)
    plt.xlabel('q / g')
    plt.title(f'Fourier Transform, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
    plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_FT.png', dpi=500)
    plt.clf()

    for j, offset in enumerate(offsets):
        av = numpy.mean(data_q_LT[j], axis=0)[1:show_to]
        std = numpy.std(data_q_LT[j], axis=0)[1:show_to]
        plt.fill_between(q_s[1:show_to] / g, av - std, av + std,
                         label=f'offset={offset}')
        plt.plot(q_s[1:show_to] / g, av)
        plt.xlabel('q / g')
        plt.title(f'Laplace Transform, offset={offset}, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
        plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_LT_offset{offset}.png', dpi=500)
        plt.clf()

        av = numpy.mean(data_x_sum[j], axis=0)[1:show_to]
        std = numpy.std(data_x_sum[j], axis=0)[1:show_to]
        plt.fill_between(q_s[1:show_to] / g, (av - std).real, (av + std).real,
                         label='real', color='k')
        plt.plot(q_s[1:show_to] / g, av.real, color='k')
        plt.fill_between(q_s[1:show_to] / g, (av - std).imag, (av + std).imag,
                         label='imag', color='r')
        plt.plot(q_s[1:show_to] / g, av.imag, color='r')
        plt.xlabel('x a')
        plt.legend()
        plt.title(f'Sum x-space, offset={offset}, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
        plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_x_sum_offset{offset}.png', dpi=500)
        plt.clf()

    # Use my favourite offset of 1
    offset = 1
    av = numpy.mean(data_q_FT + data_q_LT[numpy.argwhere(offsets == offset)[0, 0]], axis=0)[1:show_to]
    std = numpy.std(data_q_LT[numpy.argwhere(offsets == offset)[0, 0]] + data_q_FT, axis=0)[1:show_to]
    plt.fill_between(q_s[1:show_to] / g, av - std, av + std)
    plt.plot(q_s[1:show_to] / g, av)
    plt.xlabel('q / g')
    plt.title(f'Fourier Transform + Laplace Transform, offset=1, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
    plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_sum_offset{offset}.png', dpi=500)
    plt.clf()


def analysis_ND_Laplace(params, dim=3):
    alpha, beta, gamma, eps = params

    q_s = numpy.arange(L) * numpy.pi * 2 / L

    data_q_ND = analytic((q_s,) * dim, g, alpha, beta, gamma)

    data_q_FT = numpy.zeros((no_samples, ) + (L, ) * dim)
    data_q_LT = numpy.zeros((no_samples, ) + (L, ) * dim)
    data_q_LT2 = numpy.zeros((no_samples, ) + (L, ) * dim)

    for i in tqdm(range(no_samples)):
        noise = eps * numpy.random.randn(*((L, ) * dim))

        data_q = data_q_ND + noise

        data_x = ifftn(data_q).real

        data_q_FT[i] = fftn(data_x).real

        data_q_LT[i] = Laplace_Transform_ND(data_x, dim=dim, offset=(1, ) * dim)
        data_q_LT2[i] = Laplace_Transform_1D(data_x)

        print("Hello")


params = [[1, 0, 0, 0.1], [0, 1, 0, 0.1], [0, 0, 1, 0.1], [1, 1, 0.1, 0.1]]


analysis_ND_Laplace(params[2], dim=1)

p = Pool(4)
p.map(analysis, params)
