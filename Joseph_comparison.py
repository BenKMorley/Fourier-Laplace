import numpy
import matplotlib.pyplot as plt
from numpy.core.numeric import tensordot, zeros_like
from numpy.fft import fftn, ifftn
from numpy.lib.function_base import gradient
from scipy.optimize import minimize, least_squares
from tqdm import tqdm
from multiprocessing import Pool
import pickle
from copy import copy
from functools import reduce


## Define the expected q dependence function
def analytic(L, dim, N, g, alpha, beta, gamma, eta):
    q_s = []  # start as list to append
    q_fac = 2 * numpy.pi / L

    for i in range(dim):
        q_part = numpy.arange(L).reshape((1, ) * i + (L, ) + (1, ) * (dim - i - 1)) * q_fac
        for j in range(dim - 1):
            q_part.repeat(L, axis=i)

        q_s.append(q_part)

    q_sq = numpy.zeros((L, ) * dim)

    for d in range(dim):
        q_hat = 2 * numpy.sin(q_s[d] / 2)
        q_sq += q_hat ** 2

    return N ** 2 * (q_sq / g ** 2) * (alpha * numpy.sqrt(q_sq) + beta * g * (1 / 2) * \
        numpy.log(q_sq / g ** 2, out=numpy.zeros_like(q_sq), where=q_sq!=0) + gamma) + eta


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
    q_fac = 2 * numpy.pi / L
    p_s = numpy.arange(L) * q_fac
    assert len(offset) == dim

    # prefactor = numpy.divide(1, 1 - numpy.exp(-p_s * L), out=numpy.zeros_like(p_s), where=p_s!=0)

    result = numpy.zeros((L, ) * dim)

    for d in range(dim):
        x_s = (numpy.arange(L) + offset[d]) % L - offset[d]

        comb = numpy.outer(p_s, x_s)
        # factor = prefactor.reshape((L, 1)).repeat(L, axis=1) * numpy.exp(-comb)

        # if dim == 2:
        #     data_new = numpy.zeros_like(data)
        #     for i in range(L):
        #         for j in range(L):
        #             for k in range(L):
        #                 data_new[i, j] += numpy.exp(-comb)[i, k] * data[k, j]

        data = numpy.tensordot(numpy.exp(-comb), data, axes=(1, d))

    return data


def gen_data(params, dim=3, rerun=False, offsets=range(10)):
    alpha, beta, gamma, eps, eta = params

    data_q = analytic(q_tuple, g, alpha, beta, gamma, eta)

    data_q_FT = numpy.zeros((no_samples, L))
    data_q_LT = numpy.zeros((len(offsets), no_samples, L))
    data_x_FT = numpy.zeros((no_samples, L))
    data_x_LT = numpy.zeros((len(offsets), no_samples, L), dtype=numpy.complex128)
    data_x_sum = numpy.zeros((len(offsets), no_samples, L), dtype=numpy.complex128)

    if not rerun:
        try:
            data_full = pickle.load(open(f"data/fake_data/alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_samples{no_samples}.pcl", "rb"))

        except Exception:
            rerun = True

    if rerun:
        for i in tqdm(range(no_samples)):
            noise = eps * numpy.random.randn(*(L, ) * dim)

            data_q = data_q + noise

            data_x = fftn(data_q).real

            data_x_1D = numpy.mean(data_x, axis=tuple(range(1, dim)))

            data_q_FT[i] = fftn(data_x_1D).real

            for j, offset in enumerate(offsets):
                data_q_LT[j, i] = Laplace_Transform_1D(data_x_1D, offset=offset)

                # Have a look in x-space of the IFT of FT + LT
                data_x_LT[j, i] = ifftn(data_q_LT[j, i])
                data_x_FT[i] = ifftn(data_q_FT[i])

                data_x_sum[j, i] = data_x_LT[j, i] + data_x_FT[i]

        data_full = data_q_FT, data_q_LT, data_x_FT, data_x_LT, data_x_sum

        pickle.dump(data_full, open(f"data/fake_data/alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_samples{no_samples}.pcl", "wb"))

    q_s = numpy.arange(L) * q_fac

    data_x_1D = ifftn(data_q_FT, axes=(1, )).real

    return data_full, params


def plotting(data):  # data assumed to be the form of what is returned by analysis
    data_full, params = data
    alpha, beta, gamma, eps = params
    data_q_FT, data_q_LT, data_x_FT, data_x_LT, data_x_sum = data_full

    q_s = numpy.arange(L) * q_fac

    data_x_1D = ifftn(data_q_FT, axes=(1, )).real

    av = numpy.roll(numpy.mean(data_x_1D, axis=0), 2)[:show_to + 2]
    std = numpy.roll(numpy.std(data_x_1D, axis=0), 2)[:show_to + 2]
    plt.fill_between(numpy.arange(-2, L)[:show_to + 2] / g, av - std, av + std)
    plt.plot(numpy.arange(-2, L)[:show_to + 2] / g, av)
    plt.xlabel('a x')
    plt.title(f'Fourier Transform, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
    plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_x_space.pdf')
    plt.clf()

    for j, offset in enumerate(offsets):
        av = numpy.mean(data_q_FT, axis=0)[1:show_to]
        std = numpy.std(data_q_FT, axis=0)[1:show_to]
        plt.fill_between(q_s[1:show_to] / g, av - std, av + std, color='red', alpha=0.2)
        plt.plot(q_s[1:show_to] / g, av, color='r', label='FT')
        plt.xlabel('q / g')
        # plt.title(f'Fourier Transform, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
        # plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_FT.pdf')
        # plt.clf()

        av = numpy.mean(data_q_LT[j], axis=0)[1:show_to]
        std = numpy.std(data_q_LT[j], axis=0)[1:show_to]
        plt.fill_between(q_s[1:show_to] / g, av - std, av + std,
                        label=f'offset={offset}', color='b', alpha=0.2)
        plt.plot(q_s[1:show_to] / g, av, color='b', alpha=0.2, label='LT')

        av = numpy.mean(data_q_FT + data_q_LT[numpy.argwhere(offsets == offset)[0, 0]], axis=0)[1:show_to]
        std = numpy.std(data_q_LT[numpy.argwhere(offsets == offset)[0, 0]] + data_q_FT, axis=0)[1:show_to]
        plt.fill_between(q_s[1:show_to] / g, av - std, av + std, color='k', alpha=0.2)
        plt.plot(q_s[1:show_to] / g, av, color='k', ls='--', label='LT+FT')

        plt.xlabel('q / g')
        plt.legend()
        plt.title(f'Laplace Transform, offset={offset}, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
        plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_LT_offset{offset}.pdf')
        plt.clf()

        gradient1, intercept1 = numpy.polyfit(numpy.log(q_s[show_to // 3:show_to] / g),
                                            numpy.log(numpy.abs(numpy.mean(data_q_FT, axis=0)[show_to // 3:show_to])),
                                            1)
        gradient2, intercept2 = numpy.polyfit(numpy.log(q_s[show_to // 3:show_to] / g),
                                            numpy.log(numpy.abs(numpy.mean(data_q_LT[j], axis=0)[show_to // 3:show_to])),
                                            1)
        gradient3, intercept3 = numpy.polyfit(numpy.log(q_s[show_to // 3:show_to] / g),
                                            numpy.log(numpy.abs(numpy.mean(data_q_LT[numpy.argwhere(offsets == offset)[0, 0]] + data_q_FT, axis=0)[show_to // 3:show_to])),
                                            1)

        av = numpy.abs(numpy.mean(data_q_FT, axis=0)[1:show_to])
        std = numpy.std(data_q_FT, axis=0)[1:show_to]
        plt.fill_between(q_s[1:show_to] / g, av - std, av + std, color='red', alpha=0.2)
        plt.plot(q_s[1:show_to] / g, av, color='r', label=f'|FT|')
        plt.xlabel('q / g')
        # plt.title(f'Fourier Transform, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
        # plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_FT.pdf')
        # plt.clf()

        av = numpy.abs(numpy.mean(data_q_LT[j], axis=0)[1:show_to])
        std = numpy.std(data_q_LT[j], axis=0)[1:show_to]
        plt.fill_between(q_s[1:show_to] / g, av - std, av + std,
                        label=f'offset={offset}', color='b', alpha=0.2)
        plt.plot(q_s[1:show_to] / g, av, color='b', alpha=0.2, label=f'|LT|')

        av = numpy.abs(numpy.mean(data_q_FT + data_q_LT[numpy.argwhere(offsets == offset)[0, 0]], axis=0)[1:show_to])
        std = numpy.std(data_q_LT[numpy.argwhere(offsets == offset)[0, 0]] + data_q_FT, axis=0)[1:show_to]
        plt.fill_between(q_s[1:show_to] / g, av - std, av + std, color='k', alpha=0.2)
        plt.plot(q_s[1:show_to] / g, av, color='k', label=f'|LT+FT|')

        plt.plot(q_s[1:show_to] / g, numpy.exp(intercept1) * (q_s[1:show_to] / g) ** gradient1, color='r', ls='--', label=f'gradient={gradient1}')
        plt.plot(q_s[1:show_to] / g, numpy.exp(intercept2) * (q_s[1:show_to] / g) ** gradient2, color='b', ls='--', label=f'gradient={gradient2}')
        plt.plot(q_s[1:show_to] / g, numpy.exp(intercept3) * (q_s[1:show_to] / g) ** gradient3, color='k', ls='--', label=f'gradient={gradient3}')

        plt.xlabel('q / g')
        plt.legend()
        plt.loglog()
        plt.title(f'Laplace Transform, offset={offset}, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
        plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_LT_offset{offset}_log_log.pdf')
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
        plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_x_sum_offset{offset}.pdf')
        plt.clf()

        # Plot the relative difference between the signal expected
        av = numpy.abs(numpy.mean(data_q_FT + data_q_LT[numpy.argwhere(offsets == offset)[0, 0]], axis=0)[1:show_to])
        std = numpy.std(data_q_LT[numpy.argwhere(offsets == offset)[0, 0]] + data_q_FT, axis=0)[1:show_to]
        plt.plot(q_s[1:show_to] / g, av.real / std, color='k')
        plt.xlabel('q / g')
        plt.loglog()
        plt.title(f'Relative Difference, offset={offset}, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
        plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_rel_diff_offset{offset}.pdf')
        plt.clf()

    # Use my favourite offset of 1
    offset = 1
    av = numpy.mean(data_q_FT + data_q_LT[numpy.argwhere(offsets == offset)[0, 0]], axis=0)[1:show_to]
    std = numpy.std(data_q_LT[numpy.argwhere(offsets == offset)[0, 0]] + data_q_FT, axis=0)[1:show_to]
    plt.fill_between(q_s[1:show_to] / g, av - std, av + std)
    plt.plot(q_s[1:show_to] / g, av)
    plt.xlabel('q / g')
    plt.title(f'Fourier Transform + Laplace Transform, offset=1, alpha={alpha}, beta={beta}, gamma={gamma}, eps={eps}')
    plt.savefig(f'graphs/Fake_data_alpha{alpha}_beta{beta}_gamma{gamma}_eps{eps}_sum_offset{offset}.pdf')
    plt.clf()


def analysis_ND_Laplace(params, L, N=2, dim=3, g=0.1, no_samples=500, plot=False):
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    cut = params["cut"]
    eps = params["eps"]
    eta = params["eta"]

    q_s = numpy.arange(L) * numpy.pi * 2 / L

    data_q_ND = analytic(L, dim, N, g, alpha, beta, gamma, eta)

    data_q_FT = numpy.zeros((no_samples, ) + (L, ) * dim)
    data_q_LT = numpy.zeros((no_samples, ) + (L, ) * dim)

    for i in tqdm(range(no_samples)):
        noise = eps * numpy.random.randn(*((L, ) * dim))

        data_q = data_q_ND + noise

        data_x = ifftn(data_q).real
        data_q_FT[i] = fftn(data_x).real
        data_q_LT[i] = Laplace_Transform_ND(data_x, dim=dim, offset=(1, ) * dim)

    # Combine the Fourier Transform and the Laplace Transform to remove the q^2
    # piece
    data_processed = data_q_FT + data_q_LT

    # Cut away the higher momentum modes from the data
    for d in range(dim):
        data_processed = numpy.take(data_processed, numpy.arange(1, int(numpy.rint(L * cut))), axis=1 + d)

    # Get the covariance matrix using the samples
    # cov_matrix = numpy.cov(data_processed)
    # cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    # cov_inv = numpy.linalg.inv(cov_1_2)

    # Use the linearity of the Fourier and Laplace Transforms to prerun them
    data_q_alpha = analytic(L, dim, g, 1, 0, 0)
    data_x_alpha = ifftn(data_q_alpha).real
    data_q_FT_alpha = fftn(data_x_alpha).real
    data_q_LT_alpha = Laplace_Transform_ND(data_x_alpha, dim=dim, offset=(1, ) * dim)
    data_q_alpha = data_q_FT_alpha + data_q_LT_alpha

    data_q_beta = analytic(L, dim, g, 0, 1, 0)
    data_x_beta = ifftn(data_q_beta).real
    data_q_FT_beta = fftn(data_x_beta).real
    data_q_LT_beta = Laplace_Transform_ND(data_x_beta, dim=dim, offset=(1, ) * dim)
    data_q_beta = data_q_FT_beta + data_q_LT_beta

    data_q_gamma = analytic(L, dim, g, 0, 0, 1)
    data_x_gamma = ifftn(data_q_gamma).real
    data_q_FT_gamma = fftn(data_x_gamma).real
    data_q_LT_gamma = Laplace_Transform_ND(data_x_gamma, dim=dim, offset=(1, ) * dim)
    data_q_gamma = data_q_FT_gamma + data_q_LT_gamma

    data_processed_mean = numpy.mean(data_processed, axis=0)

    # Apply the same cut to the reference data
    for d in range(dim):
        data_q_alpha = numpy.take(data_q_alpha, numpy.arange(1, int(numpy.rint(L * cut))), axis=d)
        data_q_beta = numpy.take(data_q_beta, numpy.arange(1, int(numpy.rint(L * cut))), axis=d)
        data_q_gamma = numpy.take(data_q_gamma, numpy.arange(1, int(numpy.rint(L * cut))), axis=d)

    # Now compare to model data with no q^2 piece
    def minimize_this(x, i, cov_inv=None):
        alpha, beta = x

        data_q_ref = alpha * data_q_alpha + beta * data_q_beta

        residuals = data_q_ref - data_processed_mean
        residuals = residuals.reshape(numpy.product(residuals.shape))

        # normalized_residuals = numpy.dot(cov_inv, residuals)

        return residuals

    # res = least_squares(minimize_this, [0, 0], args=(cov_inv, ), method="lm")
    res_x_s = numpy.zeros((no_samples, 2))

    for i in range(no_samples):
        res = least_squares(minimize_this, [0, 0], args=(i, ), method="lm")

        if plot:
            alpha, beta = res.x

            data_q_ref = alpha * data_q_alpha + beta * data_q_beta

            plt.scatter(q_s[1:int(numpy.rint(L * cut))], data_processed_mean[(0, ) * (dim - 1)], label='noisy data')
            plt.plot(q_s[1:int(numpy.rint(L * cut))], data_q_ref[(0, ) * (dim - 1)], label=rf'fit: $\alpha$ = {alpha}, $\beta$ = {beta}')
            plt.title(rf'$\alpha$ = {params["alpha"]}, $\beta$ = {params["beta"]}, $\epsilon$={eps}, $\gamma$ = {gamma}, cut={cut}')
            plt.legend()
            plt.show()

        return res


def Fourier_to_Laplace_analytic(f_p, r=1):
    """
        f_p is the momentum space evaluation of the function over a comb of
        values.
    """
    L = len(f_p)
    p_s = numpy.arange(L)

    def A(p, q):
        return numpy.exp(2 * numpy.pi / L * (q - 1j * p))

    results = numpy.zeros(L)

    for q in range(L):
        results[q] = numpy.mean(f_p * A(p_s, q) ** r / (1 - A(p_s, q) ** -1))

    return results


class explore_params():
    def __init__(self, L, N, test_points, no_samples, alpha=0, beta=0, gamma=0, eps=1, g=0.1, cut=1, dim=2, eta=0):
        self.N = N
        self.L = L
        self.test_points = test_points
        self.g = g
        self.dim = dim
        self.no_samples = no_samples
        self.q_s = numpy.arange(self.L) * numpy.pi * 2 / L

        self.params = {}
        self.params["alpha"] = alpha
        self.params["beta"] = beta
        self.params["gamma"] = gamma
        self.params["eps"] = eps
        self.params["cut"] = cut

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.eps = eps

        self.param_ranges = {}
        self.param_ranges["alpha"] = 10 ** numpy.linspace(-2, 2, test_points)
        self.param_ranges["beta"] = 10 ** numpy.linspace(-2, 2, test_points)
        self.param_ranges["gamma"] = 10 ** numpy.linspace(-2, 2, test_points)
        self.param_ranges["eps"] = 10 ** numpy.linspace(-2, 3, test_points)
        self.param_ranges["cut"] = numpy.linspace(4 / L, 0.5, test_points)

        self.params_latex = {}
        self.params_latex["alpha"] = r"$\alpha$"
        self.params_latex["beta"] = r"$\beta$"
        self.params_latex["gamma"] = r"$\gamma$"
        self.params_latex["eps"] = r"$\epsilon$"
        self.params_latex["cut"] = r"cut"

        # Use the linearity of the Fourier and Laplace Transforms to prerun them
        data_q_alpha = analytic(self.L, self.dim, self.N, self.g, 1, 0, 0, 0)
        data_x_alpha = ifftn(data_q_alpha).real
        data_q_FT_alpha = fftn(data_x_alpha).real
        data_q_LT_alpha = Laplace_Transform_ND(data_x_alpha, dim=self.dim, offset=(1, ) * self.dim)
        self.data_q_alpha = data_q_FT_alpha + data_q_LT_alpha

        data_q_beta = analytic(self.L, self.dim, self.N, self.g, 0, 1, 0, 0)
        data_x_beta = ifftn(data_q_beta).real
        data_q_FT_beta = fftn(data_x_beta).real
        data_q_LT_beta = Laplace_Transform_ND(data_x_beta, dim=self.dim, offset=(1, ) * self.dim)
        self.data_q_beta = data_q_FT_beta + data_q_LT_beta

        data_q_gamma = analytic(self.L, self.dim, self.N, self.g, 0, 0, 1, 0)
        data_x_gamma = ifftn(data_q_gamma).real
        data_q_FT_gamma = fftn(data_x_gamma).real
        data_q_LT_gamma = Laplace_Transform_ND(data_x_gamma, dim=self.dim, offset=(1, ) * self.dim)
        self.data_q_gamma = data_q_FT_gamma + data_q_LT_gamma

        data_q_eta = analytic(self.L, self.dim, self.N, self.g, 0, 0, 0, 1)
        data_x_eta = ifftn(data_q_eta).real
        data_q_FT_eta = fftn(data_x_eta).real
        data_q_LT_eta = Laplace_Transform_ND(data_x_eta, dim=self.dim, offset=(1, ) * self.dim)
        self.data_q_eta = data_q_FT_eta + data_q_LT_eta

    def test_param(self, param_name, xlog=False, ylog=False, show=False):
        param_mem = copy(self.params[param_name])
        alphas = numpy.zeros(self.test_points)
        betas = numpy.zeros(self.test_points)

        for i, x in enumerate(self.param_ranges[param_name]):
            self.params.pop(param_name)
            self.params.update({param_name: x})

            res = analysis_ND_Laplace(self.params, self.L, dim=self.dim, no_samples=self.no_samples, g=self.g, N=self.N)

            alphas[i], betas[i] = res.x

        self.params.pop(param_name)

        plt.plot(self.param_ranges[param_name], alphas, label=r'$\alpha$ out')
        plt.xlabel(self.params_latex[param_name])
        plt.title("Data In: " + reduce(lambda a, b: a + b,
                    [rf"{self.params_latex[param]}: {self.params[param]} " for param in self.params]))
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.legend()

        if show:
            plt.show()

        else:
            plt.savefig(f"graphs/{param_name}_dependence_for_alpha" + reduce(lambda a, b: a + b, [f"_{param}{self.params[param]}" for param in self.params]) + ".pdf")

        plt.clf()

        plt.plot(self.param_ranges[param_name], betas, label=r'$\beta$ out')
        plt.xlabel(self.params_latex[param_name])
        plt.title("Data In: " + reduce(lambda a, b: a + b,
                    [rf"{self.params_latex[param]}: {self.params[param]} " for param in self.params]))
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.legend()

        if show:
            plt.show()

        else:
            plt.savefig(f"graphs/{param_name}_dependence_for_beta" + reduce(lambda a, b: a + b, [f"_{param}{self.params[param]}" for param in self.params]) + ".pdf")

        plt.clf()

        # Add the parameter back in
        self.params[param_name] = param_mem

    def plot_curves(self):
        analysis_ND_Laplace(self.params, self.L, dim=self.dim, no_samples=self.no_samples, g=self.g, plot=True)

    def plot_gamma(self, num_cuts, num_gammas, plot=False):
        cuts = numpy.linspace(0.05, 0.3, num_cuts)
        gammas = 10 ** numpy.linspace(-1, 5, num_gammas)

        fake_data_no_gamma = self.alpha * self.data_q_alpha + self.beta * self.data_q_beta + \
                self.eta * self.data_q_eta

        # To collect the results
        alphas = numpy.zeros((num_gammas, num_cuts, self.no_samples))
        betas = numpy.zeros((num_gammas, num_cuts, self.no_samples))
        etas = numpy.zeros((num_gammas, num_cuts, self.no_samples))

        for i in tqdm(range(self.no_samples)):
            noise = self.eps * numpy.random.randn(*((self.L, ) * self.dim))
            noise_x = ifftn(noise).real
            noise_q_FT = fftn(noise_x).real
            noise_q_LT = Laplace_Transform_ND(noise_x, dim=self.dim, offset=(1, ) * self.dim)
            noise_q = noise_q_FT + noise_q_LT

            for j, cut in enumerate(cuts):
                # Cut away the higher momentum modes from the data
                for d in range(self.dim):
                    if d == 0:
                        noise_data_q = numpy.take(noise_q, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        fake_data_q = numpy.take(fake_data_no_gamma, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        alpha_data_q = numpy.take(self.data_q_alpha, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        beta_data_q = numpy.take(self.data_q_beta, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        gamma_data_q = numpy.take(self.data_q_gamma, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        eta_data_q = numpy.take(self.data_q_eta, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)

                    else:
                        noise_data_q = numpy.take(noise_data_q, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        fake_data_q = numpy.take(fake_data_q, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        alpha_data_q = numpy.take(alpha_data_q, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        beta_data_q = numpy.take(beta_data_q, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        gamma_data_q = numpy.take(gamma_data_q, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)
                        eta_data_q = numpy.take(eta_data_q, numpy.arange(1, int(numpy.rint(self.L * cut))), axis=d)

                for k, gamma in enumerate(gammas):
                    noisy_data = fake_data_q + noise_data_q + gamma * gamma_data_q

                    # Now compare to model data with no q^2 piece
                    def minimize_this(x, cov_inv=None):
                        alpha, beta, eta = x
                        # Reference clean data with no gamma or epsilon
                        data_q_ref = alpha * alpha_data_q + beta * beta_data_q + eta * eta_data_q

                        residuals = data_q_ref - noisy_data
                        residuals = residuals.reshape(numpy.product(residuals.shape))

                        # normalized_residuals = numpy.dot(cov_inv, residuals)

                        return residuals

                    res = least_squares(minimize_this, [0, 0, 0], method="lm")
                    alphas[k, j, i] = res.x[0]
                    betas[k, j, i] = res.x[1]
                    etas[k, j, i] = res.x[2]

        alphas_mean = numpy.mean(alphas, axis=2)
        betas_mean = numpy.mean(betas, axis=2)
        etas_mean = numpy.mean(etas, axis=2)

        alphas_std = numpy.std(alphas, axis=2)
        betas_std = numpy.std(betas, axis=2)
        etas_std = numpy.std(etas, axis=2)

        if plot:
            for j, gamma in enumerate(gammas):
                plt.plot(cuts, alphas_mean[j] / gamma, color='k', label=f'eps/gamma = {self.eps/gamma}')

                plt.fill_between(cuts, (alphas_mean[j] - alphas_std[j]) / gamma,
                                 (alphas_mean[j] + alphas_std[j]) / gamma, alpha=0.1,
                                 color='b')

            plt.xlabel('cut')
            plt.ylabel(r'\alpha / \gamma')

            plt.show()

            for j, gamma in enumerate(gammas):
                plt.plot(cuts, betas_mean[j] / gamma, color='k', label=f'eps/gamma = {self.eps/gamma}')

                plt.fill_between(cuts, (betas_mean[j] - betas_std[j]) / gamma,
                                 (betas_mean[j] + betas_std[j]) / gamma, alpha=0.1,
                                 color='b')

            plt.xlabel('cut')
            plt.ylabel(r'\beta / \gamma')

            plt.show()

            for j, gamma in enumerate(gammas):
                plt.plot(cuts, etas_mean[j] / gamma, color='k', label=f'eps/gamma = {self.eps/gamma}')

                plt.fill_between(cuts, (etas_mean[j] - etas_std[j]) / gamma,
                                 (etas_mean[j] + etas_std[j]) / gamma, alpha=0.1,
                                 color='b')

            plt.xlabel('cut')
            plt.ylabel(r'\eta / \gamma')

            plt.xscale('log')
            plt.show()

        return alphas, betas, etas


x = explore_params(256, 2, 20, 100, gamma=0, alpha=0, eps=1, dim=2, cut=0.1, g=0.1, eta=0)
x.plot_gamma(20, 100, plot=True)
# x.plot_curves()
# x.test_param("alpha", xlog=True, ylog=True)
# x.test_param("beta", xlog=True, ylog=True)
# x.test_param("gamma", xlog=True, ylog=True)
# x.test_param("eps", xlog=True, ylog=True)
# x.test_param("cut", show=True)
