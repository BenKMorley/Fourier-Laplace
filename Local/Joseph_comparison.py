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

    return N ** 2 * (q_sq / g ** 2) * (alpha * numpy.sqrt(q_sq) + beta * g * (1 / 2) *
        numpy.log(q_sq / g ** 2, out=numpy.zeros_like(q_sq), where=q_sq != 0) + gamma) + eta


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

        self.find_momenta()

        # Use the linearity of the Fourier and Laplace Transforms to prerun them
        data_q_alpha = self.analytic(1, 0, 0, 0)
        data_x_alpha = ifftn(data_q_alpha).real
        data_q_FT_alpha = fftn(data_x_alpha).real
        data_q_LT_alpha = Laplace_Transform_ND(data_x_alpha, dim=self.dim, offset=(1, ) * self.dim)
        self.data_q_alpha = data_q_FT_alpha + data_q_LT_alpha

        data_q_beta = self.analytic(0, 1, 0, 0)
        data_x_beta = ifftn(data_q_beta).real
        data_q_FT_beta = fftn(data_x_beta).real
        data_q_LT_beta = Laplace_Transform_ND(data_x_beta, dim=self.dim, offset=(1, ) * self.dim)
        self.data_q_beta = data_q_FT_beta + data_q_LT_beta

        data_q_gamma = self.analytic(0, 0, 1, 0)
        data_x_gamma = ifftn(data_q_gamma).real
        data_q_FT_gamma = fftn(data_x_gamma).real
        data_q_LT_gamma = Laplace_Transform_ND(data_x_gamma, dim=self.dim, offset=(1, ) * self.dim)
        self.data_q_gamma = data_q_FT_gamma + data_q_LT_gamma

        data_q_eta = self.analytic(0, 0, 0, 1)
        data_x_eta = ifftn(data_q_eta).real
        data_q_FT_eta = fftn(data_x_eta).real
        data_q_LT_eta = Laplace_Transform_ND(data_x_eta, dim=self.dim, offset=(1, ) * self.dim)
        self.data_q_eta = data_q_FT_eta + data_q_LT_eta

    def find_momenta(self):
        # Make the q squared and q_hat squared contributions
        q_s = []  # start as list to append
        q_fac = 2 * numpy.pi / self.L

        for i in range(self.dim):
            q_part = numpy.arange(self.L).reshape((1, ) * i + (self.L, ) + (1, ) * (self.dim - i - 1))

            # Apply the periodicity of the lattice
            q_part = (q_part + self.L // 2) % self.L - self.L // 2
            q_part = q_part * q_fac

            for j in range(self.dim - 1):
                q_part.repeat(self.L, axis=i)

            q_s.append(q_part)

        self.q_sq = numpy.zeros((self.L, ) * self.dim)
        self.q_hat_sq = numpy.zeros((self.L, ) * self.dim)

        for d in range(self.dim):
            q_hat = 2 * numpy.sin(q_s[d] / 2)
            self.q_hat_sq += q_hat ** 2
            self.q_sq += q_s[d] ** 2

    def analytic(self, alpha, beta, gamma, eta):
        q_sq = self.q_hat_sq

        return self.N ** 2 * (q_sq / self.g ** 2) * (alpha * numpy.sqrt(q_sq) + beta * self.g * (1 / 2) *
            numpy.log(q_sq / self.g ** 2, out=numpy.zeros_like(q_sq), where=q_sq != 0) + gamma) + eta

    def plot_curves(self):
        analysis_ND_Laplace(self.params, self.L, dim=self.dim, no_samples=self.no_samples, g=self.g, plot=True)

    def cut_dependence(self, plot=False, gamma=1, eps=1, rerun=False, use_octant=True):
        cut_ints = numpy.arange(2, self.L // 2 + 1)
        cuts = numpy.arange(2, self.L // 2 + 1) * numpy.pi * 2 / self.L
        num_cuts = len(cuts)
        fake_data_no_gamma = self.alpha * self.data_q_alpha + self.beta * self.data_q_beta + \
                      self.eta * self.data_q_eta

        # Try to extract saved results
        try:
            alphas = numpy.load(f"data/alphas_L{self.L}_samples{self.no_samples}.npy")
            betas = numpy.load(f"data/betas_L{self.L}_samples{self.no_samples}.npy")
            etas = numpy.load(f"data/etas_L{self.L}_samples{self.no_samples}.npy")

        except Exception:
            rerun = True

        if rerun:
            # To collect the results
            alphas = numpy.zeros((num_cuts, self.no_samples))
            betas = numpy.zeros((num_cuts, self.no_samples))
            etas = numpy.zeros((num_cuts, self.no_samples))

            for i in tqdm(range(self.no_samples)):
                noise = self.eps * numpy.random.randn(*((self.L, ) * self.dim))
                noise_x = ifftn(noise).real
                noise_q_FT = fftn(noise_x).real
                noise_q_LT = Laplace_Transform_ND(noise_x, dim=self.dim, offset=(1, ) * self.dim)
                noise_q = noise_q_FT + noise_q_LT

                for j, cut in enumerate(cuts):
                    # Use only one octant of the data as the signal is the same for all octants
                    if use_octant:
                        indices = numpy.arange(cut_ints[j] + 1)

                        for d in range(self.dim):
                            if d == 0:
                                noise_data_q = numpy.take(noise_q, indices, axis=d)
                                fake_data_q = numpy.take(fake_data_no_gamma, indices, axis=d)
                                alpha_data_q = numpy.take(self.data_q_alpha, indices, axis=d)
                                beta_data_q = numpy.take(self.data_q_beta, indices, axis=d)
                                gamma_data_q = numpy.take(self.data_q_gamma, indices, axis=d)
                                eta_data_q = numpy.take(self.data_q_eta, indices, axis=d)
                                q_sq = numpy.take(self.q_sq, indices, axis=d)

                            else:
                                noise_data_q = numpy.take(noise_data_q, indices, axis=d)
                                fake_data_q = numpy.take(fake_data_q, indices, axis=d)
                                alpha_data_q = numpy.take(alpha_data_q, indices, axis=d)
                                beta_data_q = numpy.take(beta_data_q, indices, axis=d)
                                gamma_data_q = numpy.take(gamma_data_q, indices, axis=d)
                                eta_data_q = numpy.take(eta_data_q, indices, axis=d)
                                q_sq = numpy.take(q_sq, indices, axis=d)

                        mask = q_sq < (cut ** 2 + 10 ** -10)
                        noisy_data = fake_data_q + noise_data_q + gamma * gamma_data_q

                    else:
                        mask = self.q_sq < (cut ** 2 + 10 ** -10)
                        noisy_data = fake_data_no_gamma + noise_q + gamma * self.data_q_gamma
                        alpha_data_q = self.data_q_alpha
                        beta_data_q = self.data_q_beta
                        eta_data_q = self.data_q_eta

                    # Ignore the troublesome origin point
                    mask[(0,) * self.dim] = 0

                    # Now compare to model data with no q^2 piece
                    def minimize_this(x, cov_inv=None):
                        alpha, beta, eta = x
                        # Reference clean data with no gamma or epsilon
                        data_q_ref = alpha * alpha_data_q + beta * beta_data_q + eta * eta_data_q

                        residuals = (data_q_ref - noisy_data) * mask
                        residuals = data_q_ref - noisy_data

                        residuals = residuals.reshape(numpy.product(residuals.shape))

                        return residuals

                    res = least_squares(minimize_this, [0, 0, 0], method="lm")
                    alphas[j, i] = res.x[0]
                    betas[j, i] = res.x[1]
                    etas[j, i] = res.x[2]

            numpy.save(f"data/alphas_L{self.L}_samples{self.no_samples}.npy", alphas)
            numpy.save(f"data/betas_L{self.L}_samples{self.no_samples}.npy", betas)
            numpy.save(f"data/etas_L{self.L}_samples{self.no_samples}.npy", etas)

        alphas_mean = numpy.mean(alphas, axis=1)
        betas_mean = numpy.mean(betas, axis=1)
        etas_mean = numpy.mean(etas, axis=1)

        alphas_std = numpy.std(alphas, axis=1)
        betas_std = numpy.std(betas, axis=1)
        etas_std = numpy.std(etas, axis=1)

        if plot:
            color_sys = 'k'
            color_stat = 'b'
            axis_font = {'size': '20'}

            fig, ax1 = plt.subplots()

            ax1.set_xlabel(r'$|q|_{max}$', rotation=0)
            ax1.set_ylabel(r'$\frac{\Delta \alpha}{\gamma}$', color=color_sys, rotation=0, **axis_font)
            lns1 = ax1.plot(cuts, alphas_mean, color=color_sys, label=f'systematic error')
            ax1.tick_params(axis='y', labelcolor=color_sys)

            ax2 = ax1.twinx()
            ax2.set_ylabel(r'$\frac{\sigma_\alpha}{\epsilon}$', color=color_stat, rotation=0, **axis_font)  # we already handled the x-label with ax1
            lns2 = ax2.plot(cuts, alphas_std, color='b', label='statistical error')
            ax2.tick_params(axis='y', labelcolor=color_stat)

            lns = lns1 + lns2
            labs = [ln.get_label() for ln in lns]
            ax1.legend(lns, labs)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig(f"graphs/cut_dependance_L{self.L}_alpha.pdf")
            plt.show()

            fig, ax1 = plt.subplots()

            ax1.set_xlabel(r'$|q|_{max}$', rotation=0)
            ax1.set_ylabel(r'$\frac{\Delta \beta}{\gamma}$', color=color_sys, rotation=0, **axis_font)
            lns1 = ax1.plot(cuts, betas_mean, color=color_sys, label=f'systematic error')
            ax1.tick_params(axis='y', labelcolor=color_sys)

            ax2 = ax1.twinx()
            ax2.set_ylabel(r'$\frac{\sigma_\beta}{\epsilon}$', color=color_stat, rotation=0, **axis_font)  # we already handled the x-label with ax1
            lns2 = ax2.plot(cuts, betas_std, color='b', label='statistical error')
            ax2.tick_params(axis='y', labelcolor=color_stat)

            lns = lns1 + lns2
            labs = [ln.get_label() for ln in lns]
            ax1.legend(lns, labs)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig(f"graphs/cut_dependance_L{self.L}_beta.pdf")
            plt.show()

            fig, ax1 = plt.subplots()

            ax1.set_xlabel(r'$|q|_{max}$', rotation=0)
            ax1.set_ylabel(r'$\frac{\Delta \eta}{\gamma}$', color=color_sys, rotation=0, **axis_font)
            lns1 = ax1.plot(cuts, etas_mean, color=color_sys, label=f'systematic error')
            ax1.tick_params(axis='y', labelcolor=color_sys)

            ax2 = ax1.twinx()
            ax2.set_ylabel(r'$\frac{\sigma_\eta}{\epsilon}$', color=color_stat, rotation=0, **axis_font)  # we already handled the x-label with ax1
            lns2 = ax2.plot(cuts, etas_std, color='b', label='statistical error')
            ax2.tick_params(axis='y', labelcolor=color_stat)

            lns = lns1 + lns2
            labs = [ln.get_label() for ln in lns]
            ax1.legend(lns, labs)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig(f"graphs/cut_dependance_L{self.L}_eta.pdf")
            plt.show()

        return alphas, betas, etas
