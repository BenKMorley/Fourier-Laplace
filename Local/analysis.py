import numpy
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn
from scipy.optimize import minimize, least_squares
from tqdm import tqdm
import sys
import os
import re
import pdb
from Joseph_comparison import Laplace_Transform_ND

# Import from the Core directory
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../Core')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.MISC import GRID_convention_g, GRID_convention_L, GRID_convention_m, GRID_convention_N


class analysis():
    def __init__(self, L, N, g, m, components1, components2, no_samples=100, base_dir="/mnt/drive2/Fourier-Laplace/data", subtract=True, dim=3):
        self.N = N
        self.L = L
        self.g = g
        self.subtract_mean = subtract
        self.dim = dim
        self.q_s = numpy.arange(self.L) * numpy.pi * 2 / L
        self.no_samples = no_samples
        self.components1 = components1
        self.components2 = components2
        self.directory = f"{base_dir}/{GRID_convention_g(g)}/{GRID_convention_N(N)}/{GRID_convention_L(L)}/{GRID_convention_m(m)}/config"

        self.find_momenta()

        self.get_prediction_curves()

        self.detect_configs()

    def analytic(self, alpha, beta, gamma, eta):
        q_sq = self.q_hat_sq

        return self.N ** 2 * (q_sq / self.g ** 2) * (alpha * numpy.sqrt(q_sq) + beta * self.g * (1 / 2) *
            numpy.log(q_sq / self.g ** 2, out=numpy.zeros_like(q_sq), where=q_sq != 0) + gamma) + eta

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

    def get_prediction_curves(self):
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

    def detect_configs(self):
        files = os.popen(f'ls {self.directory}/emtc*')
        x1, y1 = self.components1
        x2, y2 = self.components2

        configs1 = []
        configs2 = []

        for name in files:
            if (len(re.findall(rf'emtc_{x1}_{y1}_\d+_real', name)) != 0):
                config = int(re.findall(r'\d+_real', name)[0][:-5])
                configs1.append(config)

        # We only want configs that have both correlators
        self.configs = numpy.sort(numpy.array(list(set(configs1).union(set(configs2)))))

        self.configs = self.configs[:50]

    def get_mom_space(self, config):
        x1, y1 = self.components1
        x2, y2 = self.components2

        # Reshape the data which has been saved linearlly
        data_1 = numpy.loadtxt(f"{self.directory}/emtc_{x1}_{y1}_{config}_real.txt").reshape((self.L, self.L, self.L))
        data_2 = numpy.loadtxt(f"{self.directory}/emtc_{x2}_{y2}_{config}_real.txt").reshape((self.L, self.L, self.L))

        # Subtract the mean of each operator
        if self.subtract_mean:
            data_1 = data_1 - numpy.mean(data_1)
            data_2 = data_2 - numpy.mean(data_2)

        data_1_p = fftn(data_1)

        # Use the opposite momentum for the second correlator - do this by taking the conjugate
        data_2_p = numpy.conj(fftn(data_2))

        # Correlator is given by the product of these two variables
        result = data_1_p * data_2_p

        return result

    def cut_dependence(self, plot=True, rerun=False, use_octant=True, axis=0, plot_errors=False, fitting_dims=1):
        cuts = numpy.arange(4, self.L // 2) * numpy.pi * 2 / self.L
        num_cuts = len(cuts)

        x1, y1 = self.components1
        x2, y2 = self.components2

        # Try to extract saved results
        try:
            alphas = numpy.load(f"Local/data/emtc_{x1}_{y1}_emtc_{x2}_{y2}_alphas_L{self.L}_samples{self.no_samples}.npy")
            betas = numpy.load(f"Local/data/emtc_{x1}_{y1}_emtc_{x2}_{y2}_betas_L{self.L}_samples{self.no_samples}.npy")
            etas = numpy.load(f"Local/data/emtc_{x1}_{y1}_emtc_{x2}_{y2}_etas_L{self.L}_samples{self.no_samples}.npy")

        except Exception:
            rerun = True

        rerun = True
        if rerun:
            # To collect the results
            alphas = numpy.zeros((num_cuts, self.no_samples))
            betas = numpy.zeros((num_cuts, self.no_samples))
            etas = numpy.zeros((num_cuts, self.no_samples))

            mom_data = numpy.zeros((len(self.configs), ) + (self.L, ) * fitting_dims)

            # All directions are identical by construction
            alpha_data_q = self.data_q_alpha[(0, ) * (self.dim - fitting_dims)]
            beta_data_q = self.data_q_beta[(0, ) * (self.dim - fitting_dims)]
            eta_data_q = self.data_q_eta[(0, ) * (self.dim - fitting_dims)]

            for i, config in tqdm(enumerate(self.configs)):
                correlator_p = self.get_mom_space(config)

                # Get the correlator in position space
                correlator_x = ifftn(correlator_p)

                # Take the Laplace Transform
                Laplace_p = Laplace_Transform_ND(correlator_x, self.dim, (1, ) * self.dim)

                # Add the two contributions together
                result_p = correlator_p + Laplace_p

                # Keep a 1D sample of the correlator in the 3 available directions
                mom_data[i] = result_p[(0, ) * (self.dim - fitting_dims)]

            pdb.set_trace()
            # Flatten the mom_data
            mom_data_flat = mom_data.reshape(len(self.configs), numpy.product(mom_data.shape[1:]))

            for j, cut in enumerate(cuts):
                # Flatten the momentum
                q_sq = self.q_sq[(0, ) * (self.dim - fitting_dims)]
                q_sq = q_sq.reshape(numpy.product(q_sq.shape))

                keep = q_sq <= cut + 10 ** -15

                # Keep only the momenta that satisfy the cut
                mom_data_keep = mom_data_flat[:, keep]

                # Use a frozen covariance matrix
                cov_matrix = numpy.cov(mom_data_keep.T)

                # Remove the origin as it contains the disconnected contribution we don't want
                keep[(0, ) * len(keep.shape)] = False

                keep = keep.reshape(numpy.product(keep.shape))

                # Invert the covariance matrix
                cov_1_2 = numpy.linalg.cholesky(cov_matrix)
                cov_inv = numpy.linalg.inv(cov_1_2)

                alpha_data = alpha_data_q[keep]
                beta_data = beta_data_q[keep]
                eta_data = eta_data_q[keep]

                indices = numpy.random.randint(len(self.configs), size=(self.no_samples, len(self.configs)))

                for i in range(self.no_samples):
                    mom_data_boot = mom_data_flat[indices[i]]

                    mom_piece = numpy.mean(mom_data_boot, axis=0)[keep]

                    # Now compare to model data with no q^2 piece
                    def minimize_this(x):
                        alpha, beta, eta = x

                        # Reference clean data with no gamma or epsilon
                        data_q_ref = alpha * alpha_data + beta * beta_data + eta * eta_data

                        residuals = data_q_ref - mom_piece
                        residuals = numpy.dot(cov_inv, residuals)

                        return residuals

                    res = least_squares(minimize_this, [0, 0, 0], method="lm")

                    alphas[j, i] = res.x[0]
                    betas[j, i] = res.x[1]
                    etas[j, i] = res.x[2]

            numpy.save(f"Local/data/emtc_{x1}_{y1}_emtc_{x2}_{y2}_alphas_L{self.L}_samples{self.no_samples}.npy", alphas)
            numpy.save(f"Local/data/emtc_{x1}_{y1}_emtc_{x2}_{y2}_betas_L{self.L}_samples{self.no_samples}.npy", betas)
            numpy.save(f"Local/data/emtc_{x1}_{y1}_emtc_{x2}_{y2}_etas_L{self.L}_samples{self.no_samples}.npy", etas)

        alphas_mean = numpy.mean(alphas, axis=1)
        betas_mean = numpy.mean(betas, axis=1)
        etas_mean = numpy.mean(etas, axis=1)

        alphas_std = numpy.std(alphas, axis=1)
        betas_std = numpy.std(betas, axis=1)
        etas_std = numpy.std(etas, axis=1)

        if plot:
            plt.fill_between(cuts, alphas_mean - alphas_std, alphas_mean + alphas_std, alpha=0.1, color='k')
            plt.title(f"Alphas: T {self.components1} T {self.components2}")
            plt.xlabel(r'$|q|_{max}$')
            plt.show()

            plt.fill_between(cuts, betas_mean - betas_std, betas_mean + betas_std, alpha=0.1, color='k')
            plt.title(f"Betas: T {self.components1} T {self.components2}")
            plt.xlabel(r'$|q|_{max}$')
            plt.show()

            plt.fill_between(cuts, etas_mean - etas_std, etas_mean + etas_std, alpha=0.1, color='k')
            plt.title(f"Etas: T {self.components1} T {self.components2}")
            plt.xlabel(r'$|q|_{max}$')
            plt.show()

        if plot_errors:
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
            # plt.savefig(f"graphs/cut_dependance_L{self.L}_alpha.pdf")
            plt.title(f"T {self.components1} T {self.components2}")
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
            # plt.savefig(f"graphs/cut_dependance_L{self.L}_beta.pdf")
            plt.title(f"T {self.components1} T {self.components2}")
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
            # plt.savefig(f"graphs/cut_dependance_L{self.L}_eta.pdf")
            plt.title(f"T {self.components1} T {self.components2}")
            plt.show()

        return alphas, betas, etas
