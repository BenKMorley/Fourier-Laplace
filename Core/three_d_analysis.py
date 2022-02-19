import numpy
import matplotlib.pyplot as plt
import matplotlib
from numpy.fft import fftn, ifftn
from scipy.optimize import least_squares
from tqdm import tqdm
import sys
import os
import re
import pdb
import pickle
import h5py
from ast import literal_eval


# Import from the Core directory
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Core.parameters import FL_dir

from Core.MISC import GRID_convention_g, GRID_convention_L, GRID_convention_m, GRID_convention_N
from Core.Laplace import Laplace_Transform_1D, Laplace_Transform_ND

matplotlib.rcParams.update({'text.usetex': True})


def get_result_index(f, comps1, comps2):
    result_index = None

    for i in range(len(f['FL'].keys())):
        x_ = re.findall(r'_\d+', f['FL'][f'FL_{i}'].attrs['source'][0].astype(str))
        y_ = re.findall(r'_\d+', f['FL'][f'FL_{i}'].attrs['sink'][0].astype(str))

        x = tuple([int(i[1:]) for i in x_])
        y = tuple([int(i[1:]) for i in y_])

        if x == comps1 and y == comps2:
            result_index = i
            break

        if x == comps2 and y == comps1:
            result_index = i
            break

    return result_index


def analytic(p_sq, alpha, beta, gamma, eta, N, g, dims=3):
    return N ** 2 * (p_sq / g ** 2) * (alpha * numpy.sqrt(p_sq) + beta * g * (1 / 2) *
        numpy.log(p_sq / g ** 2, out=numpy.zeros_like(p_sq), where=p_sq != 0) + gamma) + eta


class analysis_3D_one_config(object):
    def __init__(self, L, N, g, m, components1, components2, config, base_dir=FL_dir,
                 x_max=numpy.inf, offset=(1, 1, 1)):
        self.N = N
        self.L = L
        self.g = g
        self.m = m
        self.dim = 3
        self.x_max = x_max
        self.offset = offset
        self.components1 = components1
        self.components2 = components2
        self.config = config
        self.base_dir = base_dir
        self.directory = f"{base_dir}/{GRID_convention_g(g)}/{GRID_convention_N(N)}/{GRID_convention_L(L)}/{GRID_convention_m(m)}/FL/"

        self.load_in_data()

    def load_in_data(self):
        import h5py

        result_file = f'cosmhol-su{self.N}_L{self.L}_g{self.g}_m2{self.m}-FL.{self.config}.h5'

        f = h5py.File(f'{self.directory}{result_file}')

        print(f'T1: {self.components1}')
        print(f'T2: {self.components2}')

        result_index = get_result_index(f, self.components1, self.components2)

        if result_index is None:
            raise FileNotFoundError("h5 data file doesn't contain appropriate components")

        data = f['FL'][f'FL_{result_index}']['P_FULL_3D'][()]

        self.correlator_p = numpy.array(data)['re'] + 1j * numpy.array(data)['im']

    def process_p_correlator(self):
        # We subtract the disconnected part from the p = 0 point
        onept1 = pickle.load(open(f'Server/data/onept_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.components1}.pcl', 'rb'))
        onept2 = pickle.load(open(f'Server/data/onept_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.components2}.pcl', 'rb'))

        # For self consistancy check if the [0, 0, 0] component matches the product of the onept functions
        onept_dir = f"{FL_dir}/{GRID_convention_g(self.g)}/{GRID_convention_N(self.N)}/{GRID_convention_L(self.L)}/{GRID_convention_m(self.m)}/onept/"
        f = h5py.File(f'{onept_dir}cosmhol-su{self.N}_L{self.L}_g{self.g}_m2{self.m}-emtc.{self.config}.h5')
        onept_conf1 = f['emt']['value'][()][self.components1]['re'] + 1j * f['emt']['value'][()][self.components1]['im']
        onept_conf2 = f['emt']['value'][()][self.components2]['re'] + 1j * f['emt']['value'][()][self.components2]['im']

        try:
            assert ((self.correlator_p[0, 0, 0] - onept_conf1 * onept_conf2) / self.correlator_p[0, 0, 0]) < 10 ** -12

        except AssertionError:
            print("Error: Two-Point and onept don't agree")
            self.correlator_p = numpy.full((self.L, self.L, self.L), numpy.nan, dtype=numpy.complex128)

        # Keep this here for reference so the data can be cross checked
        self.onept = onept1 * onept2
        self.correlator_p[0, 0, 0] -= self.onept

    def get_x_correlator(self):
        self.correlator_x = ifftn(self.correlator_p)

    def get_LT(self):
        # We first remove entries with |x| > x_max
        x_s = numpy.indices((256, 256, 256))

        # Shift the location of the period boundary so x goes between -L / 2 and L / 2
        x_s = (x_s + self.L // 2) % self.L - self.L // 2

        mod_x_sq = x_s[0] ** 2 + x_s[1] ** 2 + x_s[2] ** 2

        # Set all x values further from the origin than x_max to 0
        # Add a small buffer so that we can use x_max for integers inclusively
        self.correlator_x[mod_x_sq - 10 ** -7 > self.x_max ** 2] = 0

        self.Laplace_p = Laplace_Transform_ND(self.correlator_x, 3, (1, 1, 1))

        # Recalculate the x correlator
        self.correlator_x = ifftn(self.correlator_p)


class full_analysis_3D(object):
    def __init__(self, L, N, g, m, T1, T2, x_max, dims, offset=(1, 1, 1), Nboot=500, p_max=1,
                 use_eta=False, seed=544287412):
        self.N = N
        self.L = L
        self.g = g
        self.m = m
        self.dims = dims
        self.x_max = x_max
        self.offset = offset
        self.T1 = T1
        self.T2 = T2
        self.p_max = p_max
        self.Nboot = Nboot
        self.use_eta = use_eta

        if use_eta:
            self.nparams = 3

        else:
            self.nparams = 2

        self.directory = 'Server/data'

        self.find_momenta()

        # The momenta to be used in the fit
        self.keep = self.p_sq[self.dims] <= self.p_max

        # Don't use the origin
        self.keep[(0, ) * dims] = 0

        # Don't use the p < 0 data as the Laplace Transform is not symmetric
        pos_p = numpy.arange(L) > L // 2

        for d in range(dims):
            numpy.put_along_axis(self.keep, numpy.arange(L)[pos_p], False, d)

        # Set the RNG seed
        numpy.random.seed(seed)

    def find_momenta(self):
        print('Calculating momenta')
        # Make the q squared and p_hat squared contributions
        p_fac = 2 * numpy.pi / self.L

        self.p_hat_sq = {}
        self.p_sq = {}
        for dims in range(1, 4):
            p_s = []  # start as list to append

            for i in range(dims):
                p_part = numpy.arange(self.L).reshape((1, ) * i + (self.L, ) + (1, ) * (dims - 1 - i))

                # Apply the periodicity of the lattice
                p_part = (p_part + self.L // 2) % self.L - self.L // 2
                p_part = p_part * p_fac

                for j in [k for k in range(0, i)] + [k for k in range(i + 1, dims)]:
                    p_part = p_part.repeat(self.L, axis=j)

                p_s.append(p_part)

            self.p_sq[dims] = numpy.zeros((self.L, ) * dims)
            self.p_hat_sq[dims] = numpy.zeros((self.L, ) * dims)

            for d in range(dims):
                p_hat = 2 * numpy.sin(p_s[d] / 2)
                self.p_hat_sq[dims] += p_hat ** 2
                self.p_sq[dims] += p_s[d] ** 2

        print('Finished Calculating Momenta')

    def generate_fake_data(self):
        print('Generating Fake data')
        # Use the linearity of the Fourier and Laplace Transforms to prerun them
        data_p_alpha = analytic(self.p_hat_sq[3], 1, 0, 0, 0, self.N, self.g)
        data_p_beta = analytic(self.p_hat_sq[3], 0, 1, 0, 0, self.N, self.g)
        data_p_gamma = analytic(self.p_hat_sq[3], 0, 0, 1, 0, self.N, self.g)
        data_p_eta = analytic(self.p_hat_sq[3], 0, 0, 0, 1, self.N, self.g)

        # Figure out the impact of the onept contribution
        data_p_onept = numpy.zeros((self.L, ) * 3)
        data_p_onept[0, 0, 0] = -1

        self.data_x_alpha = ifftn(data_p_alpha)
        self.data_x_beta = ifftn(data_p_beta)
        self.data_x_gamma = ifftn(data_p_gamma)
        self.data_x_eta = ifftn(data_p_eta)
        self.data_x_onept = ifftn(data_p_onept)

        self.data_p_FT_alpha = fftn(self.data_x_alpha)
        self.data_p_FT_beta = fftn(self.data_x_beta)
        self.data_p_FT_gamma = fftn(self.data_x_gamma)
        self.data_p_FT_eta = fftn(self.data_x_eta)
        self.data_p_FT_onept = fftn(self.data_x_onept)

        self.data_p_LT_alpha = Laplace_Transform_ND(self.data_x_alpha, dim=3, offset=self.offset, x_max=self.x_max)
        self.data_p_LT_beta = Laplace_Transform_ND(self.data_x_beta, dim=3, offset=self.offset, x_max=self.x_max)
        self.data_p_LT_gamma = Laplace_Transform_ND(self.data_x_gamma, dim=3, offset=self.offset, x_max=self.x_max)
        self.data_p_LT_eta = Laplace_Transform_ND(self.data_x_eta, dim=3, offset=self.offset, x_max=self.x_max)
        self.data_p_LT_onept = Laplace_Transform_ND(self.data_x_onept, dim=3, offset=self.offset, x_max=self.x_max)

        self.data_p_alpha = self.data_p_FT_alpha + self.data_p_LT_alpha
        self.data_p_beta = self.data_p_FT_beta + self.data_p_LT_beta
        self.data_p_gamma = self.data_p_FT_gamma + self.data_p_LT_gamma
        self.data_p_eta = self.data_p_FT_eta + self.data_p_LT_eta
        self.data_p_onept = self.data_p_FT_onept + self.data_p_LT_onept

        self.data_x_alpha = self.data_x_alpha[(0, ) * (3 - self.dims)]
        self.data_x_beta = self.data_x_beta[(0, ) * (3 - self.dims)]
        self.data_x_gamma = self.data_x_gamma[(0, ) * (3 - self.dims)]
        self.data_x_eta = self.data_x_eta[(0, ) * (3 - self.dims)]
        self.data_x_onept = self.data_x_onept[(0, ) * (3 - self.dims)]

        self.data_p_LT_alpha = self.data_p_LT_alpha[(0, ) * (3 - self.dims)]
        self.data_p_LT_beta = self.data_p_LT_beta[(0, ) * (3 - self.dims)]
        self.data_p_LT_gamma = self.data_p_LT_gamma[(0, ) * (3 - self.dims)]
        self.data_p_LT_eta = self.data_p_LT_eta[(0, ) * (3 - self.dims)]
        self.data_p_LT_onept = self.data_p_LT_onept[(0, ) * (3 - self.dims)]

        self.data_p_FT_alpha = self.data_p_FT_alpha[(0, ) * (3 - self.dims)]
        self.data_p_FT_beta = self.data_p_FT_beta[(0, ) * (3 - self.dims)]
        self.data_p_FT_gamma = self.data_p_FT_gamma[(0, ) * (3 - self.dims)]
        self.data_p_FT_eta = self.data_p_FT_eta[(0, ) * (3 - self.dims)]
        self.data_p_FT_onept = self.data_p_FT_onept[(0, ) * (3 - self.dims)]

        self.data_p_alpha = self.data_p_alpha[(0, ) * (3 - self.dims)]
        self.data_p_beta = self.data_p_beta[(0, ) * (3 - self.dims)]
        self.data_p_gamma = self.data_p_gamma[(0, ) * (3 - self.dims)]
        self.data_p_eta = self.data_p_eta[(0, ) * (3 - self.dims)]
        self.data_p_onept = self.data_p_onept[(0, ) * (3 - self.dims)]
        print('Finished Generating fake data')

    def load_in_data(self):
        print('Loading in data')
        self.configs = pickle.load(open(f'Server/data/configs_N{self.N}_g{self.g}_L{self.L}_m{self.m}.pcl', 'rb'))
        self.n_conf = len(self.configs)

        self.full_p_correlator = numpy.load(f'Server/data/full_data/Fourier_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}_{self.T2}_dims{self.dims}.npy')
        self.full_x_correlator = numpy.load(f'Server/data/full_data/Correlator_x_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}_{self.T2}_dims{self.dims}.npy')
        self.Laplace_p = numpy.load(f'Server/data/full_data/Laplace_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}_{self.T2}_xmax{self.x_max:.1f}_dims{self.dims}.npy')
        Onepts = numpy.load(f'Server/data/full_data/Onept_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}_{self.T2}_dims{self.dims}.npy')

        self.full_data = self.full_p_correlator + self.Laplace_p

        # Also load in the onept data
        self.onepts1 = numpy.load(f'Server/data/onept/onept_full_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}.npy')
        self.onepts2 = numpy.load(f'Server/data/onept/onept_full_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T2}.npy')

        onept1 = pickle.load(open(f'Server/data/onept_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}.pcl', 'rb'))
        onept2 = pickle.load(open(f'Server/data/onept_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T2}.pcl', 'rb'))
        self.onept = onept1 * onept2

        # Check that the onepts are consistant
        assert numpy.sum(numpy.abs(Onepts - self.onept) > (10 ** -15 * self.onept)) == 0

        # Check the onept and two-point functions are consistant
        assert numpy.max(numpy.abs(self.full_p_correlator[:, 0] - self.onepts1 * self.onepts2)) < (10 ** -12 * self.onept)

        # Check that we recalculate the same onept
        assert abs(numpy.mean(self.onepts1) - onept1) < abs(onept1 * 10 ** -15)
        assert abs(numpy.mean(self.onepts2) - onept2) < abs(onept2 * 10 ** -15)

        print('Finished Loading in data')

    def calculate_bootstrap(self):
        print('Calculating Bootstrap')
        self.bootstraps = numpy.random.randint(len(self.configs), size=(self.Nboot, len(self.configs)))

        self.correlator_p_samples = numpy.zeros((self.Nboot, ) + (self.L, ) * self.dims, dtype=numpy.complex128)
        self.correlator_x_samples = numpy.zeros((self.Nboot, ) + (self.L, ) * self.dims, dtype=numpy.complex128)
        self.Laplace_p_samples = numpy.zeros((self.Nboot, ) + (self.L, ) * self.dims, dtype=numpy.complex128)
        self.boot_samples = numpy.zeros((self.Nboot, ) + (self.L, ) * self.dims, dtype=numpy.complex128)

        for i in tqdm(range(self.Nboot)):
            self.correlator_p_samples[i] = numpy.mean(self.full_p_correlator[self.bootstraps[i]], axis=0)
            self.correlator_x_samples[i] = numpy.mean(self.full_x_correlator[self.bootstraps[i]], axis=0)
            self.Laplace_p_samples[i] = numpy.mean(self.Laplace_p[self.bootstraps[i]], axis=0)
            self.boot_samples[i] = numpy.mean(self.full_data[self.bootstraps[i]], axis=0)

            # Add in the effect of a different value of the onept in this case
            onepts1 = self.onepts1[self.bootstraps[i]]
            onepts2 = self.onepts2[self.bootstraps[i]]

            delta_onept = numpy.mean(onepts1) * numpy.mean(onepts2) - self.onept

            # Add in the effect of this change to the onept
            self.correlator_p_samples[i] += self.data_p_FT_onept * delta_onept
            self.correlator_x_samples[i] += self.data_x_onept * delta_onept
            self.Laplace_p_samples[i] += self.data_p_LT_onept * delta_onept

        # For the final fit we only want the real part - otherwise concepts like covariance are
        # confusing
        self.boot_samples = self.boot_samples.real
        self.correlator_p_samples = self.correlator_p_samples.real
        self.Laplace_p_samples = self.Laplace_p_samples.real

        print('Finished calculating bootstrap')

    def get_covariance_matrix(self):
        print('Finding Covariance matrix')
        if self.dims == 1:
            cov_matrix = numpy.cov(self.boot_samples[:, self.keep], rowvar=False)
            cov_1_2 = numpy.linalg.cholesky(cov_matrix)
            self.cov_inv = numpy.linalg.inv(cov_1_2)

        else:
            print("Multidimensional fitting not implemented yet")
            exit()

        print('Finished finding covariance matrix')

    def fit(self):
        print('Running fits')
        # Run the central fit
        self.fit_data = numpy.mean(self.full_data, axis=0).real[self.keep]

        def minimize_me(x):
            alpha, beta = x[0: 2]

            predictions = alpha * self.data_p_alpha.real[self.keep]
            predictions += beta * self.data_p_beta.real[self.keep]

            if self.use_eta:
                eta = x[2]
                predictions += eta * self.data_p_eta.real[self.keep]

            residuals = self.fit_data - predictions

            normalized_residuals = numpy.dot(self.cov_inv, residuals)

            return normalized_residuals

        if self.use_eta:
            x0 = [0, 0, 0]

        else:
            x0 = [0, 0]

        res = least_squares(minimize_me, x0, method="lm")

        self.fit_central = res.x

        # Make new bootstrap indices
        self.bootstraps = numpy.random.randint(len(self.configs), size=(self.Nboot, len(self.configs)))
        self.fit_params = numpy.zeros((self.Nboot, self.nparams))

        for i in range(self.Nboot):
            results = numpy.mean(self.full_data[self.bootstraps[i]], axis=0).real[self.keep]

            def minimize_me(x):
                alpha = x[0]
                beta = x[1]

                if self.use_eta:
                    eta = x[2]

                predictions = alpha * self.data_p_alpha.real[self.keep]
                predictions += beta * self.data_p_beta.real[self.keep]

                if self.use_eta:
                    predictions += eta * self.data_p_eta.real[self.keep]

                residuals = results - predictions

                normalized_residuals = numpy.dot(self.cov_inv, residuals)

                return normalized_residuals

            res = least_squares(minimize_me, x0, method="lm")

            self.fit_params[i] = res.x

        print('Finished running fits')

    def plot_fit(self):
        print('Plotting the fit')
        # Plot the actual data
        mean_data = numpy.mean(self.full_data, axis=0).real[self.keep]
        std_data = numpy.std(self.boot_samples, axis=0)[self.keep]

        mean_data_FT = numpy.mean(self.full_p_correlator, axis=0).real[self.keep]
        std_data_FT = numpy.std(self.correlator_p_samples, axis=0)[self.keep]

        mean_data_LT = numpy.mean(self.Laplace_p, axis=0).real[self.keep]
        std_data_LT = numpy.std(self.Laplace_p_samples, axis=0)[self.keep]

        p_s = numpy.sqrt(self.p_sq[1])[self.keep]

        plt.plot(p_s, mean_data, color='k')
        plt.fill_between(p_s, mean_data - std_data, mean_data + std_data, alpha=0.2, color='k', label='data total')
        plt.fill_between(p_s, mean_data_FT - std_data_FT, mean_data_FT + std_data_FT, alpha=0.2, color='g', label='data FT')
        plt.fill_between(p_s, mean_data_LT - std_data_LT, mean_data_LT + std_data_LT, alpha=0.2, color='r', label='data LT')

        # Now add in the fit result
        alpha, beta = self.fit_central[0: 2]
        predictions = alpha * self.data_p_alpha.real[self.keep]
        predictions += beta * self.data_p_beta.real[self.keep]

        predictions_FT = alpha * self.data_p_FT_alpha.real[self.keep]
        predictions_FT += beta * self.data_p_FT_beta.real[self.keep]

        predictions_LT = alpha * self.data_p_LT_alpha.real[self.keep]
        predictions_LT += beta * self.data_p_LT_beta.real[self.keep]

        if self.use_eta:
            eta = self.fit_central[2]
            predictions += eta * self.data_p_eta.real[self.keep]
            predictions_FT += eta * self.data_p_FT_eta.real[self.keep]
            predictions_LT += eta * self.data_p_LT_eta.real[self.keep]

            delta_alpha, delta_beta, delta_eta = numpy.std(self.fit_params, axis=0)
            plt.title(rf'$\alpha = {alpha} \pm {delta_alpha}$, $\beta={beta} \pm {delta_beta}$, $\eta={eta} \pm {delta_eta}$')

        else:
            delta_alpha, delta_beta = numpy.std(self.fit_params, axis=0)
            plt.title(rf'$\alpha = {alpha} \pm {delta_alpha}$, $\beta={beta} \pm {delta_beta}$')

        plt.plot(p_s, predictions, ls='--', color='k', label='fit total')
        plt.plot(p_s, predictions_FT, ls='--', color='g', label='fit FT')
        plt.plot(p_s, predictions_LT, ls='--', color='r', label='fit LT')

        plt.xlabel(r'$p$')
        x0, y0 = literal_eval(self.T1)
        x1, y1 = literal_eval(self.T2)

        plt.ylabel(rf'$\langle T{x0}{y0} T{x1}{y1} \rangle (p)$')
        plt.legend()
        plt.show()

        print('Finished Plotting')

        return None
