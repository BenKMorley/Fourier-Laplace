import numpy
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn
from scipy.optimize import minimize, least_squares
from tqdm import tqdm
import sys
import os
import re
import pdb
import pickle

# Import from the Core directory
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Server.parameters import FL_dir

from Core.MISC import GRID_convention_g, GRID_convention_L, GRID_convention_m, GRID_convention_N
from Core.Laplace import Laplace_Transform_1D, Laplace_Transform_ND


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
        self.process_p_correlator()
        self.get_x_correlator()

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

        self.correlator_p[0, 0, 0] -= onept1 * onept2

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
    def __init__(self, L, N, g, m, T1, T2, x_max=numpy.inf, offset=(1, 1, 1),
                 dims=1):
        self.N = N
        self.L = L
        self.g = g
        self.m = m
        self.dims = dims
        self.x_max = x_max
        self.offset = offset
        self.T1 = T1
        self.T2 = T2

        self.directory = 'Server/data'

        self.load_in_data()
        self.find_momenta()

    def find_momenta(self):
        # Make the q squared and p_hat squared contributions
        p_s = []  # start as list to append
        p_fac = 2 * numpy.pi / self.L

        for i in range(3):
            p_part = numpy.arange(self.L).reshape((1, ) * i + (self.L, ) + (1, ) * (2 - i))

            # Apply the periodicity of the lattice
            p_part = (p_part + self.L // 2) % self.L - self.L // 2
            p_part = p_part * p_fac

            for j in range(2):
                p_part.repeat(self.L, axis=i)

            p_s.append(p_part)

        self.p_sq = numpy.zeros((self.L, ) * 3)
        self.p_hat_sq = numpy.zeros((self.L, ) * 3)

        for d in range(3):
            p_hat = 2 * numpy.sin(p_s[d] / 2)
            self.p_hat_sq += p_hat ** 2
            self.p_sq += p_s[d] ** 2

    def analytic(self, alpha, beta, gamma, eta):
        p_sq = self.p_hat_sq

        return self.N ** 2 * (p_sq / self.g ** 2) * (alpha * numpy.sqrt(p_sq) + beta * self.g * (1 / 2) *
            numpy.log(p_sq / self.g ** 2, out=numpy.zeros_like(p_sq), where=p_sq != 0) + gamma) + eta

    def generate_fake_data(self):
        # Use the linearity of the Fourier and Laplace Transforms to prerun them
        data_p_alpha = self.analytic(1, 0, 0, 0)
        data_p_beta = self.analytic(0, 1, 0, 0)
        data_p_gamma = self.analytic(0, 0, 1, 0)
        data_p_eta = self.analytic(0, 0, 0, 1)

        # Figure out the impact of the onept contribution
        data_p_onept = numpy.zeros((self.L, ) * 3)
        data_p_onept[0, 0, 0] = -1
        
        self.data_x_alpha = ifftn(data_p_alpha)
        self.data_x_beta = ifftn(data_p_beta)
        self.data_x_gamma = ifftn(data_p_gamma)
        self.data_x_eta = ifftn(data_p_eta)
        self.data_x_onept = ifftn(data_p_onept)

        self.data_p_FT_alpha = fftn(self.data_x_alpha)[(0, ) * (3 - self.dims)]
        self.data_p_FT_beta = fftn(self.data_x_beta)[(0, ) * (3 - self.dims)]
        self.data_p_FT_gamma = fftn(self.data_x_gamma)[(0, ) * (3 - self.dims)]
        self.data_p_FT_eta = fftn(self.data_x_eta)[(0, ) * (3 - self.dims)]
        self.data_p_FT_onept = fftn(self.data_x_onept)[(0, ) * (3 - self.dims)]

        self.data_p_LT_alpha = Laplace_Transform_ND(self.data_x_alpha, dim=3, offset=(1, ) * self.dim, x_max=self.x_max)[(0, ) * (3 - self.dims)]
        self.data_p_LT_beta = Laplace_Transform_ND(self.data_x_beta, dim=3, offset=(1, ) * self.dim, x_max=self.x_max)[(0, ) * (3 - self.dims)]
        self.data_p_LT_gamma = Laplace_Transform_ND(self.data_x_gamma, dim=3, offset=(1, ) * self.dim, x_max=self.x_max)[(0, ) * (3 - self.dims)]
        self.data_p_LT_eta = Laplace_Transform_ND(self.data_x_eta, dim=3, offset=(1, ) * self.dim, x_max=self.x_max)[(0, ) * (3 - self.dims)]
        self.data_p_LT_onept = Laplace_Transform_ND(self.data_x_onept, dim=3, offset=(1, ) * self.dim, x_max=self.x_max)[(0, ) * (3 - self.dims)]

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

    def load_in_data(self):
        self.configs = pickle.load(open(f'Server/data/configs_N{self.N}_g{self.g}_L{self.L}_m{self.m}.pcl', 'wb'))
        self.full_p_correlator = numpy.zeros((len(self.configs),) + (self.L, ) * self.dims)
        self.full_x_correlator = numpy.zeros((len(self.configs),) + (self.L, ) * self.dims)
        self.Laplace_p = numpy.zeros((len(self.configs),) + (self.L, ) * self.dims)

        for i, config in enumerate(self.configs):
            self.full_p_correlator[i] = numpy.load(f'Server/data/Correlator_p_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}_{self.T2}_config{config}_dims{self.dims}.npy')
            self.full_x_correlator[i] = numpy.load(f'Server/data/Correlator_x_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}_{self.T2}_config{config}_dims{self.dims}.npy')
            self.Laplace_p[i] = numpy.load(f'Server/data/Laplace_p_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}_{self.T2}_config{config}_dims{self.dims}.npy')

        self.full_data = self.full_p_correlator + self.Laplace_p

        # Also load in the onept data
        self.onepts1 = numpy.load(f'Server/data/onept/onept_full_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}.npy')
        self.onepts2 = numpy.load(f'Server/data/onept/onept_full_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T2}.npy')

        onept1 = pickle.load(open(f'Server/data/onept_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T1}.pcl', 'rb'))
        onept2 = pickle.load(open(f'Server/data/onept_N{self.N}_g{self.g}_L{self.L}_m{self.m}_{self.T2}.pcl', 'rb'))

        self.onept = onept1 * onept2 

    def calculate_bootstrap(self):
        bootstraps = numpy.random.randint(len(self.configs), size=(self.num_bootstraps,
                                                                   len(self.configs)))

        self.correlator_p_samples = numpy.zeros((self.num_boostraps, ) + (self.L, ) * (3 - self.dims))
        self.correlator_x_samples = numpy.zeros((self.num_boostraps, ) + (self.L, ) * (3 - self.dims))
        self.Laplace_p_samples = numpy.zeros((self.num_boostraps, ) + (self.L, ) * (3 - self.dims))

        for i in range(self.num_bootstraps):
            self.correlator_p_samples[i] = numpy.mean(self.full_p_correlator[bootstraps[i]], axis=0)
            self.correlator_x_samples[i] =  numpy.mean(self.full_x_correlator[bootstraps[i]], axis=0)
            self.Laplace_p_samples[i] =  numpy.mean(self.Laplace_p[bootstraps[i]], axis=0)

            # Add in the effect of a different value of the onept in this case
            onepts1 = self.onepts1[bootstraps[i]]
            onepts2 = self.onepts2[bootstraps[i]]

            delta_onept = numpy.mean(onepts1) * numpy.mean(onepts2) - self.onept

            # Add in the effect of this change to the onept
            self.correlator_p_samples += self.data_p_FT_onept * delta_onept
            self.correlator_x_samples += self.data_x_onept * delta_onept
            self.Laplace_p_samples += self.data_p_LT_onept * delta_onept

        self.boot_samples = self.correlator_p_samples + self.Laplace_p_samples

    def get_covariance_matrix(self):
        cov_matrix = numpy.cov(self.boot_samples, rowvar=True)
        cov_1_2 = numpy.linalg.cholesky(cov_matrix)
        self.cov_inv = numpy.linalg.inv(cov_1_2)

    def fit(self):
        results = numpy.mean(self.full_data, axis=0)

        def minimize_me(alpha, beta, gamma, eta):
            p_correlator = alpha * self.data_p_alpha
            p_correlator += beta * self.data_p_beta
            p_correlator += gamma * self.data_p_gamma
            p_correlator += eta * self.data_p_eta

            x_corrleator = ifftn(p_correlator)
            FT = fftn(x_corrleator)[0, 0]
            LT = Laplace_Transform_ND(x_corrleator, 3, (1, 1, 1), x_max=self.x_max)
            predictions = FT + LT

            residuals = results - predictions

            normalized_residuals = numpy.dot(self.cov_inv, residuals)

            return normalized_residuals

        res = least_squares(minimize_me, [0, 0, 0, 0], method="lm")

        print(res.x)





