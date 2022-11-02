import pdb
import sys
import os
import numpy
from scipy.fft import ifftn
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool

sys.path.append(os.getcwd())

from Core.Laplace import Laplace_Transform_ND
from Core.three_d_analysis import full_analysis_3D


# Parameters
N = 2
L = 256
T1 = '(0, 0)'
T2 = '(1, 1)'
g = 0.2
m = -0.062
x_max = 1
dims = 1
alpha = 1
beta = 1
eta = 1
quality = 100
parrallel = False

# Create a range of noises
eps_s = 10 ** numpy.linspace(-1, 3, 56)

# Use a reasonable number of configs
num_configs = quality

# Make the data
fit_params = numpy.zeros((quality, 3))
fit_sigma = numpy.zeros((quality, 3))


def run(eps):
    a = full_analysis_3D(L, N, g, m, T1, T2, x_max, dims, use_eta=True, seed=int(eps * 10 ** 4),
                         Nboot=quality, configs=numpy.arange(num_configs))
    a.generate_fake_data()
    a.full_data = numpy.zeros((num_configs, L), dtype=numpy.complex128)
    a.full_p_correlator = numpy.zeros((num_configs, L), dtype=numpy.complex128)
    a.full_x_correlator = numpy.zeros((num_configs, L), dtype=numpy.complex128)
    a.Laplace_p = numpy.zeros((num_configs, L), dtype=numpy.complex128)

    base_data_FT = alpha * a.data_p_FT_alpha + beta * a.data_p_FT_beta + eta * a.data_p_FT_eta
    base_data_LT = alpha * a.data_p_LT_alpha + beta * a.data_p_LT_beta + eta * a.data_p_LT_eta

    # Set the data ourselves
    for i in tqdm(range(num_configs)):
        FT_noise = eps * (numpy.random.rand(L, L, L) - 0.5)
        x_noise = ifftn(FT_noise)
        LT_noise = Laplace_Transform_ND(x_noise, 3, (1, 1, 1), x_max=x_max)

        a.full_p_correlator[i] = base_data_FT + FT_noise[(0, ) * (3 - a.dims)]
        a.Laplace_p[i] = base_data_LT + LT_noise[(0, ) * (3 - a.dims)]
        a.full_data[i] = a.full_p_correlator[i] + a.Laplace_p[i]

    # The onept variation in this method is insignificant compared to the rest of the noise
    # so set it to 1
    a.onepts1 = numpy.zeros(num_configs)
    a.onepts2 = numpy.zeros(num_configs)
    a.onept = 0

    a.calculate_bootstrap()
    a.get_covariance_matrix()
    a.fit()

    return a.fit_params


p = Pool(os.cpu_count())
results = p.map(run, eps_s)
results = numpy.array(results)
fit_params = numpy.mean(results, axis=1)
fit_sigma = numpy.std(results, axis=1)

numpy.save('Server/data/fake_data/results.npy')

for i, param in enumerate(['alpha', 'beta', 'eta']):
    params = fit_params[:, i]
    sigmas = fit_sigma[:, i]

    plt.plot(eps_s, params, color='k', ls='--')
    plt.fill_between(eps_s, params - sigmas, params + sigmas, alpha=0.2, color='k')
    plt.title(f'{param}')
    plt.xlabel('eps')
    plt.xscale('log')
    plt.show()
