from scipy.fft import ifftn
import sys
import os
import numpy
import argparse

sys.path.append(os.getcwd())

from Core.Laplace import Laplace_Transform_ND

parser = argparse.ArgumentParser()

parser.add_argument('i', metavar="i", type=int)

args = parser.parse_args()

# Parameters
L = 256
x_max = 1
dims = 1
num_configs = 100

Noise_FT = numpy.zeros((num_configs, L))
Noise_LT = numpy.zeros((num_configs, L))

for j in range(num_configs):
    print(j)
    FT_noise = numpy.random.randn(L, L, L)
    x_noise = ifftn(FT_noise)
    LT_noise = Laplace_Transform_ND(x_noise, 3, (1, 1, 1), x_max=x_max)

    Noise_FT[j] = FT_noise[(0, ) * (3 - dims)].real
    Noise_LT[j] = LT_noise[(0, ) * (3 - dims)].real

numpy.save(f'Server/data/fake_data/data{args.i}_L{L}_xmax{x_max}_dims{dims}_configs{num_configs}_FT.npy', Noise_FT)
numpy.save(f'Server/data/fake_data/data{args.i}_L{L}_xmax{x_max}_dims{dims}_configs{num_configs}_LT.npy', Noise_LT)
