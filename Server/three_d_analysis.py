import numpy
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn
from scipy.optimize import minimize, least_squares
from tqdm import tqdm
import sys
import os
import re
import pdb

# Import from the Core directory
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Server.parameters import FL_dir

from Core.MISC import GRID_convention_g, GRID_convention_L, GRID_convention_m, GRID_convention_N
from Core.Laplace import Laplace_Transform_1D, Laplace_Transform_ND

class analysis_3D_one_config(object):
    def __init__(self, L, N, g, m, components1, components2, config, base_dir=FL_dir):
        self.N = N
        self.L = L
        self.g = g
        self.m = m
        self.components1 = components1
        self.components2 = components2
        self.config = config
        self.directory = f"{base_dir}/{GRID_convention_g(g)}/{GRID_convention_N(N)}/{GRID_convention_L(L)}/{GRID_convention_m(m)}/FL/"

        self.load_in_data()
        self.get_x_correlator()
        self.get_LT()

    def load_in_data(self):
        import h5py

        result_file = f'cosmhol-su{self.N}_L{self.L}_g{self.g}_m2{self.m}-FL.{self.config}.h5'

        f = h5py.File(f'{self.directory}{result_file}')
        for i in range(len(f['FL'].keys())):
            x_ = re.findall(r'_\d+', f['FL'][f'FL_{i}'].attrs['source'][0].astype(str))
            y_ = re.findall(r'_\d+', f['FL'][f'FL_{i}'].attrs['sink'][0].astype(str))

            x = tuple([int(i[1:]) for i in x_])
            y = tuple([int(i[1:]) for i in y_])

            if x == self.components1 and y == self.components2:
                self.result_index = i
                break

        if self.result_index is None:
            raise FileNotFoundError("h5 data file doesn't contain appropriate components")

        data = f['FL'][f'FL_{self.result_index}']['P_FULL_3D'][()]

        self.correlator_p = numpy.array(data)['re'] + 1j * numpy.array(data)['im']

    def process_p_correlator(self):
        self.correlator_p[:, 0, 0, 0] = 0

    def get_x_correlator(self):
        self.correlator_x = ifftn(self.correlator_p)

    def get_LT(self):
        self.Laplace_p = Laplace_Transform_ND(self.correlator_x, 3, (1, 1, 1))


a = analysis_3D_one_config(256, 2, 0.2, -0.062, (1, 1), (0, 0), 100)
