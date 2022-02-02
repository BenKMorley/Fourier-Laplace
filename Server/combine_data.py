import os
import re
import numpy
from tqdm import tqdm
import pdb
import h5py
import sys


# Import from the Core directory
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Core')

from Server.parameters import FL_dir
from Server.three_d_analysis import get_result_index

from Core.MISC import GRID_convention_g, GRID_convention_L, GRID_convention_m, GRID_convention_N


def combine(N, g, L, m, T0, T1, x_max, dims):
    base_dir = f"Server/data/"

    files = os.popen(f'ls {base_dir}')
    configs = []

    for name in files:
        if len(re.findall(rf'Laplace_N{N}_g{g}_L{L}_m{m}_\({T0[0]}, {T0[1]}\)_\({T1[0]}, {T1[1]}\)_config\d+_dims{dims}_xmax{x_max:.1f}.npy',
                          name)) != 0:
            config = int(re.findall(r'config\d+', name)[0][6:])
            configs.append(config)

    full_data_Laplace = numpy.zeros((len(configs), ) + (L, ) * dims)
    full_data_Fourier = numpy.zeros((len(configs), ) + (L, ) * dims)
    full_data_correlator_x = numpy.zeros((len(configs), ) + (L, ) * dims)

    for i, config in tqdm(enumerate(configs)):
        data = numpy.load(f'Server/data/Laplace_N{N}_g{g}_L{L}_m{m}_{T0}_{T1}_config{config}_dims{dims}_xmax{x_max:.1f}.npy')
        full_data_Laplace[i] = data

        try:
            data = numpy.load(f'Server/data/Correlator_x_N{N}_g{g}_L{L}_m{m}_{T0}_{T1}_config{config}_dims{dims}.npy')
            full_data_correlator_x[i] = data

        except Exception:
            print(f"No correlator data found config = {config}")

        try:
            data = numpy.load(f'Server/data/Correlator_p_N{N}_g{g}_L{L}_m{m}_{T0}_{T1}_config{config}_dims{dims}.npy')
            full_data_Fourier[i] = data

        except Exception:
            print(f"No correlator data found config = {config}")
    
    numpy.save(f'Server/data/full_data/Laplace_N{N}_g{g}_L{L}_m{m}_{T0}_{T1}_xmax{x_max:.1f}_dims{dims}.npy', full_data_Laplace)
    numpy.save(f'Server/data/full_data/Fourier_N{N}_g{g}_L{L}_m{m}_{T0}_{T1}_dims{dims}.npy', full_data_Fourier)
    numpy.save(f'Server/data/full_data/Correlator_x_N{N}_g{g}_L{L}_m{m}_{T0}_{T1}_dims{dims}.npy', full_data_correlator_x)


if __name__ == "__main__":
    import argparse
    from ast import literal_eval

    parser = argparse.ArgumentParser()

    parser.add_argument('N', metavar="N", type=int)
    parser.add_argument('g', metavar="ag", type=float)
    parser.add_argument('m', metavar="m2", type=float)
    parser.add_argument('L', metavar="L / a", type=int)
    parser.add_argument('T1', metavar="T1", type=str)
    parser.add_argument('T2', metavar="T2", type=str)
    parser.add_argument('x_max', metavar="x_max", type=float)
    parser.add_argument('dims', metavar="dims", type=int)

    args = parser.parse_args()

    count = 0
    while (type(args.T1) != tuple) and (count < 100):
        args.T1 = literal_eval(args.T1)
        count += 1

    count = 0
    while (type(args.T2) != tuple) and (count < 100):
        args.T2 = literal_eval(args.T2)
        count += 1

    combine(args.N, args.g, args.L, args.m, args.T1, args.T2, args.x_max, args.dims)
