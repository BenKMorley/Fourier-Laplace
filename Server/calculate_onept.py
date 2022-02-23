import os
import sys
import pdb
import pickle
import argparse
import h5py
import numpy
from tqdm import tqdm

sys.path.append(os.getcwd())
from Core.parameters import FL_dir as base_dir
from Core.MISC import GRID_convention_g, GRID_convention_L, GRID_convention_m, GRID_convention_N


def get_onept(config, base_dir, g, N, L, m, components):
    # Now load in the onept function
    onept_dir = f"{base_dir}/{GRID_convention_g(g)}/{GRID_convention_N(N)}/" +\
                f"{GRID_convention_L(L)}/{GRID_convention_m(m)}/onept/"

    try:
        f = h5py.File(f'{onept_dir}cosmhol-su{N}_L{L}_g{g}_m2{m}-emtc.{config}.h5')
        onept = f['emt']['value'][()][components]
        assert onept[1] == 0, "The onept function has no imaginary component"

    except Exception:
        return None

    return onept[0]


parser = argparse.ArgumentParser()

parser.add_argument('N', metavar="N", type=int)
parser.add_argument('g', metavar="ag", type=float)
parser.add_argument('m', metavar="m2", type=float)
parser.add_argument('L', metavar="L / a", type=int)
parser.add_argument('T_1', metavar="first comp of T", type=int)
parser.add_argument('T_2', metavar="second comp of T", type=int)

args = parser.parse_args()

configs = pickle.load(open(f'Server/data/configs_N{args.N}_g{args.g}_L{args.L}_m{args.m}.pcl', 'rb'))

# Make T
T = (args.T_1, args.T_2)

onepts = []

for config in tqdm(configs):
    onept = get_onept(config, base_dir, args.g, args.N, args.L, args.m, T)
    onepts.append(onept)

onepts = numpy.array(onepts, dtype=numpy.float64)

numpy.save(f'Server/data/onept/onept_full_N{args.N}_g{args.g}_L{args.L}_m{args.m}_T{T[0]}{T[1]}.npy', onepts)

onept = numpy.mean(onepts)

# Save the result
pickle.dump(onept, open(f'Server/data/onept/onept_N{args.N}_g{args.g}_L{args.L}_m{args.m}_T{T[0]}{T[1]}.pcl', 'wb'))
