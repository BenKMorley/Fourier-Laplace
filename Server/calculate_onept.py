import os
import sys
import pdb
import pickle
import argparse
import h5py
import numpy
from tqdm import tqdm
from ast import literal_eval

sys.path.append(os.getcwd())
from Server.parameters import FL_dir as base_dir
from Core.MISC import GRID_convention_g, GRID_convention_L, GRID_convention_m, GRID_convention_N


def get_onept(config, base_dir, g, N, L, m, components):
    # Now load in the onept function
    onept_dir = f"{base_dir}/{GRID_convention_g(g)}/{GRID_convention_N(N)}/" +\
                f"{GRID_convention_L(L)}/{GRID_convention_m(m)}/onept/"
 
    f = h5py.File(f'{onept_dir}cosmhol-su{N}_L{L}_g{g}_m2{m}-emtc.{config}.h5')
    onept = f['emt']['value'][()][components]

    return onept


parser = argparse.ArgumentParser()

parser.add_argument('N', metavar="N", type=int)
parser.add_argument('g', metavar="ag", type=float)
parser.add_argument('m', metavar="m2", type=float)
parser.add_argument('L', metavar="L / a", type=int)
parser.add_argument('T', metavar="components_of_T", type=str)

args = parser.parse_args()

args.T = literal_eval(args.T)

configs = pickle.load(open(f'Server/data/configs_N{args.N}_g{args.g}_L{args.L}_m{args.m}.pcl', 'rb'))

onepts = []

for config in tqdm(configs):
    onepts.append(get_onept(config, base_dir, args.g, args.N, args.L, args.m, args.T))


onepts = numpy.array(onepts)['re'] + 1j * numpy.array(onepts)['im']

numpy.save(f'Server/data/onept/onept_full_N{args.N}_g{args.g}_L{args.L}_m{args.m}_{args.T}.npy', onepts)

onept = numpy.mean(numpy.array(onepts))

# Save the result
pickle.dump(onept, open(f'Server/data/onept_N{args.N}_g{args.g}_L{args.L}_m{args.m}_{args.T}.pcl', 'wb'))
