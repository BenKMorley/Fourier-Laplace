import sys
import os
import argparse
import numpy
from multiprocessing import Pool
from ast import literal_eval

sys.path.append(os.getcwd())

from Core.three_d_analysis import analysis_3D_one_config

# Use multiprocessing module
parallel = True

parser = argparse.ArgumentParser()

parser.add_argument('N', metavar="N", type=int)
parser.add_argument('g', metavar="ag", type=float)
parser.add_argument('m', metavar="m2", type=float)
parser.add_argument('L', metavar="L / a", type=int)
parser.add_argument('T1_1', metavar="T1_1", type=int)
parser.add_argument('T1_2', metavar="T1_2", type=int)
parser.add_argument('T2_1', metavar="T2_1", type=int)
parser.add_argument('T2_2', metavar="T2_2", type=int)
parser.add_argument('configs', metavar="configs", type=str)

parser.add_argument('-x_max', metavar="x_max", type=float, default=argparse.SUPPRESS)

# The number of dimensions of the correlator to keep
parser.add_argument('-dims', metavar="dims", type=int, default=argparse.SUPPRESS)

args = parser.parse_args()

# Extract any optional arguments
kwargs = vars(parser.parse_args())
del kwargs['N']
del kwargs['g']
del kwargs['m']
del kwargs['L']
del kwargs['T1_1']
del kwargs['T1_2']
del kwargs['T2_1']
del kwargs['T2_2']
del kwargs['configs']

if 'x_max' in kwargs:
    x_max = kwargs['x_max']

else:
    x_max = numpy.inf


# Remove quotes
configs = args.configs

count = 0
while (type(configs) != list) and (count < 100):
    configs = literal_eval(configs)
    count += 1

# Make T's
T1 = (args.T1_1, args.T1_2)
T2 = (args.T2_1, args.T2_2)

def run(config):
    print(f'Running for config {config}')
    analysis = analysis_3D_one_config(args.L, args.N, args.g, args.m, T1, T2, config, x_max=x_max)


    # Save the momentum space
    numpy.save(f'Server/data/Correlator_p_N{args.N}_g{args.g}_L{args.L}_m{args.m}_T{T1[0]}{T1[1]}_T{T2[0]}{T2[1]}' + \
            f'_config{config}_dims{args.dims}.npy', analysis.correlator_p[(0, ) * (3 - args.dims)])

    # Save the position space
    analysis.process_p_correlator()
    analysis.get_x_correlator()
    numpy.save(f'Server/data/Correlator_x_N{args.N}_g{args.g}_L{args.L}_m{args.m}_T{T1[0]}{T1[1]}_T{T2[0]}{T2[1]}' + \
            f'_config{config}_dims{args.dims}.npy', analysis.correlator_x[(0, ) * (3 - args.dims)])

    # Save the onept function for the sake of cross-checking
    numpy.save(f'Server/data/Onept_N{args.N}_g{args.g}_L{args.L}_m{args.m}_T{T1[0]}{T1[1]}_T{T2[0]}{T2[1]}' + \
               f'_config{config}_dims{args.dims}.npy', analysis.onept)

    # Only keep one dimension
    analysis.get_LT()
    numpy.save(f'Server/data/Laplace_N{args.N}_g{args.g}_L{args.L}_m{args.m}_T{T1[0]}{T1[1]}_T{T2[0]}{T2[1]}' + \
        f'_config{config}_dims{args.dims}_xmax{x_max}.npy', analysis.Laplace_p[(0, ) * (3 - args.dims)])

    print('Finished - next config please!')

# Use all the CPU's to run this
if parallel:
    p = Pool(min(28, len(configs)))
    p.map(run, configs)
    p.close()

else:
    for config in configs:
        run(config)

