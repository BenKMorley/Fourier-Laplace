import numpy
import argparse
import os
import re
import sys
import pickle
import pdb

sys.path.append(os.getcwd())

from Core.MISC import GRID_convention_g, GRID_convention_L, GRID_convention_m, GRID_convention_N
from Core.parameters import FL_dir as base_dir

parser = argparse.ArgumentParser()

parser.add_argument('N', metavar="N", type=int)
parser.add_argument('g', metavar="ag", type=float)
parser.add_argument('m', metavar="m2", type=float)
parser.add_argument('L', metavar="L / a", type=int)
parser.add_argument('size', metavar="group size", type=int)

parser.add_argument('-ID', metavar="ID", type=int, default=argparse.SUPPRESS)

kwargs = vars(parser.parse_args())
args = parser.parse_args()

return_only_ID = False
if 'ID' in kwargs:
    return_only_ID=True

base_dir = f"/rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor/{GRID_convention_g(args.g)}/{GRID_convention_N(args.N)}/" + \
            f"{GRID_convention_L(args.L)}/{GRID_convention_m(args.m)}/config/"

files = os.popen(f'ls {base_dir}')
configs = []

for name in files:
    if len(re.findall(r'lat.\d+', name)) != 0:
        config = int(re.findall(r'lat.\d+', name)[0][4:])
        configs.append(config)

# Put the configs into order
configs = list(numpy.sort(numpy.array(configs)))

# Save the configs
pickle.dump(configs, open(f'Server/data/configs_N{args.N}_g{args.g}_L{args.L}_m{args.m}.pcl', 'wb'))


# Now print the configs to output for use by bash
group_size = args.size


if not return_only_ID:
    i = 0
    open = False

    for config in configs[:-1]:
        if i == 0:
            print('[', end='')
            open = True

        if i < group_size - 1:
            print(config, end=',')
            i += 1

        else:
            print(config, end='] ')
            i = 0

    if not open:
        print('[', end='')
        print(configs[-1], end=']')

    else:
        print(configs[-1], end=']')

else:
    i = group_size * kwargs['ID']

    for config in configs[group_size * kwargs['ID']: ]:
        if i - group_size * kwargs['ID'] < group_size - 1:
            print(config, end=' ')
            i += 1

        else:
            print(config, end='')
            exit()
