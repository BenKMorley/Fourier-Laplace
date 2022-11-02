import sys
import os
import argparse
from ast import literal_eval

sys.path.append(os.getcwd())

from Server.three_d_analysis import full_analysis_3D

# Use multiprocessing module
parallel = True

parser = argparse.ArgumentParser()

parser.add_argument('N', metavar="N", type=int)
parser.add_argument('g', metavar="ag", type=float)
parser.add_argument('m', metavar="m2", type=float)
parser.add_argument('L', metavar="L / a", type=int)
parser.add_argument('T1', metavar="T1", type=str)
parser.add_argument('T2', metavar="T2", type=str)
parser.add_argument('x_max', metavar="x_max", type=float)

# The number of dimensions of the correlator to keep
parser.add_argument('dims', metavar="dims", type=int)

args = parser.parse_args()

# Remove quotes
T1 = args.T1
T2 = args.T2

count = 0
while (type(T1) != tuple) and (count < 100):
    T1 = literal_eval(T1)
    count += 1

count = 0
while (type(T2) != tuple) and (count < 100):
    T2 = literal_eval(T2)
    count += 1

a = full_analysis_3D(args.L, args.N, args.g, args.m, args.T1, args.T2, args.x_max, args.dims)
