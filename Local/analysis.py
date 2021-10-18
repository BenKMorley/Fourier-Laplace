import numpy
import matplotlib.pyplot as plt
from numpy.core.numeric import tensordot, zeros_like
from numpy.fft import fftn, ifftn
from numpy.lib.function_base import gradient
from scipy.optimize import minimize, least_squares
from tqdm import tqdm
from multiprocessing import Pool
import pickle
from copy import copy
from functools import reduce