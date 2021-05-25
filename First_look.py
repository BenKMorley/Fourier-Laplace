import numpy
import matplotlib.pyplot as plt
import h5py
import re
from tqdm import tqdm
import pdb

L = 256

f = h5py.File("data/twopt_emtc_2_2_emtc_2_2_0_0.h5")

data = f['twopt_emtc_2_2_emtc_2_2_0_0.h5']
samples = data.attrs['nSample'][0]

data_central = numpy.zeros(L, dtype=numpy.complex128)
data_boot = numpy.zeros((samples, L), dtype=numpy.complex128)

data_central[:L // 2] = data[f'data_C'][:L // 2, 0]
data_central[L // 2:] = data[f'data_C'][:L // 2, 1]

for i in range(samples):
    data_boot[i, :L // 2] = data[f'data_S_{i}'][:L // 2, 0]
    data_boot[i, L // 2:] = data[f'data_S_{i}'][:L // 2, 1]

# Remove the one-point function on each sample - do twice because of how small
# the signal is!
data_central = data_central - numpy.mean(data_central.real)
data_central = data_central - numpy.mean(data_central.real)

data_boot = data_boot - numpy.mean(data_boot.real, axis=1).reshape((samples, 1)).repeat(256, axis=1)
data_boot = data_boot - numpy.mean(data_boot.real, axis=1).reshape((samples, 1)).repeat(256, axis=1)

# Let's find the Fourier Transform
data_central_p = numpy.zeros(L, dtype=numpy.complex128)
data_boot_p = numpy.zeros((samples, L), dtype=numpy.complex128)

for i in range(L):
    data_central_p[i] = numpy.mean(data_central * numpy.exp(-1j * 2 * numpy.pi * i / L))

for j in tqdm(range(samples)):
    for i in range(L):
        data_boot_p[j, i] = numpy.mean(data_boot[j] * numpy.exp(-1j * 2 * numpy.pi * i / L))

std = numpy.std(data_boot_p.real, axis=0)

plt.fill_between([2 * numpy.pi * i / L for i in range(L)], data_central_p - std, data_central_p + std, alpha=0.1)
plt.plot([2 * numpy.pi * i / L for i in range(L)], data_central_p)
plt.savefig("graphs/Fourier.png", dpi=500)
plt.show()

# for i in range(100):
#     plt.plot(data_boot_p[i].real)

# Find the Laplace Transform
data_central_p_L = numpy.zeros(L, dtype=numpy.complex128)
data_boot_p_L = numpy.zeros((samples, L), dtype=numpy.complex128)

for i in range(L):
    data_central_p_L[i] = numpy.mean(data_central * numpy.exp(-2 * numpy.pi * i / L))

for j in tqdm(range(samples)):
    for i in range(L):
        data_boot_p_L[j, i] = numpy.mean(data_boot[j] * numpy.exp(-2 * numpy.pi * i / L))

std = numpy.std(data_boot_p_L.real, axis=0)

plt.fill_between([2 * numpy.pi * i / L for i in range(L)], data_central_p_L - std, data_central_p_L + std, alpha=0.1)
plt.plot([2 * numpy.pi * i / L for i in range(L)], data_central_p_L)
plt.savefig("graphs/Laplace.png", dpi=500)
plt.show()


## Attempt 2: Remove the x = L / 2 spike by just ignoring that data
# for i in range(100):
#     plt.plot(data_boot_p[i].real)
