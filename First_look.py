import numpy
import matplotlib.pyplot as plt
import matplotlib
import h5py
import re
from tqdm import tqdm
import pdb
from read_in_raw import read_in_twopt, read_in_onept_emtc


# PARAMETERS
L = 256
g = 0.1
N = 2
m2 = -0.0305

twopt_data = read_in_twopt("emtc_2_2", "emtc_2_2", (0, 0), L, g, N, m2)
onept_data = read_in_onept_emtc((2, 2), L, g, N, m2) / (L ** 2)


def Fourier_Transform(data, p):
    L = data.shape[0]

    return numpy.mean(data * numpy.exp(-1j * p * numpy.arange(L)))


def Laplace_Transform(data, p, offset=1):
    L = data.shape[0]

    # For example if the offset is 2, x = L - 2 -> 0 -> -2; x = L - 1 -> 1 -> -1
    # all other x are unaffected
    x_s = (numpy.arange(L) + offset) % L - offset

    prefactor = 1 / (1 - numpy.exp(-p * L))

    return prefactor * numpy.mean(data * numpy.exp(-p * x_s))


# We want to randomly shuffle the data and calculate the connected twopt
# function on each bootstrap sample.
size, L = twopt_data.shape
no_samples = 1000
result = numpy.zeros((no_samples, L))
FT = numpy.zeros((no_samples, L))
LT = numpy.zeros((no_samples, L))
p_s = numpy.arange(L) * 2 * numpy.pi / L

boot_samples = numpy.random.randint(size, size=(no_samples, size))

for i in tqdm(range(no_samples)):
    result[i] = numpy.mean(twopt_data[boot_samples[i]], axis=0) - (numpy.mean(onept_data[boot_samples[i]])) ** 2

    for j in range(L):
        FT[i, j] = Fourier_Transform(result[i], p_s[j])
        LT[i, j] = Laplace_Transform(result[i], p_s[j])


mean = numpy.mean(result, axis=0)
std = numpy.std(result, axis=0)

mean_FT = numpy.mean(FT, axis=0)
std_FT = numpy.std(FT, axis=0)

mean_LT = numpy.mean(LT, axis=0)
std_LT = numpy.std(LT, axis=0)

plt.errorbar(range(L), mean, std, ls='')
plt.title("Real Space")
plt.xlabel('x')
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 8)
plt.savefig("graphs/First_look_twopt.png", dpi=500)
plt.show()

plt.errorbar(p_s, mean_FT, std_FT, ls='')
plt.ylim(-0.001, 0.001)
plt.xlabel('p')
plt.title("Fourier Transform")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 8)
plt.savefig("graphs/First_look_Fourier.png", dpi=500)
plt.show()

plt.errorbar(p_s, mean_LT, std_LT, ls='')
plt.ylim(-0.001, 0.001)
plt.xlabel('p')
plt.title("Laplace Transform")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 8)
plt.savefig("graphs/First_look_Laplace.png", dpi=500)
plt.show()


#### Script using the summarized data files is below


# L = 256

# f = h5py.File("data/twopt_emtc_2_2_emtc_2_2_0_0.h5")

# data = f['twopt_emtc_2_2_emtc_2_2_0_0.h5']
# samples = data.attrs['nSample'][0]

# data_central = numpy.zeros(L, dtype=numpy.complex128)
# data_boot = numpy.zeros((samples, L), dtype=numpy.complex128)

# data_central[:L // 2] = data[f'data_C'][:L // 2, 0]
# data_central[L // 2:] = data[f'data_C'][:L // 2, 1]

# for i in range(samples):
#     data_boot[i, :L // 2] = data[f'data_S_{i}'][:L // 2, 0]
#     data_boot[i, L // 2:] = data[f'data_S_{i}'][:L // 2, 1]

# # Remove the one-point function on each sample - do twice because of how small
# # the signal is!
# data_central = data_central - numpy.mean(data_central.real)
# data_central = data_central - numpy.mean(data_central.real)

# data_boot = data_boot - numpy.mean(data_boot.real, axis=1).reshape((samples, 1)).repeat(256, axis=1)
# data_boot = data_boot - numpy.mean(data_boot.real, axis=1).reshape((samples, 1)).repeat(256, axis=1)

# # Let's find the Fourier Transform
# data_central_p = numpy.zeros(L, dtype=numpy.complex128)
# data_boot_p = numpy.zeros((samples, L), dtype=numpy.complex128)

# for i in range(L):
#     data_central_p[i] = numpy.mean(data_central * numpy.exp(-1j * 2 * numpy.pi * i / L))

# for j in tqdm(range(samples)):
#     for i in range(L):
#         data_boot_p[j, i] = numpy.mean(data_boot[j] * numpy.exp(-1j * 2 * numpy.pi * i / L))

# std = numpy.std(data_boot_p.real, axis=0)

# plt.fill_between([2 * numpy.pi * i / L for i in range(L)], data_central_p - std, data_central_p + std, alpha=0.1)
# plt.plot([2 * numpy.pi * i / L for i in range(L)], data_central_p)
# plt.savefig("graphs/Fourier.png", dpi=500)
# plt.show()

# # for i in range(100):
# #     plt.plot(data_boot_p[i].real)

# # Find the Laplace Transform
# data_central_p_L = numpy.zeros(L, dtype=numpy.complex128)
# data_boot_p_L = numpy.zeros((samples, L), dtype=numpy.complex128)

# for i in range(L):
#     data_central_p_L[i] = numpy.mean(data_central * numpy.exp(-2 * numpy.pi * i / L))

# for j in tqdm(range(samples)):
#     for i in range(L):
#         data_boot_p_L[j, i] = numpy.mean(data_boot[j] * numpy.exp(-2 * numpy.pi * i / L))

# std = numpy.std(data_boot_p_L.real, axis=0)

# plt.fill_between([2 * numpy.pi * i / L for i in range(L)], data_central_p_L - std, data_central_p_L + std, alpha=0.1)
# plt.plot([2 * numpy.pi * i / L for i in range(L)], data_central_p_L)
# plt.savefig("graphs/Laplace.png", dpi=500)
# plt.show()


# ## Attempt 2: Remove the x = L / 2 spike by just ignoring that data
# # for i in range(100):
# #     plt.plot(data_boot_p[i].real)
