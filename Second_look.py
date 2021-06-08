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

    # Remove the zero-mode
    result[i] = result[i] - numpy.mean(result[i][L // 4: 3 * L // 4])

    for j in range(L):
        FT[i, j] = Fourier_Transform(result[i], p_s[j])
        LT[i, j] = Laplace_Transform(result[i], p_s[j])


mean = numpy.mean(result, axis=0)
std = numpy.std(result, axis=0)

mean_FT = numpy.mean(FT, axis=0)
std_FT = numpy.std(FT, axis=0)

mean_LT = numpy.mean(LT, axis=0)
std_LT = numpy.std(LT, axis=0)

prefactor = 1 / (1 - numpy.exp(-p_s * L))
mean_sum = numpy.mean(LT / prefactor + FT, axis=0)
std_sum = numpy.std(LT / prefactor + FT, axis=0)


plt.errorbar(range(L), mean, std, ls='')
plt.title("Real Space")
plt.ylim(-0.01, 0.01)
plt.xlabel('x')
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 8)
plt.savefig("graphs/Second_look_twopt.png", dpi=500)
plt.show()


plt.errorbar(p_s, mean_FT, std_FT, ls='')
plt.ylim(-0.001, 0.001)
plt.xlabel('p')
plt.title("Fourier Transform")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 8)
plt.savefig("graphs/Second_look_Fourier.png", dpi=500)
plt.show()

plt.errorbar(p_s, mean_LT, std_LT, ls='')
plt.ylim(-0.001, 0.001)
plt.xlabel('p')
plt.title("Laplace Transform")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 8)
plt.savefig("graphs/Second_look_Laplace.png", dpi=500)
plt.show()

plt.errorbar(p_s, mean_sum, std_sum, ls='')
plt.ylim(-0.001, 0.001)
plt.xlabel('p')
plt.title("Sum")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15, 8)
plt.savefig("graphs/Second_look_Sum.png", dpi=500)
plt.show()