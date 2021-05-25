import matplotlib.pyplot as plt
import numpy

L = 20

def sum(n_x, L=L):
    q_s = numpy.arange(-L / 2, L / 2)
    return numpy.sum(numpy.exp(-1j * 2 * numpy.pi * n_x * q_s/ L))


n_x = numpy.linspace(-L, L, 1000)
results = []

for x in n_x:
    results.append(sum(x))

results = numpy.array(results)

plt.plot(n_x, numpy.real(results), label='real')
plt.plot(n_x, numpy.imag(results), label='imag')

integers = numpy.arange(-L + 1, L)

results2 = []
for x in integers:
    results2.append(sum(x))

results2 = numpy.array(results2)

plt.scatter(integers, numpy.real(results2), label='real')
plt.scatter(integers, numpy.imag(results2), label='imag')

plt.xlabel('n_x')
plt.ylabel('Inner Sum')
plt.legend()
plt.show()
