import numpy
import matplotlib.pyplot as plt
import pdb
import scipy


def Forward_Fourier_0(q, T, L, a):
    x_s = numpy.arange(-L / 2, L / 2, step=a)

    return (a / L) * numpy.sum(numpy.exp(-1j * q * x_s) * T(x_s))


def Inverse_Fourier_0(x, T, L, a):
    q_s = numpy.arange(-numpy.pi / a, numpy.pi / a, numpy.pi * 2 / L)

    return numpy.sum(numpy.exp(+1j * q_s * x) * T(x_s))


def Forward_Laplace(q, T, L, a):
    x_s = numpy.arange(0, L, step=a)

    return (a / (1 - numpy.exp(-qL))) * numpy.sum(numpy.exp(-1j * q * x_s) * T(x_s))


# Use dictionaries in order to define a discrete function
def discretise_function(T, L, a, var_range):
    func_dict = {}

    for x in var_range:
        func_dict[x] = T(x, L, a)

    # To anticipate what we will do later, we will also include complex values
    for x in 1j * var_range:
        func_dict[x] = T(x, L, a)

    def func(x):
        if x in func_dict:
            return func_dict[x]

        else:
            # return numpy.nan
            return T(x, L, a)

    func = numpy.vectorize(func)

    print("=============================================================")
    print(func_dict)
    print("")

    return func, func_dict


# Now let's adapt the above Transforms to work with these discrete functions
def Forward_Fourier_0_discrete(T, L, a):
    x_range = numpy.arange(-L / 2, L / 2, a).astype(complex)
    # x_range = numpy.arange(-L / 2 + a, L / 2, a).astype(complex)

    def mom_space_discrete(q, L, a):
        return (a / L) * numpy.sum(numpy.exp(-1j * q * x_range) * T(x_range))

    q_range = numpy.arange(-numpy.pi/a, numpy.pi/a, 2 * numpy.pi / L).astype(complex)

    mom_space_discrete, func_dict = discretise_function(mom_space_discrete, L, a, q_range)

    return mom_space_discrete, q_range, func_dict


def Forward_Laplace_discrete(T, L, a):
    x_range = numpy.arange(0, L, a).astype(complex)

    def mom_space_discrete(q, L, a):
        return (a / (1 - numpy.exp(-qL))) * numpy.sum(numpy.exp(-1j * q * x_range) * T(x_range))

    q_range = numpy.arange(-numpy.pi/a, numpy.pi/a, 2 * numpy.pi / L).astype(complex)
    # q_range = numpy.arange(-numpy.pi/a + numpy.pi * 2/L, numpy.pi/a, 2 * numpy.pi / L).astype(complex)

    mom_space_discrete, func_dict = discretise_function(mom_space_discrete, L, a, q_range)

    return mom_space_discrete, q_range, func_dict


def Inverse_Fourier_0_discrete(T, L, a):
    q_range = numpy.arange(-numpy.pi/a, numpy.pi/a, 2 * numpy.pi / L).astype(complex)

    def pos_space_discrete(x, L, a):
        return numpy.sum(numpy.exp(+1j * q_range * x) * T(q_range))

    x_range = numpy.arange(-L / 2, L / 2, a).astype(complex)
    # x_range = numpy.arange(-L / 2 + a, L / 2, a).astype(complex)

    pos_space_discrete, func_dict = discretise_function(pos_space_discrete, L, a, x_range)

    return pos_space_discrete, x_range, func_dict


def plot_complex(var_range, func, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if numpy.sum(numpy.abs(numpy.real(var_range))) > numpy.sum(numpy.abs(numpy.imag(var_range))):
        ax.scatter(numpy.real(var_range), numpy.imag(func(var_range)), label="imag")
        ax.scatter(numpy.real(var_range), numpy.real(func(var_range)), label="real")
        ax.set_xlabel("Real part")

    else:
        ax.scatter(numpy.imag(var_range), numpy.imag(func(var_range)), label="imag")
        ax.scatter(numpy.imag(var_range), numpy.real(func(var_range)), label="real")
        ax.set_xlabel("Imag part")

    plt.legend()

    return ax


# Let's have a look at the behavior of a T(q) prop to q^2 term:
L = 16
a = 1

q_range = numpy.arange(-numpy.pi/a, numpy.pi/a, 2 * numpy.pi / L).astype(complex)


gaussian_F_q, gaussian_F_q_dict = discretise_function(lambda q, L, a: numpy.exp(- q ** 2), L, a, q_range)
plot_complex(q_range, gaussian_F_q)
plt.savefig(f'graphs/Original_q.png', dpi=500)
plt.show()

plot_complex(q_range * 1j, gaussian_F_q)
plt.savefig(f'graphs/Original_q_imag.png', dpi=500)
plt.show()

gaussian_x, x_range, gaussian_x_dict = Inverse_Fourier_0_discrete(gaussian_F_q, L, a)
plot_complex(x_range, gaussian_x)
plt.savefig(f'graphs/x_space_real.png', dpi=500)
plt.show()

gaussian_F_q_back, q_range, gaussian_F_q_back_dict = Forward_Fourier_0_discrete(gaussian_x, L, a)
plot_complex(q_range, gaussian_F_q_back)
plt.savefig(f'graphs/Recovered_real.png', dpi=500)
plt.show()

plot_complex(q_range * 1j, gaussian_F_q_back)
plt.savefig(f'graphs/Recovered_imag.png', dpi=500)
plt.show()


Matrix = numpy.zeros((2 * L, L), dtype=complex)
Tilde_T = numpy.zeros(2 * L, dtype=complex)
T = numpy.zeros(L, dtype=complex)

for i, q in enumerate(q_range):
    Tilde_T[i] = q ** 2
    Tilde_T[i + L] = (1j * q) ** 2

    for j, x in enumerate(x_range):
        Matrix[i, j] = numpy.exp(-1j * q * x)
        Matrix[i + L, j] = numpy.exp(- q * x)

Product = numpy.dot(Matrix.T, Matrix)
Inverse = numpy.dot(numpy.linalg.inv(Product), Matrix.T)
T_new = numpy.dot(Inverse, Tilde_T)


def quadratic_position(x, L, a):
    i = numpy.argwhere(x == x_range)

    return T_new[i]


quadratic_position, quadratic_position_dict = discretise_function(quadratic_position, L, a, x_range)
plot_complex(x_range, quadratic_position)

quadratic_Fourier_back, q_range, quadratic_Fourier_back_dict = Forward_Fourier_0_discrete(quadratic_position, L, a)
plot_complex(q_range, quadratic_Fourier_back)
plt.show()


plot_complex(q_range * 1j, quadratic_Fourier_back)
plt.savefig(f'graphs/Matrix_recovery.png', dpi=500)
plt.show()


# quadratic, quadratic_dict = discretise_function(lambda q, L, a: q ** 2, L, a, q_range)

# quadratic_position, x_range, quadratic_position_dict = Inverse_Fourier_0_discrete(quadratic, L, a)
# # plot_complex(x_range, quadratic_position)

# quadratic_Fourier_back, q_range, quadratic_Fourier_back_dict = Forward_Fourier_0_discrete(quadratic_position, L, a)
# # plot_complex(q_range, quadratic_Fourier_back)
# # plot_complex(q_range[L//2 - L//16: L//2 + L//16 + 1], quadratic_Fourier_back)
# # ax = plot_complex(q_range[L//2: ] * 0.1j, quadratic_Fourier_back)
# # plot_complex(q_range[L//2: ] * 0.1, quadratic_Fourier_back, ax)
# plot_complex(q_range[L//2: ] * 0.1, lambda q: quadratic_Fourier_back(1j * q) + quadratic_Fourier_back(q) * 6)

# # Now we could easily get our cancellation using the Fourier Transform with -iq,
# # But we have set things up in such a way that we need to use the Laplace Transforms
# # def shifted_quadratic_position(x):
# #     return numpy.where(numpy.real(x) < 0, quadratic_position(x + L / 2), quadratic_position(x - L / 2))

# # plot_complex(x_range, shifted_quadratic_position)
# # quadratic_Fourier_back_shifted, q_range = Forward_Fourier_0_discrete(shifted_quadratic_position, L, a)

# # plot_complex(q_range, quadratic_Fourier_back_shifted)
# # plot_complex(q_range * -1j, quadratic_Fourier_back_shifted)
