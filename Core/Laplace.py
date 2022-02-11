import numpy


def Laplace_Transform_ND(data, dim, offset, x_max=numpy.inf):
    L = data.shape[0]
    q_fac = 2 * numpy.pi / L
    p_s = numpy.arange(L) * q_fac
    assert len(offset) == dim

    for d in range(dim):
        x_s = (numpy.arange(L) + offset[d]) % L - offset[d]
        comb = numpy.outer(p_s, x_s)
        exp_comb = numpy.exp(-comb)

        # We only need to use points at x = (-1, 0, 1). We therefore may choose to remove points
        # to reduce the noise in the Laplace Transform
        remove = x_s > x_max
        exp_comb[:, remove] = 0

        # Let D represent data and e represent the exponential comb. Use indices i, j, k to represent
        # momentum space indices and x, y, z to represent position space indices
        # D_ijk = D_zij e_k^z = (D_yzi e_j^y) e_k^z = ((D_xyz e_i^x) e_j^y) e_k^z
        data = numpy.tensordot(data, exp_comb, axes=(0, 1))

    # We don't want to introduce a constant into the method so take
    data = data - data[(0, ) * dim]

    return data


def Laplace_Transform_1D(data, dim, offset, x_max=numpy.inf):
    L = data.shape[0]
    q_fac = 2 * numpy.pi / L
    p_s = numpy.arange(L) * q_fac

    result = numpy.zeros(L)

    x_s = (numpy.arange(L) + offset) % L - offset
    remove = x_s > x_max

    for i, p in enumerate(p_s):
        exp_comb = numpy.exp(-p * x_s)
        exp_comb[remove] = 0

        result[i] = numpy.sum(exp_comb * data)

    return result
