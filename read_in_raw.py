import h5py
import numpy


def GRID_convention_m(m):
    return f"m2{m}".rstrip('0')


def GRID_convention_N(N):
    return f"su{N}"


def GRID_convention_L(L):
    return f"L{L}"


def GRID_convention_g(g):
    return f"g{g}".rstrip('0').rstrip('.')


def read_in_twopt(source, sink, L, g, N, m, step=100, start=100, end=199900):
    directory = f"data/g{g:.1f}/su{N}/L{L}/m2{m:.8f}".rstrip('0') + "/twopt/"
    filename = f"cosmhol-su{N}_L{L}_g{g:.1f}_m2{m:.8f}".rstrip('0') + "-twopt."

    size = (end - start) // step + 1
    data = numpy.zeros((size, L), dtype=numpy.complex128)

    # Get the right key - only do this once
    f = h5py.File(f"{directory}{filename}" + f"{start}.h5", "r")

    for key in f['twopt'].keys():
        source_ref = f['twopt'][key].attrs['source'][0].decode('ascii')
        sink_ref = f['twopt'][key].attrs['sink'][0].decode('ascii')

        if source == source_ref and sink == sink_ref:
            right_key = key
            break

    for i, inx in enumerate(range(start, end, step)):
        f = h5py.File(f"{directory}{filename}" + f"{start}.h5", "r")
        data[i] = numpy.array(f['twopt'][right_key]['data'])['re']

    return data


def read_in_onept_emtc(comp, L, g, N, m, step=100, start=100, end=199900):
    i0, i1 = comp  # Component of the EMT we're interested in

    directory = f"data/g{g:.1f}/su{N}/L{L}/m2{m:.8f}".rstrip('0') + "/onept/"
    filename = f"cosmhol-su{N}_L{L}_g{g:.1f}_m2{m:.8f}".rstrip('0') + f"-emtc."

    size = (end - start) // step + 1
    data = numpy.zeros(size)

    for i, inx in enumerate(range(start, end, step)):
        f = h5py.File(f"{directory}{filename}" + f"{start}.h5", "r")

        data[i] = numpy.array(f['emt']['value'])[2, 2]['re']

    return data
