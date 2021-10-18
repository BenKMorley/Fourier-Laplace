# Naming conventions
def GRID_convention_m(m):
    return f"m2{m}".rstrip('0')


def MCMC_convention_m(m):
    return f"msq={-m:.8f}"


def GRID_convention_N(N):
    return f"su{N}"


def MCMC_convention_N(N):
    return f"N={N}"


def GRID_convention_L(L):
    return f"L{L}"


def MCMC_convention_L(L):
    return f"L={L}"


def GRID_convention_g(g):
    return f"g{g}".rstrip('0').rstrip('.')


def MCMC_convention_g(g):
    return f"g={g:.2f}"
