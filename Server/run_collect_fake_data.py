import numpy
from tqdm import tqdm
# Parameters

L = 256
x_max = 1
dims = 1
num_configs = 100
num_runs = 1000

Noise_FT = numpy.zeros((num_configs * num_runs, L))
Noise_LT = numpy.zeros((num_configs * num_runs, L))

keep = numpy.ones(num_configs * num_runs)

for i in tqdm(range(num_runs)):
    try:
        data_FT = numpy.load(f'Server/data/fake_data/data{i}_L{L}_xmax{x_max}_dims{dims}_configs{num_configs}_FT.npy')
        data_LT = numpy.load(f'Server/data/fake_data/data{i}_L{L}_xmax{x_max}_dims{dims}_configs{num_configs}_FT.npy')

        Noise_FT[num_configs * i: num_configs * (i + 1)] = data_FT
        Noise_LT[num_configs * i: num_configs * (i + 1)] = data_LT

    except Exception:
        keep[num_configs * i: num_configs * (i + 1)] = 0
        print('No data found')

# Remove zeros
keep = keep.astype(bool)
Noise_FT = Noise_FT[keep]
Noise_LT = Noise_LT[keep]


numpy.save(f'Server/data/fake_data/data_full_L{L}_xmax{x_max}_dims{dims}_FT.npy', Noise_FT)
numpy.save(f'Server/data/fake_data/data_full_L{L}_xmax{x_max}_dims{dims}_LT.npy', Noise_LT)
