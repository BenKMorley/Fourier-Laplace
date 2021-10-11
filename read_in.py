from parse import parse
import os
import argparse
import h5py
import numpy


# PARAMETERS OF REQUIRED DATA
L=128
M2=-0.157
G=0.5
GROUP=su2
CONFIG_SPACING=100
OP_TYPE=trphi
OP=tr\(phi^2\)

# onept directories and file templates and onept operator
IN_DIR=../g${G}/${GROUP}/L${L}/
FILE_TEMPLATE=cosmhol-${GROUP}_L${L}_g${G}_m2${M2}-${OP_TYPE}.{config}.h5

# Template file names for numpy files
OUT_FILE=cosmhol-${GROUP}_L${L}_g${G}_m2${M2}-onept_${OP_TYPE}_${OP}.npy

# DIR the numpy files will be saved to. I've used the same convention as the
# saved files on DiRAC
OUT_DIR=../npy_files/g${G}/${GROUP}/L${L}/m2${M2}/onept/

## Read in the command line options
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('-file_template', type=str)
parser.add_argument('-path_template', type=str)
parser.add_argument('-onept_path', type=str) # onept data in a different directory to twopt
parser.add_argument('-onept_file', type=str) # onept data in a different directory to twopt
parser.add_argument('-onept_op', type=str) # onept operator, e.g. "tr(phi^2)"
parser.add_argument('-L', type=int)
parser.add_argument('-config_spacing', type=int)
parser.add_argument('-output_path', type=str)
args = parser.parse_args()

# Set parameters
L = 256
config_spacing = 100
N = 2
file_template = "cosmhol-${GROUP}_L${L}_g${G}_m2${M2}-${OP_TYPE}.{config}.h5"
path = args.path_template
out_path = args.output_path
g=0.1

# Initialise the minimum and maximum configs
min_config = 10**100
max_config = 0

config_files = os.popen(f'ls {path}')

with config_files as f:
  for line in f:
    config = parse(file_template, line)
    if config != None:
      config = int(config['config'])
      if(config < min_config):
        min_config = config
      if(config > max_config):
        max_config = config

print(f"Min config: {min_config}")
print(f"Max config: {max_config}")

## Read in the two-point functions
source = 'emtc_2_2'
sink = 'emtc_2_2'
mom_critera = [(0, 0)]

# Rescale min_config and max_config by the spacing for easy iteration
min_config = min_config // config_spacing
max_config = max_config // config_spacing

min_config = 200
max_config = 240

def get_keys(mom_criteria="all"):
  """
    This function will get all the key values e.g. indicies in the configuration
    where the momenta, sink and source match a certain condition. This is designed
    for 2-point correlators currently.
    
    INPUTS :
    --------
    mom_criteria : This is eiter an array of tuples of appropriate dimension, which
      the method will try and match, e.g. use [(0, 0)] for the zero mode OR the
      keyword "all" can be used in which case all momenta will be searched for.

    RETURNS :
    ---------
    key_dict : Dictionary containing the index of the desired data labelled by
      the momentum.
    array_dict : Dictionary containing zeroed numpy arrays for storage
      of twopt configuration data.
  """
  array_dict = {}
  key_dict = {}

  print(f"{path}{file_template.format(config=min_config * config_spacing)}")

  with h5py.File(f"{path}{file_template.format(config=min_config * config_spacing)}") as f:
    for j in range(len(f['twopt'])):
      # For each saved time-series extract its attributes
      attributes = f['twopt'][f'twopt_{j}'].attrs

      # Check if the sources and sinks are as desired
      if(attributes['sink'] == sink.encode('ascii') and attributes['source'] == source.encode('ascii')):

        # Calculate the tuple of momentum
        momentum = (attributes['mom'][0], attributes['mom'][1])

        # If the momentum matches the criteria record it
        if (mom_criteria == "all" or momentum in mom_criteria):
          array_dict[momentum] = numpy.zeros((max_config - min_config + 1, L, 2))
          key_dict[momentum] = j
    
    return array_dict, key_dict

array_dict, key_dict = get_keys(mom_criteria=mom_critera)

for config_no in range(min_config, max_config + 1):

  print(f"Taking a look at config number {config_no * config_spacing} now")

  with h5py.File(f"{path}{file_template.format(config=config_no * config_spacing)}") as f:

    for key in key_dict:
      
      # Select the (256, 2) time-series data
      data = f['twopt'][f'twopt_{key_dict[key]}']['data']

      # Change the form of the data for storage as a numpy array
      data_numpy = numpy.array([[dataset[0], dataset[1]] for dataset in data])

      # Add the data to the relevent array
      array_dict[key][config_no - min_config] = data_numpy

# Remove the .{config}. part from the file_template.
## WARNING!!! If the files don't end with .{config}.h5 this method won't work
file_template_short = file_template.format(config="")[:-4]

# Save all of the data
for entry in array_dict:
  numpy.save(f"{out_path}{file_template_short}_{source}_{sink}_{entry[0]}_{entry[1]}.npy", array_dict[entry])

## Read in the onept data
file_template = args.onept_file
path_template = args.onept_path
file_tag = args.file_tag

path = path_template.format(group=group, L=L, g=g, m2=m2)

output = numpy.zeros(max_config - min_config + 1)

for config_no in range(min_config, max_config + 1):

  with h5py.File(f"{path}{file_template.format(config=config_no * config_spacing)}") as f:

    data = f[file_tag]
    quantities = list(data)
           
    for i in range(len(data)):
      piece = data[quantities[i]].attrs

      # Extract the onept operator of interest
      if(piece["op"] == op):
        output[config_no - min_config] = piece["value"][0]

numpy.save(f"{out_path}{file_template[:-2]}_{onept_op}.npy", output)
