'''
script to remove nan from interfile sinograms


Usage:
  remove_nans_from_projdata [--help | options]

Options:
  -s <file>, --sino=<file>                          raw data file. All the file paths needs to be relative to out_path/working_dir
  -v <value>, --value=<value>                       value to use instead of nana [default: 1] [type: float]
'''

# Author: Daniel Deidda
# Copyright 2024 National Physical Laboratory.
# First version: 5th of Aug 2024
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import pathlib
# Import the PET reconstruction engine
import sirf.STIR as pet
# Set the verbosity
pet.set_verbosity(0)
# Store temporary sinograms in RAM
pet.AcquisitionData.set_storage_scheme("memory")
# SIRF STIR message redirector
import sirf

import sirf.STIR as pet
msg = sirf.STIR.MessageRedirector(info=None, warn=None, errr=None)
# Load dataset and model
from artcertainty.utils.torch_operations import  treshold_tensor
from artcertainty.utils.system import create_working_dir_and_move_into

import os
import numpy as np

def remove_nans_and_negatives_from_projdata(data_path, value):
    sino = pet.AcquisitionData(data_path);
    sino_np = sino.as_array()
    sino_np = np.nan_to_num(sino_np, nan=value, posinf=value, neginf=value) 
    # also remove negatives
    mask = (sino_np >= 0)
    c = np.zeros(sino_np.shape)
    c[mask] = sino_np[mask]

    sino.fill(c)
    sino.write(data_path)

if __name__ == '__main__':
    from type_docopt import docopt
    args = docopt(__doc__, version='0.1.0')
    print(args)

    
    working_dir =   os.getcwd()
    prefix = working_dir + '/'

    sino_file = args['--sino']

    data_path= prefix+sino_file
    value = args['--value']

    remove_nans_and_negatives_from_projdata(data_path, value)
    print("nan succesfully removed from "+data_path+"!")
    