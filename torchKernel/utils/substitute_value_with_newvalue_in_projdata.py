'''
script to substitute a crtain value with another from interfile sinograms


Usage:
  substitute_value_with_newvalue_in_projdata [--help | options]

Options:
  -s <file>, --sino=<file>                          raw data file. All the file paths needs to be relative to out_path/working_dir
  -v <value>, --value=<value>                       value to be substituted [default: 0] [type: float]
  -n <new-value>, --new-value=<new-value>           new value to use instead of value [default: 1] [type: float]
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
import sirf

import sirf.STIR as pet
from sirf.contrib.NEMA import generate_nema_rois
msg = sirf.STIR.MessageRedirector(info=None, warn=None, errr=None)
# Load dataset and model
from artcertainty.utils.torch_operations import  treshold_tensor
from artcertainty.utils.system import create_working_dir_and_move_into

import os
import numpy as np

def substitute_value_with_newvalue_in_projdata(data_path, value, new_value):
    sino = pet.AcquisitionData(data_path);
    sino_np = sino.as_array()
    mask = (sino_np != 0)
    c = np.ones(sino_np.shape)
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
    new_value = args['--new-value']

    substitute_value_with_newvalue_in_projdata(data_path, value, new_value)
    print(str(value)+" succesfully substitute with "+str(new_value)+ " from "+data_path+"!")
    