'''
script to remove nan from interfile imagegrams


Usage:
  remove_nans_from_projdata [--help | options]

Options:
  -s <file>, --image=<file>                          image interfile. All the file paths needs to be relative to out_path/working_dir
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
# Store temporary imagegrams in RAM
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

def remove_nans_and_negatives_from_image(data_path, value):
    image = pet.ImageData(data_path);
    image_np = image.as_array()
    image_np = np.nan_to_num(image_np, nan=value, posinf=value, neginf=value) 
    # also remove negatives
    mask = (image_np >= 0)
    c = np.zeros(image_np.shape)
    c[mask] = image_np[mask]

    image.fill(c)
    image.write(data_path)

if __name__ == '__main__':
    from type_docopt import docopt
    args = docopt(__doc__, version='0.1.0')
    print(args)

    
    working_dir =   os.getcwd()
    prefix = working_dir + '/'

    image_file = args['--image']

    data_path= prefix+image_file
    value = args['--value']

    remove_nans_and_negatives_from_image(data_path, value)
    print("nan succesfully removed from "+data_path+"!")
    