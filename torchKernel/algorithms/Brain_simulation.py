'''running a brainweb simulation with 3 tumours
Usage:
  Brain_simulation [--help | options]

Options:
  -o <out_path>, --outpath=<out_path>     path to data files, defaults to current directory
  --seed=<seed>                           seed of the random process for reproducibility [type: int][default: 0]
  --counts_ratio=<counts_ratio>           when we do counts reduction is the ratio between the original and the increased noise [type: float][default: 1]
'''

# Author: Daniel Deidda
# Copyright 2024 National Physical Laboratory.
# First version: 13th of Nov 2023
# SPDX-License-Identifier: Apache-2.0

import sys
import pathlib
import os
# parent_dir =pathlib.Path(__file__).parent.parent.parent.resolve() 
# #os.path.dirname(os.path.realpath('.'))
# # Add the parent directory to sys.path
# os.chdir(parent_dir)
# sys.path.append(parent_dir) 
# print(parent_dir, )   


# Import the PET reconstruction engine
import sirf
import sirf.STIR as pet
# Set the verbosity
pet.set_verbosity(0)
# Store temporary sinograms in RAM
pet.AcquisitionData.set_storage_scheme("file")
# SIRF STIR message redirector

msg = sirf.STIR.MessageRedirector(info=None, warn=None, errr=None)

from artcertainty.utils.torch_operations import  add_noise, tdivide, save_as_template
from artcertainty.utils.sirf_torch import primal_op as F
from artcertainty.utils.sirf_torch import dual_op as B
from artcertainty.utils.system import create_working_dir_and_move_into, install
from artcertainty.utils.sirf_modelling import add_np_noise, crop_and_save, get_acquisition_model, get_acquisition_model_real_with_norm_and_umap

import brainweb
from brainweb import volshow
import numpy as np
from tqdm.auto import tqdm
import logging
logging.basicConfig(level=logging.INFO)
import sirf.STIR as pet
msg_red = pet.MessageRedirector('info.txt', 'warnings.txt')

from scipy.ndimage import gaussian_filter
#the following creates a working director at the same level of the parent folder
#you can choose another folder if yoy want   
import numpy as np
import time
import torch
from tqdm.notebook import trange, tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sirf.Utilities import examples_data_path
    
def simulate_brain_with_lesion(template_sino, seed, ratio):
    fname, url= sorted(brainweb.utils.LINKS.items())[0]
    files = brainweb.get_file(fname, url, ".")
    data = brainweb.load_file(fname)
    brainweb.seed(1337)

    for f in tqdm([fname], desc="mMR ground truths", unit="subject"):
        vol = brainweb.get_mmr_fromfile(
            f,
            petNoise=1, t1Noise=0.75, t2Noise=0.75,
            petSigma=1, t1Sigma=1, t2Sigma=1)
        vol_amyl = brainweb.get_mmr_fromfile(
            f,
            petNoise=1, t1Noise=0.75, t2Noise=0.75,
            petSigma=1, t1Sigma=1, t2Sigma=1,
            PetClass=brainweb.Amyloid)
        
        FDG_arr  = vol['PET']
        uMap_arr = vol['uMap']
        T1_arr   = vol['T1']
        T2_arr   = vol['T2']
    # We'll need a template sinogram
   # print('open from example_data')
    #mMR_template_sino = examples_data_path('PET') + "/mMR/mMR_template_span11.hs"
    
        
    FDG  = crop_and_save(template_sino, FDG_arr,  "FDG"    )
    uMap = crop_and_save(template_sino, uMap_arr, "uMap"   )
    T1   = crop_and_save(template_sino, T1_arr,   "T1"     )
    T2   = crop_and_save(template_sino, T2_arr,   "T2"     )

    #create tumours
    tumour_arr = FDG.get_uniform_copy(0).as_array()
    # The value of the tumour will be 1.4*the max in the FDG image
    tumour_val = 1.4 * FDG.max()
    ROI1_ar=FDG.get_uniform_copy(0).as_array()
    ROI2_ar=FDG.get_uniform_copy(0).as_array()
    ROI3_ar=FDG.get_uniform_copy(0).as_array()

    # Give the radius of the tumour
    tumour_radius_in_voxels0 = 2
    tumour_radius_in_voxels1 = 3
    tumour_radius_in_voxels2 = 4
    # Amount of smoothing
    gaussian_sigma = 1
    # Index of centre of the tumour
    tumour_centre0 = np.array([7, 50, 90])
    tumour_centre1 = np.array([7, 70, 90])
    tumour_centre2 = np.array([7, 90, 95])
        # Loop over all voxels in the cube containing the sphere
    for i in range(-tumour_radius_in_voxels0, tumour_radius_in_voxels0):
        for j in range(-tumour_radius_in_voxels0, tumour_radius_in_voxels0):
            for k in range(-tumour_radius_in_voxels0, tumour_radius_in_voxels0):
                # If the index is inside of the sphere, set the tumour value
                if (i*i+j*j+k*k < tumour_radius_in_voxels0*tumour_radius_in_voxels0):
                    tumour_arr[tumour_centre0[0]+i,tumour_centre0[1]+j,tumour_centre0[2]+k] = tumour_val
                    ROI1_ar[tumour_centre0[0]+i,tumour_centre0[1]+j,tumour_centre0[2]+k] = 1
    for i in range(-tumour_radius_in_voxels1, tumour_radius_in_voxels1):
        for j in range(-tumour_radius_in_voxels1, tumour_radius_in_voxels1):
            for k in range(-tumour_radius_in_voxels1, tumour_radius_in_voxels1):
                # If the index is inside of the sphere, set the tumour value
                if (i*i+j*j+k*k < tumour_radius_in_voxels1*tumour_radius_in_voxels1):
                    tumour_arr[tumour_centre1[0]+i,tumour_centre1[1]+j,tumour_centre1[2]+k] = tumour_val
                    ROI2_ar[tumour_centre1[0]+i,tumour_centre1[1]+j,tumour_centre1[2]+k] = 1

    for i in range(-tumour_radius_in_voxels2, tumour_radius_in_voxels2):
        for j in range(-tumour_radius_in_voxels2, tumour_radius_in_voxels2):
            for k in range(-tumour_radius_in_voxels2, tumour_radius_in_voxels2):
                # If the index is inside of the sphere, set the tumour value
                if (i*i+j*j+k*k < tumour_radius_in_voxels2*tumour_radius_in_voxels2):
                    tumour_arr[tumour_centre2[0]+i,tumour_centre2[1]+j,tumour_centre2[2]+k] = tumour_val
                    ROI3_ar[tumour_centre2[0]+i,tumour_centre2[1]+j,tumour_centre2[2]+k] = 1

        # Smooth the tumour image
        #tumour_arr = gaussian_filter(tumour_arr, sigma=0.6)
               
    # Overwrite add
    tumour_arr = np.max([FDG.as_array(),tumour_arr],axis=0)

    # Fill into new ImageData object
    pet_tumour = FDG.clone()
    ROI1 = FDG.get_uniform_copy(0)
    ROI2 = FDG.get_uniform_copy(0)
    ROI3 = FDG.get_uniform_copy(0)
    pet_tumour.fill(tumour_arr)
    pet_tumour.write('FDG_tumour_small')
    ROI1.fill(ROI1_ar)
    ROI1.write('ROI1')
    ROI2.fill(ROI2_ar)
    ROI2.write('ROI2')
    ROI3.fill(ROI3_ar)
    ROI3.write('ROI3')

    # am = get_acquisition_model(template_sino, uMap)
    am = get_acquisition_model(template_sino, uMap)
    # am.set_up(template_sino, pet_tumour)
        
    # FDG
    sino_FDG = am.forward(FDG)
    sino_FDG.write("FDG_sino")
    sino_FDG_noisy = add_np_noise(sino_FDG,ratio)
    sino_FDG_noisy.write("FDG_sino_noisy_seed"+str(seed))
        
    # with tumours
    sino_tumours = am.forward(pet_tumour)
    sino_tumours.write("FDG_tumour_sino_small")
    sino_tumours_noisy = add_np_noise(sino_tumours, ratio)
    sino_tumours_noisy.write("FDG_tumour_sino_small_noisy_seed"+str(seed))
    # ones=FDG.get_uniform_copy(1)
    # onest =  am.forward(ones)
    # onest.write('FPones')
    return sino_tumours_noisy
    
def make_image_simulation_2d(sinogram_template):#it's a 2d template
    
    true_brain = pet.ImageData('FDG_tumour_small.hv') #(150x150)
    umap = pet.ImageData('uMap_small.hv')
    T1 = pet.ImageData('T1_small.hv')
    ROI1 = pet.ImageData('ROI1.hv')
    ROI2 = pet.ImageData('ROI2.hv')
    ROI3 = pet.ImageData('ROI3.hv')

    true_brain_np = true_brain.as_array()
    umap_np = umap.as_array()
    T1_np = T1.as_array()

    # x and y size to in Unet need to be divisible  by 2, 3 times
    true_brain_np_2d = np.expand_dims(true_brain_np[7,2:146,2:146], axis=0)
    umap_np_2d = np.expand_dims(umap_np[7,2:146,2:146], axis=0)
    T1_np_2d = np.expand_dims(T1_np[7,2:146,2:146], axis=0)
    roi1_2d = np.expand_dims(ROI1.as_array()[7,2:146,2:146], axis=0)
    roi2_2d = np.expand_dims(ROI2.as_array()[7,2:146,2:146], axis=0)
    roi3_2d = np.expand_dims(ROI3.as_array()[7,2:146,2:146], axis=0)

    #make 2d template image
    image_template = sinogram_template.create_uniform_image(1.0,true_brain_np_2d.shape[1]);

    true_brain_2d=image_template.clone().fill(true_brain_np_2d)
    true_brain_2d.write("FDG_tumour_2d")

    T1_2d=image_template.clone().fill(T1_np_2d)
    T1_2d.write("T1_2d")

    umap_2d = image_template.clone().fill(umap_np_2d)
    umap_2d.write("uMap_2d")

    ROI12d = image_template.clone().fill(roi1_2d)
    ROI12d.write("ROI1_2d")
    ROI22d = image_template.clone().fill(roi2_2d)
    ROI22d.write("ROI2_2d")
    ROI32d = image_template.clone().fill(roi3_2d)
    ROI32d.write("ROI3_2d")

    return umap_2d, umap_np_2d, T1_np_2d, true_brain_np_2d
    
    # tt torch tensor
def get_tensors(umap_np_2d,T1_np_2d, true_brain_np_2d):
    true_brain_tt_2d = torch.from_numpy(true_brain_np_2d).repeat(1,1,1,1).to(device)
    T1_tt_2d = torch.from_numpy(T1_np_2d).repeat(1,1,1,1).to(device)
    umap_tt_2d = torch.from_numpy(umap_np_2d).repeat(1,1,1,1).to(device)
    return true_brain_tt_2d, T1_tt_2d, umap_tt_2d
    
def simulate_2d_data(sinogram_template, umap, true_brain_tt_2d, seed, ratio):
    acq_model = get_acquisition_model(sinogram_template, umap)
    #umap here is used as a image template
    image_template=umap
    # acq_model.set_up(sinogram_template,image_template);
    xt  = true_brain_tt_2d
    f = F(image_template,sinogram_template,acq_model).to(device)
    b = B(image_template,sinogram_template,acq_model).to(device)
        
    y0=f(xt)
    y =  add_noise(y0,torch.Generator(seed),ratio).to(device)
    return y, y0

def Brain_simulation(seed,ratio):
    sinogram_template_2d = pet.AcquisitionData(examples_data_path('PET')\
                                        + '/thorax_single_slice/template_sinogram.hs');
    sinogram_template = pet.AcquisitionData(examples_data_path('PET') + "/mMR/mMR_template_span11.hs");
    
    sino_tumours_noisy=simulate_brain_with_lesion(sinogram_template, seed, ratio)
    umap, umap_np_2d, T1_np_2d, true_brain_np_2d = make_image_simulation_2d(sinogram_template_2d)
    true_brain_tt_2d, T1_tt_2d, umap_tt_2d = get_tensors(umap_np_2d,T1_np_2d, true_brain_np_2d)
    image_template = sinogram_template_2d.create_uniform_image(1.0,true_brain_np_2d.shape[1]);
    true_brain_2d_sirf = image_template.clone().fill(true_brain_np_2d)

    am = get_acquisition_model(sinogram_template_2d, umap)
    sino_2d_tumours = am.forward(true_brain_2d_sirf)
    sino_2d_tumours_noisy = add_np_noise(sino_2d_tumours, ratio)
    print(sino_2d_tumours_noisy.max(), sino_2d_tumours.max())
    sino_tumours_noisy.write("FDG_tumour_sino_small_noisy_seed"+str(seed))
    # y, y0 = simulate_2d_data(sinogram_template_2d, umap, true_brain_tt_2d)

    # y_sirf = sinogram_template_2d.clone().fill(sino_2d_tumours_noisy)
    # print(y_sirf.max())
    sino_2d_tumours_noisy.write("FDG_tumour_sino_2d_noisy_seed"+str(seed))

    y0_sirf = sinogram_template_2d.fill(sino_2d_tumours)
    y0_sirf.write("FDG_tumour_sino_2d")

if __name__ == '__main__':
    from docopt import docopt
    __version__ = '0.1.0'
    args = docopt(__doc__, version=__version__)
    print (args)

    data_output_path = args['--outpath']
    if data_output_path is None:
            data_output_path ='.'
    prefix = data_output_path+'/'
    print(data_output_path)
    seed = args['--seed']
    ratio = args['--counts_ratio']

    create_working_dir_and_move_into(prefix)


    # Start timer
    start_time = time.time()

    Brain_simulation(seed,ratio)

    # End timer
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
