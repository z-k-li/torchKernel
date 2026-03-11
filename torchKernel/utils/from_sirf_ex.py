"""
Functions inspire by SIRF exercises

Author: Daniel Deidda

Copyright 2024 National Physical Laboratory.

SPDX-License-Identifier: Apache-2.0

"""
import sirf.STIR as pet
# Set the verbosity
pet.set_verbosity(0)
# Store temporary sinograms in RAM
pet.AcquisitionData.set_storage_scheme("memory")
# SIRF STIR message redirector
import sirf
import sirf.STIR as pet
import numpy as np
def get_acquisition_model_with_normacf(templ_sino, templ_image, norm_acf_sino):
    '''create an acq_model given a sinogram template, an image template, and a
       normacf sinogram
    '''

    norm_acf_sino_np = norm_acf_sino.as_array()
    norm_acf_sino_np=np.nan_to_num(norm_acf_sino_np, nan=0.0, posinf=0.0, neginf=0.0) 
    norm_acf_sino.fill(norm_acf_sino_np)


    inv_norm_acf_sino_np = np.reciprocal(norm_acf_sino_np)
    inv_norm_acf_sino_np = np.nan_to_num(inv_norm_acf_sino_np, nan=0.0, posinf=0.0, neginf=0.0) 
    inv_norm_acf_sino = norm_acf_sino.copy().fill(inv_norm_acf_sino_np)
    
    # create acquisition model
    am = pet.AcquisitionModelUsingParallelproj()
    
    ## Norm

    asm_norm = pet.AcquisitionSensitivityModel(inv_norm_acf_sino)
    # asm_norm.set_up(templ_sino)
    # det_efficiencies=templ_sino.get_uniform_copy(1)
    # asm_norm.unnormalise(det_efficiencies)
    am.set_acquisition_sensitivity(asm_norm)
    
    # update the acquisition model etc
    # am.set_acquisition_sensitivity(asm_norm)
    am.set_up(templ_sino,templ_image)   
    return am, inv_norm_acf_sino

def get_acquisition_model_real_with_norm_and_umap(templ_sino, norm_sino, uMap):
    '''create an acq_model given a a template, norm sinogram, and a mu-map
    '''
    norm_sino_np = norm_sino.as_array()
    norm_sino_np=np.nan_to_num(norm_sino_np, nan=0.0, posinf=0.0, neginf=0.0) 
    norm_sino.fill(norm_sino_np)


    inv_norm_sino_np = 1/norm_sino_np
    inv_norm_sino_np = np.nan_to_num(inv_norm_sino_np, nan=0.0, posinf=0.0, neginf=0.0) 
    inv_norm_sino = norm_sino.copy().fill(inv_norm_sino_np)
    acq_model = pet.AcquisitionModelUsingParallelproj()
    ## Norm
    asm_norm = pet.AcquisitionSensitivityModel(inv_norm_sino)
    acq_model.set_acquisition_sensitivity(asm_norm)
    # Attenuation
    attn_acq_model = pet.AcquisitionModelUsingParallelproj()
    asm_attn = pet.AcquisitionSensitivityModel(uMap, attn_acq_model)
    # converting attenuation into attenuation factors 
    asm_attn.set_up(templ_sino)
    attn_factors = templ_sino.get_uniform_copy(1)
    print('applying attenuation (please wait, may take a while)...')
    asm_attn.normalise(attn_factors)
    # use these in the final attenuation model
    attn_factors_inv = attn_factors.clone().fill( np.nan_to_num(np.reciprocal(attn_factors.as_array()), nan=0.0, posinf=0.0, neginf=0.0) )
    asm_attn = pet.AcquisitionSensitivityModel(attn_factors_inv)
    # chain attenuation and normalisation
    asm = pet.AcquisitionSensitivityModel(asm_norm, asm_attn)
    acq_model.set_acquisition_sensitivity(asm)
    acq_model.set_up(templ_sino,uMap)

    return acq_model, asm_norm, attn_factors
#%% this is a copy from SIRFexercise BrainWeb
def get_acquisition_model(templ_sino, uMap, global_factor=1):
    '''create an acq_model given a mu-map and a global sensitivity factor
    
    The default global_factor is chosen such that the mean values of the
    forward projected BrainWeb data have a reasonable magnitude
    '''
    #%% create acquisition model
    am = pet.AcquisitionModelUsingParallelproj()
    
    # Set up sensitivity due to attenuation
    asm_attn = pet.AcquisitionSensitivityModel(uMap, am)
    asm_attn.set_up(templ_sino)
    attn_factors = templ_sino.get_uniform_copy(global_factor)
    #print('applying attenuation (please wait, may take a while)...')
    asm_attn.normalise(attn_factors)
    # use these in the final attenuation model
    asm_attn = pet.AcquisitionSensitivityModel(attn_factors)
    # chain attenuation and normalisation
    am.set_acquisition_sensitivity(asm_attn)
    
    am.set_up(templ_sino,uMap);#using mumap as template
    return am


# Function for adding noise
def add_np_noise(proj_data,noise_factor = 1):
    proj_data_arr = proj_data.as_array() / noise_factor
    # Data should be >=0 anyway, but add abs just to be safe
    proj_data_arr = np.abs(proj_data_arr)
    noisy_proj_data_arr = np.random.poisson(proj_data_arr).astype('float32');
    noisy_proj_data = proj_data.get_uniform_copy()
    noisy_proj_data.fill(noisy_proj_data_arr);
    return noisy_proj_data
def crop_and_save(templ_sino, vol, fname):
    # Crop from (127,344,344) to (127,285,285) and save to file
    vol = vol[60-7:60+7,17:17+285,17:17+285]
    im = pet.ImageData(templ_sino)
    im = im.zoom_image(size=(14,-1,-1),offsets_in_mm=(0,0,0))
    im.fill(vol)
    im.write(fname)
    # Create an optional smaller version, (127,150,150)
    # For extra speeeed.
    # Also shift by (25,25) in (x,y) to recentre the image
    im = im.zoom_image(size=(-1,150,150),offsets_in_mm=(0,25,25))
    im = im.move_to_scanner_centre(templ_sino)
    im.write(fname + "_small.hv")
    return im
