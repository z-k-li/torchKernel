'''
Parent class for all the algorithms.
'''

# Author: Daniel Deidda
# Copyright 2024 National Physical Laboratory.
# First version: 13th of Jan 2024
# SPDX-License-Identifier: Apache-2.0

import sys
# Import the PET reconstruction engine
import sirf.STIR as pet
# Set the verbosity
pet.set_verbosity(0)
# Store temporary sinograms in RAM
#pet.AcquisitionData.set_storage_scheme("memory")
# SIRF STIR message redirector
import sirf
import sirf.STIR as pet
msg = sirf.STIR.MessageRedirector(info=None, warn=None, errr=None)
from artcertainty.kernel.LHK import kernelise_image, set_KOSMAPOSL
from artcertainty.architectures.UNet import  AttentionUNet
from artcertainty.utils.torch_operations import tdivide, npdivide, save_as, update_subset_model
from artcertainty.utils.sirf_torch import primal_op as FP
from artcertainty.utils.sirf_torch import dual_op as BP
from artcertainty.utils.system import create_working_dir_and_move_into
from artcertainty.utils.sirf_modelling import  get_acquisition_model_real_with_norm_and_umap, get_acquisition_model

import numpy as np
import time
import torch
import os
from tqdm.auto import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Algorithm:

    def __init__(self, num_subsets, is2d, anat, sinogram_template, norm, additive,
                 umap, format, save_every, is_real, psf_fwhm, epochs, epoch_checkpoint,
                 a_seed): #a_seed for aleatory uncertanty. takes into account different data samples

        self.num_subsets = num_subsets
        self.is2d = is2d
        self.anat = anat
        self.sinogram_template = sinogram_template
        self.umap = umap
        self.psf_fwhm = psf_fwhm
        self.norm = norm
        self.add = additive
        self.format = format
        self.save_every = save_every
        self.is_real = is_real
        self.epochs = epochs
        self.e_cpoint = epoch_checkpoint
        self.a_seed = a_seed


        assert self.epochs>= (self.epochs -1), "the number of epochs ("+str(self.epochs)+ ") must be higher than <save_every>="+str(self.save_every)

        umap_np = self.umap.as_array()
        umap_np=np.nan_to_num(umap_np, nan=0.0, posinf=0.0, neginf=0.0) 
        self.umap.fill(umap_np)
        
        self.image_template = umap.clone()

        anat_np = self.anat.as_array()
        anat_np=np.nan_to_num(anat_np, nan=0.0, posinf=0.0, neginf=0.0) 
        self.anat.fill(anat_np/anat_np.max())
        # anat=self.anat.clone().fill(umap_np/umap_np.max())

        if(self.is_real):
            self.acq_model, asm_norm, self.acf = get_acquisition_model_real_with_norm_and_umap(self.sinogram_template, self.norm,self.umap)
            del asm_norm
            torch.cuda.empty_cache()
            acf_np = self.acf.as_array()
            self.norm_np = self.norm.as_array()
            self.norm_np=np.nan_to_num(self.norm_np, nan=0.0, posinf=0.0, neginf=0.0) 
            normacf_np = self.norm_np* acf_np
            self.normacf_np=np.nan_to_num(normacf_np, nan=0.0, posinf=0.0, neginf=0.0) 
            
            if (self.psf_fwhm>0):
                psf = pet.SeparableGaussianImageFilter()
                psf.set_fwhms((self.psf_fwhm, self.psf_fwhm, self.psf_fwhm))
                self.acq_model.set_image_data_processor(psf)

            if(self.is2d):

                # self.normacf_tt = torch.from_numpy(normacf_np).repeat(1,1,1,1).to(device)
                # self.norm_tt = torch.from_numpy(self.norm_np).repeat(1,1,1,1).to(device)
                # self.acf_tt = torch.from_numpy(acf_np).repeat(1,1,1,1).to(device)
                unnorm_add_np = npdivide(self.add.as_array(),normacf_np)
                self.unnorm_add_tt = torch.from_numpy(unnorm_add_np).repeat(1,1,1,1).to(device) 
            else:
                # self.normacf_tt = torch.from_numpy(normacf_np).repeat(1,1,1,1,1).to(device)
                # self.norm_tt = torch.from_numpy(self.norm_np).repeat(1,1,1,1,1).to(device)
                # self.acf_tt = torch.from_numpy(acf_np).repeat(1,1,1,1,1).to(device)
                unnorm_add_np = npdivide(self.add.as_array(),normacf_np)
                self.unnorm_add_tt = torch.from_numpy(unnorm_add_np).repeat(1,1,1,1,1).to(device)
            
            self.unnorm_add = self.sinogram_template.clone().fill(unnorm_add_np)
        
        else:
            self.acq_model = get_acquisition_model(self.sinogram_template,self.umap)
        
        # anat_np = self.anat.as_array()
        # anat_np = np.nan_to_num(anat_np, nan=0.0, posinf=0.0, neginf=0.0)
        # anat.fill(anat_np)
        # umap_np = self.umap.as_array()
        # umap_np = np.nan_to_num(umap_np, nan=0.0, posinf=0.0, neginf=0.0)
        # umap.fill(umap_np)

        if self.is2d: 
            self.data_tensor=torch.from_numpy(self.sinogram_template.as_array()).repeat(1,1,1,1).to(device)
            umap_tt = torch.from_numpy(umap_np).repeat(1,1,1,1).to(device)  
            self.anatomical_tensor = torch.from_numpy(anat_np/anat_np.max()).repeat(1,1,1,1).to(device)
            
        else:
            self.data_tensor=torch.from_numpy(self.sinogram_template.as_array()).repeat(1,1,1,1,1).to(device)
            umap_tt = torch.from_numpy(umap_np).repeat(1,1,1,1,1).to(device)  
            self.anatomical_tensor = torch.from_numpy(anat_np/anat_np.max()).repeat(1,1,1,1,1).to(device)       

        self.anat=self.anat/self.anat.max()
        self.image_template=self.umap.clone()

        # set up model
        self.fp = FP(self.image_template,self.sinogram_template,self.acq_model).to(device).requires_grad_(False)
        self.bp = BP(self.image_template,self.sinogram_template,self.acq_model).to(device).requires_grad_(False)
        torch.cuda.empty_cache()

    def read_checkpoint(self, out_filename_pref, net, optimiser):
        
        ati_np=np.load(out_filename_pref.replace('.','_')+str(self.e_cpoint-1)+'.npy', allow_pickle=True)
        ati = torch.from_numpy(ati_np).to(device)

        data_logloss = np.load(out_filename_pref.replace('.','_')+'_data_log.npy', allow_pickle=True).item()

        checkpoint = torch.load(out_filename_pref.replace('.','_')+'_trained_extra.torch_model', map_location=device)
        net.load_state_dict(checkpoint['model_state_dict_net'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        
        return ati, data_logloss, net, optimiser

def read_simulation(is2d, a_seed):
    # read the simulated brain data. This need to be generated with Brain_simulation.py first, or else you will get an error
    print('Warning: since no data has been provided I am going to use the data obtained from Brain_simulation.py')
    if a_seed is None:
        a_seed_str = '_seed0'
    else:
        a_seed_str = '_seed'+str(a_seed)

    if is2d:
        if (os.path.isfile('FDG_tumour_sino_2d_noisy'+a_seed_str+'.hs')):
            sinogram_template = pet.AcquisitionData('FDG_tumour_sino_2d_noisy'+a_seed_str+'.hs')
            true_brain = pet.ImageData('FDG_tumour_2d.hv') #(144x144)
            umap = pet.ImageData('uMap_2d.hv')
            anat = pet.ImageData('T1_2d.hv')
            true_brain_np = true_brain.as_array()
            true_brain_tt = torch.from_numpy(true_brain_np).repeat(1,1,1,1).to(device)# tt torch tensor
        else:
            sys.exit('Missing the input sinogram as interfile you need to run the Brain_simulation')
    else:

        if (os.path.isfile('FDG_tumour_sino_small_noisy'+a_seed_str+'.hs')):
            sinogram_template = pet.AcquisitionData('FDG_tumour_sino_small_noisy'+a_seed_str+'.hs')
            true_brain = pet.ImageData('FDG_tumour_small.hv') #(144x144)
            umap = pet.ImageData('uMap_small.hv')
            anat = pet.ImageData('T1_small.hv')
            true_brain_np = true_brain.as_array()
            true_brain_tt = torch.from_numpy(true_brain_np).repeat(1,1,1,1,1).to(device)# tt torch tensor
    
        else:
            sys.exit('Missing the input sinogram as interfile you need to run the Brain_simulation')

    norm = sinogram_template.get_uniform_copy(1)
    add = sinogram_template.get_uniform_copy(0)
    return true_brain_tt, umap, anat, sinogram_template, norm, add

def get_working_dir_from_outpath(data_output_path):
        
    if data_output_path is None:
        data_output_path =  os.getcwd()
    prefix = data_output_path + '/'
    # print(os.path.abspath(prefix))

    return create_working_dir_and_move_into(os.path.abspath(prefix))
