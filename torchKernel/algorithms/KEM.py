"""
script running the KEM with patches

this is an implementation of the following manuscript:

Wang G, Qi J. PET image reconstruction using kernel method. IEEE transactions on medical imaging. 2014 Jul 30;34(1):61-71.
doi: 10.1109/TMI.2014.2343916

Usage:
  KEM [--help | options]

Options:
  -s <file> --sino=<file>                                       raw data file. All the file paths needs to be relative to out_path/working_dir
  -f <format> --format=<format>                                 format to save into 'template', 'npy', 'nii' [default: npy]
  -a <anatomic_image> --anatomic=<anatomic_image>               interfile image of the image you want to use as a prior
  -u <umap_image> --umap=<umap_image>                           interfile image of the umap image for attenuation
  --add_sino=<addittive_sino>                                   interfile data with randoms and scatter
  --n_sino=<norm_sino>                                          interfile data of the normalisation
  -o <out_path> --outpath=<out_path> 
  -e <epoch_cp> --epoch_cp=<epoch_cp>                           load from epoch_cp [default 0] [type: int]
  -k <knn> --knn=<knn>                                          numbero of k nearest neigbours for the Kernel [default: 48] [type: int]
  -w <window> --window=<window>                                 window or size of the neighbourhood (w x w) or (w x w x w) for 3d [default: 9] [type: int]
  --ksigma=<ksigma>                                             sigma parameter of the kernel function for edge preservation [default: 0.5] [type: float]
  --is2d=<is2d>                                                 for 2d or 3d data [default: 0] [type: int]
  -i <it>, --iter=<it>                                          number of iteration [default: 10] [type: int]
  --save_every=<save_every>                                     save image every <save_every> iterations [default: 1] [type: int]
  --num_subsets=<num_subsets>                                   number of subsets for HKEM initialisation [default: 8] [type: int]
  --psf_fwhm=<psf_fwhmfrom artcertainty.utils.system import check_reserved_memory, check_pytorch_gpu, clear_pytorch_cache>                                         full width half maximum for the point spread function (PSF) [default: 0] [type: float]
  --seed=<seed>                                                 seed of the random process for reproducibility [default: 42] [type: int]
  --a_seed=<a_seed>                                             seed of the random process for aleatory uncertanties [type: int]
  --save_mem_k=<save_mem_k>                                     if 1 it will run another function for K calculation that does not save the K [default: 0] [type: int]                                                                                      
"""

# Author: Daniel Deidda
# Copyright 2024 National Physical Laboratory.
# First version: 21st of Oct 2024
# SPDX-License-Identifier: Apache-2.0
from type_docopt import docopt
import sys
# Import the PET reconstruction engine
import sirf.STIR as pet
# Set the verbosity
pet.set_verbosity(0)
# Store temporary sinograms in RAM
pet.AcquisitionData.set_storage_scheme("file")
# SIRF STIR message redirector
import sirf
import sirf.STIR as pet
msg = sirf.STIR.MessageRedirector(info=None, warn=None, errr=None)
from artcertainty.kernel.LHK import BuildK
from artcertainty.utils.torch_operations import tdivide, save_as, update_subset_model, make_cylindrical_mask_tensor
from artcertainty.utils.sirf_torch import primal_op as FP
from artcertainty.utils.sirf_torch import dual_op as BP
from artcertainty.utils.system import create_working_dir_and_move_into
from artcertainty.utils.from_sirf_ex import get_acquisition_model
from artcertainty.algorithms.Algorithm import Algorithm
from artcertainty.algorithms.Algorithm import read_simulation, get_working_dir_from_outpath
from artcertainty.utils.system import check_reserved_memory, check_pytorch_gpu, clear_pytorch_cache

import numpy as np
import time
import torch
import os
from tqdm.auto import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KEM(Algorithm):

    def __init__(self, num_subsets, kem_iter, is2d, w, k, 
                 ksigma, anat, sinogram_template, umap, norm, 
                 additive, format, save_every, is_real, psf_fwhm, 
                 epoch_checkpoint, a_seed, save_mem_k):
        
        super().__init__(num_subsets, is2d, anat, sinogram_template, norm, additive,
                         umap, format, save_every,is_real, psf_fwhm, kem_iter,
                         epoch_checkpoint, a_seed)
        
        self.kem_iter = kem_iter
        self.is2d = is2d
        self.w = w
        self.k = k
        self.ksigma = ksigma
        self.num_subsets = num_subsets
        self.save_mem_k = save_mem_k
        # the following is needed because we do comparison with integers
        if (self.e_cpoint is None):
            self.e_cpoint = 0     
        
        
    def run(self, seed):

        #make the algorithm reproducible
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        if self.psf_fwhm>0:
            psf='_PSF'+str(self.psf_fwhm)  
        else:
            psf=''

        if self.a_seed is None:
            a_seed_str = ''
        else:
            a_seed_str = '_aseed'+str(self.a_seed)
        

        
        #create kernel
        BK=BuildK(self.ksigma,False,self.save_mem_k)
        if self.save_mem_k:
            BK(self.anatomical_tensor,self.k,self.w)
        else:
            K=BK(self.anatomical_tensor,self.k,self.w)
        print("Kernel succesfully calculated")
        
        alpha=torch.ones(self.anatomical_tensor.shape).to(device)

        alpha_masked = torch.zeros(self.anatomical_tensor.shape).to(device)
        curr_kem_i = torch.zeros(self.anatomical_tensor.shape).to(device)
        mask_im = make_cylindrical_mask_tensor(alpha)
        mask = (mask_im==1)

        out_filename_pref = 'KEM_2d'+str(self.is2d)+psf+'_k'+str(self.k)+'_w'+str(self.w)+'_ks'+str(self.ksigma)+'_seed'+str(seed)+a_seed_str+'_eit'
        out_filename_pref = out_filename_pref.replace('.','_')
        PoissonLoglikelihood = {}
        PoissonLoglikelihood["loss"] = [ ]
        PoissonLoglikelihood["epoch"] = [ ]
                

        if (self.e_cpoint>0):
            PoissonLoglikelihood = np.load(out_filename_pref.replace('.','_')+'_data_log.npy', allow_pickle=True).item()
            alpha = torch.load(out_filename_pref+'_alpha.torch').to(self.data_tensor.device)
            frozen_alpha = torch.load(out_filename_pref+'_frozen_alpha.torch').to(self.data_tensor.device)
            l=BK.kernelise_image(K,alpha) 

        for i in tqdm(range(self.e_cpoint,self.epochs+self.e_cpoint),position=0, initial=self.e_cpoint):
            for s in range(self.num_subsets):
                with torch.no_grad():
                    alpha_masked[mask]=alpha[mask] 
                    if (self.is_real):
                        fs, bs, sens, ys, ads= update_subset_model(FP,BP,self.data_tensor,self.num_subsets,s,
                                                                    self.umap,self.sinogram_template,
                                                                    self.image_template, self.unnorm_add_tt,
                                                                    self.normacf_tt)
                    else:
                        fs, bs, sens, ys= update_subset_model(FP,BP,self.data_tensor,self.num_subsets,s,
                                                                    self.umap,self.sinogram_template,
                                                                    self.image_template) 
                                
                    if self.save_mem_k:
                        ksens=BK.kernelise_image_save_mem_t(sens)
                        ka= BK.kernelise_image_save_mem_t(alpha_masked)
                        fka=fs(ka).to(device)
                        if (self.is_real):
                            fka = fka+ads
                        grad=bs((tdivide(ys,fka)))
                        kgrad=BK.kernelise_image_save_mem_t(grad)

                    else:
                        ksens=BK.kernelise_image(K.t(),sens)
                        ka= BK.kernelise_image(K,alpha_masked)
                        fka=fs(ka).to(device)
                        if (self.is_real):
                            fka = fka+ads
                        grad=bs((tdivide(ys,fka)))
                        kgrad=BK.kernelise_image(K.t(),grad)
                    curr_kem_i[mask] =  tdivide(alpha_masked[mask],ksens[mask])*kgrad[mask].to(device)
                    alpha =curr_kem_i
                    
                    alpha=torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)                

        #save things
            if self.save_mem_k:
                l=BK.kernelise_image_save_mem(alpha)
            else:
                l=BK.kernelise_image(K,alpha)  
            
            if (self.is_real):
                if self.num_subsets>1:
                    ybar= self.fp(l)+self.unnorm_add_tt.to(device)
                else:
                    subset_lenght = int(self.unnorm_add_tt.shape[self.unnorm_add_tt.dim()-2]/self.num_subsets)
                    ads=self.unnorm_add_tt.to(device)[:,:,:,s*subset_lenght:s*subset_lenght+subset_lenght,:]
                    ybar= self.fp(l) + ads.to(device)
            else:
                ybar = self.fp(l)

            PoissonLoglikelihood["loss"].append(torch.sum(self.data_tensor*torch.log(torch.sum(ybar)) - torch.sum(ybar)).cpu().item())#yilogy¯i−y¯i
            PoissonLoglikelihood["epoch"].append(i)
            np.save(out_filename_pref+'_data_log', PoissonLoglikelihood )
            if ((i+1) % self.save_every == 0):
                save_as(self.format, l, self.image_template,out_filename_pref+str(i))
        return l, alpha
def main():
    args = docopt(__doc__, version='1.0.0rc2')
    print(args)

    data_output_path = args['--outpath']
    working_dir = get_working_dir_from_outpath(data_output_path)

    is2d = args['--is2d']
    sino_file = args['--sino']
    a_seed = args['--a_seed']
    if sino_file is None:
        true_brain_tt, umap, anat, sinogram_template, norm, additive =read_simulation(is2d, a_seed)
    else:
        sinogram_template = pet.AcquisitionData(sino_file)

        norm_file = args['--n_sino']
        if norm_file is None:
            norm = sinogram_template.get_uniform_copy(1)
        else:
            norm = pet.AcquisitionData(norm_file)

        add_file = args['--add_sino']
        if add_file is None:
            additive = sinogram_template.get_uniform_copy(0)
        else:
            additive = pet.AcquisitionData(add_file)
        
        anat_file = args['--anatomic']
        if anat_file is None:
            sys.exit('Missing the anatomical image')
        anat = pet.ImageData(anat_file)

        umap_file = args['--umap']
        if umap_file is None:
            sys.exit('Missing the umap image')
        umap = pet.ImageData(umap_file)
    psf_fwhm = args['--psf_fwhm']
    seed = args['--seed']
    epoch_checkpoint  = args['--epoch_cp']

    ksigma = args['--ksigma']   
    k = args['--knn']      
    w = args['--window']     
    iter = args['--iter']    
    save_every = args['--save_every']
    format = args['--format']
    num_subsets = args['--num_subsets']

    #set parameters
    is_real=(not sino_file == None)
    save_mem_k = args['--save_mem_k']

    # Start timer
    start_time = time.time()

    algorithm = KEM(num_subsets, iter,
                          is2d, w, k, ksigma, anat, sinogram_template,
                          umap, norm, additive, format, save_every, is_real,
                          psf_fwhm, epoch_checkpoint, a_seed, save_mem_k)
    algorithm.run(seed)

    # End timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

if __name__ == '__main__':
    main()