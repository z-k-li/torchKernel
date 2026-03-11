'''
running the  Hybrid Kernelised Expectation Maximisation algorithm 
this is an implementation of the following manuscript:

Deidda, Daniel, Nicolas A. Karakatsanis, Philip M. Robson, Yu-Jung Tsai, Nikos Efthimiou, 
Kris Thielemans, Zahi A. Fayad, Robert G. Aykroyd, and Charalampos Tsoumpas.
"Hybrid PET-MR list-mode kernelized expectation maximization reconstruction."
Inverse Problems 35, no. 4 (2019): 044001.

Usage:
  HKEM [--help | options]

Options:
  -s <file>, --sino=<file>                          raw data file. All the file paths needs to be relative to out_path/working_dir
  -f <format>, --format=<format>                    format to save into 'template', 'npy', 'nii' [default: npy]
  -a <anatomic_image> --anatomic=<anatomic_image>   interfile image of the image you want to use as a prior
  -u <umap_image> --umap=<umap_image>               interfile image of the umap image for attenuation
  --add_sino=<addittive_sino>                       interfile data with randoms and scatter
  --n_sino=<norm_sino>                              interfile data of the normalisation
  -o <out_path>, --outpath=<out_path>               path to data files, defaults to alpha directory [default: .]
  -w <window>, --window=<window>                    window or size of the neighbourhood (w x w) or (w x w x w) for 3d [default: 5] [type: int]
  -p <sigma_p>, --sigma_p=<sigma_p>                 parameter for iterative image edge preservation [default: 0.6] [type: float]
  -m <sigma_m>, --sigma_m=<sigma_m>                 parameter for anatomical image edge preservation [default: 0.5] [type: float]
  --sigma_dm=<sigma_dm>                             parameter for distance based weight [default: 5] [type: int]
  --frozen=<frozen>                                 parameter to freez functional iterative update to a certain iteration [type: float]
  --is2d=<is2d>                 f                   for 2d or 3d data [default: 1] [type: int]
  -h <isHybrid>, --isHybrid=<isHybrid>              parameter to select KEM(False) or HKEM(True) [default: 1] [type: int]
  -i <it>, --iter=<it>                              number of iterations  [default: 200] [type: int]
  --save_every=<save_every>                         save image every <save_every> iterations [default: 1] [type: int]
  --num_subsets=<num_subsets>                       number of subsets for HKEM initialisation [default: 8] [type: int]
  --psf_fwhm=<psf_fwhm>                             full width half maximum for the point spread function (PSF) [default: 0] [type: float]
  -e <epoch_checkpoint>, --epoch_checkpoint=<epoch_checkpoint>    start from a previously trained model ended at epoch_checkpoint [default 0] [type: int]
  --seed=<seed>                                                 seed of the random process for reproducibility [default: 42] [type: int]
  --a_seed=<a_seed>                                 seed of the random process for aleatory uncertanties [type: int]
  --use_torch_k=<torch_k>                           uses K estimated in torch as opposed to the STIR implementation [default: 0] [type: int]
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
pet.AcquisitionData.set_storage_scheme("memory")
# SIRF STIR message redirector
import sirf
import sirf.STIR as pet
msg = sirf.STIR.MessageRedirector(info=None, warn=None, errr=None)
from artcertainty.kernel.LHK import kernelise_image as K, set_KOSMAPOSL
from artcertainty.kernel.LHK import BuildK
from artcertainty.utils.torch_operations import tdivide, save_as, update_subset_model
from artcertainty.utils.sirf_torch import primal_op as FP
from artcertainty.utils.sirf_torch import dual_op as BP
from artcertainty.utils.system import create_working_dir_and_move_into
from artcertainty.utils.sirf_modelling import get_acquisition_model
from artcertainty.algorithms.Algorithm import Algorithm
from artcertainty.algorithms.Algorithm import read_simulation, get_working_dir_from_outpath
from artcertainty.utils.plots import plot_many_tensors
from artcertainty.utils.system import check_reserved_memory, check_pytorch_gpu, clear_pytorch_cache

import numpy as np
import time
import torch
import os
from tqdm.auto import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HKEM(Algorithm):
    def __init__(self,iter, num_subsets,isHybrid, is2d, w, 
                 sigma_m, sigma_p, sigma_dm, frozen, anat, sinogram_template,
                 umap, norm, 
                 additive, format, save_every, is_real, psf_fwhm, 
                 epoch_checkpoint, seed, a_seed, torch_k):
        
        super().__init__(num_subsets, is2d, anat, sinogram_template, norm, additive,
                         umap, format, save_every,is_real, psf_fwhm, iter,
                         epoch_checkpoint, a_seed)
        
        self.iter = iter
        self.isHybrid = isHybrid
        self.is2d = is2d
        self.w = w
        self.frozen=frozen
        self.sigma_m = sigma_m

        self.sigma_dm = sigma_dm
        self.sigma_p = sigma_p
        self.anat = anat
        self.sinogram_template = sinogram_template
        self.seed =seed
        self.torch_k = torch_k

        # the following is needed because we do comparison with integers
        if (self.e_cpoint is None):
            self.e_cpoint = 0   

        self.init_alpha_tensor = torch.ones(size=self.anatomical_tensor.shape)

        # set up acquisition model 
        if (not self.torch_k):
            # set up acquisition model 
            self.obj_fun = pet.make_Poisson_loglikelihood(self.sinogram_template)
            self.obj_fun.set_acquisition_model(self.acq_model)

            self.kosmaposl=pet.KOSMAPOSLReconstructor()
            set_KOSMAPOSL(self.kosmaposl, self.obj_fun, self.anat, self.init_alpha_tensor,
                        self.isHybrid, self.w, self.sigma_m, self.sigma_p, self.is2d, self.sigma_dm)
        else:
            
            self.BK=BuildK(self.sigma_m,True,True,
                            [umap.spacing[0], umap.spacing[1], umap.spacing[2]],
                            sigma_p=self.sigma_p, sigma_dm=self.sigma_dm,
                            isHybrid=self.isHybrid)
            

    # def get_hybrid_k(self, alpha):
    #     if not self.is2d:    
    #         alpha=alpha[0,...]
        
    #     if(self.isHybrid):
    #         with torch.no_grad():
    #             BKp(alpha,np.power(self.w,-self.is2d+3),self.w)
    #             khw = BKp.Kw.to(device)
    #             BKp.Kw=self.BK.Kw*khw   
    def get_hybrid_k(self, alpha):
        """
        Compute hybrid kernel by combining anatomical kernel (BK) and functional kernel (BKp).
        Ensures BKp.Kw is replaced with the hybrid Kw and not recomputed later.
        """
        if not self.is2d:
            alpha = alpha[0, ...]

        if self.isHybrid:
            with torch.no_grad():
                # Compute BKp.Kw first (functional kernel)if(self.isHybrid):
                
                BKp = BuildK(self.sigma_m, True, True,
                            [self.anat.spacing[0], self.anat.spacing[1], self.anat.spacing[2]],
                            sigma_p=self.sigma_p, sigma_dm=self.sigma_dm, isHybrid=True)
                
                BKp.forward(self.anatomical_tensor, np.pow(self.w, -self.is2d + 3), self.w, functional_input=alpha)

            return BKp
                    
        
    def HKEM_iteration(self, frozen_alpha, alpha, K, s):
        if (self.num_subsets==1):
            ys=self.data_tensor
            fs=FP(self.image_template,self.sinogram_template,self.acq_model).to(device).requires_grad_(False)
            bs=BP(self.image_template,self.sinogram_template,self.acq_model).to(device)
            sens_s = bs(torch.ones(ys.shape)).to(device)   
        else:
            if (self.is_real):

                normacf_tt = torch.from_numpy(self.normacf_np).repeat(1,1,1,1,1).to(device)
                fs, bs, sens_s, ys, ads= update_subset_model(FP,BP,self.data_tensor,self.num_subsets,s,
                                                            self.umap,self.sinogram_template,
                                                            self.image_template, self.unnorm_add_tt,
                                                            normacf_tt)
            else:
                fs, bs, sens_s, ys= update_subset_model(FP,BP,self.data_tensor,self.num_subsets,s,
                                                            self.umap,self.sinogram_template,
                                                            self.image_template) 
        
        
        if (self.torch_k): 
            BKp=self.get_hybrid_k(frozen_alpha)  
            ksens=BKp.kernelise_image_save_mem_t(sens_s) #BK.kernelise_image(Kh.t(),sens_s)
            ka=BKp.kernelise_image_save_mem(alpha)#BK.kernelise_image(Kh,alpha)
            fka=fs(ka).to(device)
            if (self.is_real):
                fka = fka+ads.to(device)
            grad=bs((tdivide(ys,fka)))
            kgrad=BKp.kernelise_image_save_mem_t(grad) #BK.kernelise_image(Kh.t(),grad)

        else:              
            ksens=K.apply(sens_s, frozen_alpha,self.image_template, self.kosmaposl)
            ka= K.apply(alpha,frozen_alpha, self.image_template, self.kosmaposl)
            fka=fs(ka).to(device)
            if (self.is_real):
                fka = fka+ads.to(device)
            grad=bs((tdivide(ys,fka)))
            kgrad=K.apply(grad,frozen_alpha, self.image_template, self.kosmaposl)   

        return tdivide(alpha,ksens)*kgrad 

    def run(self, current_alpha=None):
        torch.manual_seed(self.seed) 

        if self.psf_fwhm>0:
            psf='_PSF'+str(self.psf_fwhm)  
        else:
            psf=''


        if self.torch_k:
            torchk='_Kt'+str(self.torch_k)  
        else:
            torchk=''

        if self.a_seed is None:
            a_seed_str = ''
        else:
            a_seed_str = '_aseed'+str(self.a_seed)
        
        if current_alpha is None:
        
            if self.is2d:
                alpha = torch.ones(self.anatomical_tensor.shape).repeat(1,1,1,1).to(device)
            else:
                alpha = torch.ones(self.anatomical_tensor.shape).repeat(1,1,1,1,1).to(device)
        else:
            alpha = current_alpha

        frozen_alpha = alpha
        if (self.torch_k):            
            self.BK.forward(self.anatomical_tensor, np.power(self.w, -self.is2d + 3), self.w, functional_input=frozen_alpha)
        
        PoissonLoglikelihood = {}
        PoissonLoglikelihood["loss"] = [ ]
        PoissonLoglikelihood["epoch"] = [ ]

        out_filename_pref = '2d'+str(self.is2d)+torchk+psf+'_H'+str(self.isHybrid)+'_N'+str(self.w)+'_M'+str(self.sigma_m)+'_P'+str(self.sigma_p)+'_seed'+str(self.seed)+a_seed_str+'_it'
        out_filename_pref = out_filename_pref.replace('.','_')

        if (self.e_cpoint>0):
            PoissonLoglikelihood = np.load(out_filename_pref.replace('.','_')+'_data_log.npy', allow_pickle=True).item()
            alpha = torch.load(out_filename_pref+'_alpha.torch').to(self.data_tensor.device)
            frozen_alpha = torch.load(out_filename_pref+'_frozen_alpha.torch').to(self.data_tensor.device)
            if (self.torch_k): 
                BKp=self.get_hybrid_k(frozen_alpha)  
                l=BKp.kernelise_image_save_mem(alpha)
            else:
                l=K.apply(alpha, frozen_alpha, self.image_template, self.kosmaposl)

        # print(self.e_cpoint,self.epochs+self.e_cpoint)
        for it in tqdm(range(self.e_cpoint,self.epochs+self.e_cpoint),position=0, initial=self.e_cpoint):
            for s in range(self.num_subsets):
                with torch.no_grad():
                    if self.torch_k:    
                        ati=self.HKEM_iteration(frozen_alpha, alpha, K, s) 
                    else:
                        ati=self.HKEM_iteration(frozen_alpha, alpha, K, s)
                    # plot_many_tensors(min=0,max=200,list_images=[ frozen_alpha,ati], list_names=[ 'frozen alpha','current alpha'], slice=0)
 
                    
                    ati=torch.nan_to_num(ati, nan=0.0, posinf=0.0, neginf=0.0)          
                    alpha=ati.to(alpha.device)
                    if (it < self.frozen):
                        frozen_alpha=alpha
                        torch.save(frozen_alpha,out_filename_pref+'_frozen_alpha.torch')
                    # plot_many_tensors(min=0,max=200,list_images=[ ati, alpha], list_names=[ 'current alpha','alpha_nonan'], slice=0)
 
            if (self.torch_k): 
                # print ("last kernelis")
                BKp=self.get_hybrid_k(frozen_alpha)  
                l=BKp.kernelise_image_save_mem(alpha)
            else:
                l = K.apply(alpha, frozen_alpha, self.image_template, self.kosmaposl)
            
            torch.save(alpha,out_filename_pref+'_alpha.torch')

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
            PoissonLoglikelihood["epoch"].append(it)
            np.save(out_filename_pref+'_data_log', PoissonLoglikelihood )
            if ((it+1) % self.save_every == 0):
                save_as(self.format, l, self.image_template,out_filename_pref+str(it+1))
        return l, alpha

def main():
    from type_docopt import docopt
    args = docopt(__doc__, version='0.1.0')
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
    epoch_checkpoint  = args['--epoch_checkpoint']
    
    w = args['--window']
    sigma_p = args['--sigma_p']   
    sigma_m = args['--sigma_m']   
      
    sigma_dm = args['--sigma_dm']     
    iter = args['--iter']   
    frozen = args['--frozen']
    if frozen is None:
        frozen=iter

    isHybrid = args['--isHybrid'] 
    save_every = args['--save_every']
    num_subsets = args['--num_subsets']
    format = args['--format']
    seed = args['--seed']
    torch_k = args['--use_torch_k']
    # parameters 
    LR = 1e-3
    is_real=(not sino_file == None)

    # Start timer
    start_time = time.time()
    algorithm =HKEM(iter, num_subsets, isHybrid, is2d,
                    w,sigma_m,sigma_p,sigma_dm, frozen, anat, sinogram_template,
                    umap, norm, additive, format, save_every, 
                    is_real, psf_fwhm, epoch_checkpoint, seed, a_seed, torch_k)

    l=algorithm.run()
    
    # End timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

if __name__ == '__main__':
    main()