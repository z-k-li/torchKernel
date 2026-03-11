'''
script running the Deep image prior with optimisation transfer
this is an implementation of the following manuscript:

S. Li, K. Gong, R. D. Badawi, E. J. Kim, J. Qi and G. Wang,
"Neural KEM: A Kernel Method With Deep Coefficient Prior for PET Image Reconstruction,
" in IEEE Transactions on Medical Imaging, vol. 42, no. 3, pp. 785-796, March 2023, doi: 10.1109/TMI.2022.3217543

Usage:
  OTDIP [--help | options]

Options:
  -s <file>, --sino=<file>                          raw data file. All the file paths needs to be relative to out_path/working_dir
  -f <format>, --format=<format>                    format to save into 'template', 'npy', 'nii' [default: npy]
  -a <anatomic_image> --anatomic=<anatomic_image>   interfile image of the image you want to use as a prior
  -u <umap_image> --umap=<umap_image>               interfile image of the umap image for attenuation
  --add_sino=<addittive_sino>                       interfile data with randoms and scatter
  --n_sino=<norm_sino>                              interfile data of the normalisation
  -o <out_path>, --outpath=<out_path> 
  --is2d=<is2d>                                     for 2d or 3d data [default: 1] [type: int]
  -i <out_it>, --out_it=<out_it>                    number of outer iteration [default: 60] [type: int]
  --save_every=<save_every>                         save image every <save_every> iterations [default: 1] [type: int]
  --psf_fwhm=<psf_fwhm>                             full width half maximum for the point spread function (PSF) 
                                                    [default: 0] [type: float]
  -e <epoch_checkpoint>, --epoch_checkpoint=<epoch_checkpoint>    start from trained model at epoch_checkpoint
                                                    [default 0] [type: int]
  --seed=<seed>                                     seed of the random process for reproducibility [default: 42] [type: int]
  --a_seed=<a_seed>                                 seed of the random process for aleatory uncertanties [type: int]
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
pet.AcquisitionData.set_storage_scheme("file")
# SIRF STIR message redirector
import sirf
import sirf.STIR as pet
msg = sirf.STIR.MessageRedirector(info=None, warn=None, errr=None)
from artcertainty.architectures.UNet import  UNet
from artcertainty.architectures.UNet3D import  UNet as UNet3d
from artcertainty.utils.torch_operations import tdivide, save_as, make_cylindrical_mask_tensor
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

class OTDIP(Algorithm):

    def __init__(self,outer_iter,em_iter,deep_iter, learning_rate, 
                 is2d, anat, sinogram_template, umap, norm, 
                 additive, format, save_every, is_real, psf_fwhm, 
                 epoch_checkpoint, a_seed=None):
                 
        super().__init__(1, is2d, anat, sinogram_template, norm, additive,
                         umap, format, save_every,is_real, psf_fwhm, outer_iter,
                         epoch_checkpoint, a_seed)
        
        self.outer_iter = outer_iter
        self.em_iter = em_iter
        self.deep_iter = deep_iter
        self.LR = learning_rate
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
        

        if self.is2d:
            net = UNet(1,16,1)
        else:
            net = UNet3d(1,16,1)

        net = net.to(device)
        net_in = self.anatomical_tensor.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor) 
        print("is the network on cuda?  "+str(next(net.parameters()).is_cuda))

        data_log = { }
        data_log["loss"] = [ ]
        data_log['epoch'] = [ ]

        p = []
        p += [x for x in net.parameters() ]
        optimiser = torch.optim.Adam(p, lr=self.LR)
        Poisson_loss = torch.nn.PoissonNLLLoss(log_input=False,reduction='none')
        curr=torch.ones(self.anatomical_tensor.shape).to(device)
        net_out=torch.zeros(self.anatomical_tensor.shape).to(device)
        scale=1

        curr_masked = torch.zeros(self.anatomical_tensor.shape).to(device)
        mask_im = make_cylindrical_mask_tensor(curr)
        mask = (mask_im==1)

        out_filename_pref = 'OTDIP_2d'+str(self.is2d)+psf+'_'+'seed'+str(seed)+a_seed_str+'_dit'+str(self.deep_iter)+'_eit'+str(self.em_iter)+'_tit'
        out_filename_pref = out_filename_pref.replace('.','_')

        if (self.e_cpoint>0):
            curr, data_log, net, optimiser = self.read_checkpoint(out_filename_pref, net, optimiser)
            net=net.to(self.data_tensor.device)

        for i in tqdm(range(self.e_cpoint,self.epochs+self.e_cpoint),position=0, initial=self.e_cpoint):
            with torch.no_grad():   
                curr_masked[mask]=curr[mask].to(device)
                for it in range(self.em_iter):
                        ys = self.data_tensor
                        sens = self.bp(torch.ones(ys.shape)).to(device) 
                        fa=self.fp(curr_masked).to(device)
                        if (self.is_real):
                            fa = fa+self.unnorm_add_tt.to(device)
                        grad=self.bp((tdivide(ys,fa))) 
                        curr_em_i = tdivide(curr_masked,sens)*grad.to(device) 

                        curr_em_i=torch.nan_to_num(curr_em_i, nan=0.0, posinf=0.0, neginf=0.0) 
            
                # if i==0:                    
                #         scale=curr_em_i.max()

            def closure():                                
                net_out = scale*net(net_in).to(device)
                optimiser.zero_grad()
                #loss=mse_loss( net_out, curr_em_i)
                tot_loss = torch.mean(sens * Poisson_loss(net_out, curr_em_i))
                net.zero_grad()
                tot_loss.backward()
                data_log["loss"].append(tot_loss.item())
                data_log['epoch'].append(len(data_log["loss"]))   
                return tot_loss

            for j in range(self.deep_iter):                
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
                optimiser.step(closure)      
                with torch.no_grad():
                    net_out = scale*net(net_in).to(device)     
   
            curr=net_out


            #save things       
            torch.save({'model_state_dict_net': net.state_dict(),
                       'optimiser_state_dict': optimiser.state_dict(),
                        }, out_filename_pref+'_trained_extra.torch_model')
            save_as(self.format, net_out, self.image_template,out_filename_pref+str(i))
            np.save(out_filename_pref+'_data_log',data_log)
        return net_out

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
    seed = args['--seed']
    epoch_checkpoint  = args['--epoch_checkpoint']
   
    num_outer_iter = args['--out_it']   
    save_every = args['--save_every']
    format = args['--format']

    num_deep_iter=150 #150
    num_em_iter=4
    LR = 1e-3
    is_real=(not sino_file == None)

    # Start timer
    start_time = time.time()

    algorithm = OTDIP(num_outer_iter, num_em_iter, num_deep_iter,LR,
                      is2d, anat, sinogram_template, umap, norm, additive, format, 
                      save_every, is_real, psf_fwhm, epoch_checkpoint, a_seed)

    algorithm.run(seed)

    # End timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

if __name__ == '__main__':
    main()