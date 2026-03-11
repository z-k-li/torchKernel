#Usefull function for visualisation

#Author: Daniel Deidda

#Copyright 2024 National Physical Laboratory.

#SPDX-License-Identifier: Apache-2.0

# Import standard extra packages

import torch
import sirf.STIR as pet
from .sirf_modelling import get_acquisition_model, get_acquisition_model_real_with_norm_and_umap, get_acquisition_model_with_normacf
import numpy as np
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_cylindrical_mask(reference):
    # make a mask based on the reference FoV
    offset_xy=reference.shape[reference.ndim()-1]/2
    mask = np.zeros(reference.shape)

    for i in range(reference.shape[reference.ndim()-3]):
        for j in range(reference.shape[reference.ndim()-2]):
            for k in range(reference.shape[reference.ndim()-1]):
                if np.sqrt(np.square(j-offset_xy)+ np.square(k-offset_xy))<reference.shape[reference.ndim()-1]/2:
                    if(reference.ndim()==3):
                        mask[i,j,k]=1
                    elif(reference.ndim()==4):
                        mask[:,i,j,k]=1
                    elif(reference.dim()==5):
                        mask[:,:,i,j,k]=1


    return mask

def make_cylindrical_mask_tensor(reference):
    # make a mask based on the reference FoV
    offset_xy=reference.shape[reference.dim()-1]/2

    mask = torch.zeros(reference.shape)
    # if (reference.dim()<3):
    #     for i in range(reference.shape[reference.dim()-3]):
    #         for j in range(reference.shape[1]):
    #                 if np.sqrt(np.square(i-offset_xy)+ np.square(j-offset_xy))<reference.shape[1]/2:
    #                     mask[i,j]=1
    # else:
    for i in range(reference.shape[reference.dim()-3]):
        for j in range(reference.shape[reference.dim()-2]):
            for k in range(reference.shape[reference.dim()-1]):
                if np.sqrt(np.square(j-offset_xy)+ np.square(k-offset_xy))<reference.shape[reference.dim()-1]/2:
                    if(reference.dim()==3):
                        mask[i,j,k]=1
                    elif(reference.dim()==4):
                        mask[:,i,j,k]=1
                    elif(reference.dim()==5):
                        mask[:,:,i,j,k]=1

    return mask

#tensor division without zero
def tdivide(a,b):
    # mask = (b != 0)
    # c = torch.zeros(a.shape).to(device)
    # c[mask] = a[mask]/b[mask]
    c = a/b
    return torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0) 

#numpy division without zero
def npdivide(a,b):
    # mask = (b != 0)
    # c = np.zeros(a.shape)
    # c[mask] = a[mask]/b[mask]
    c = a/b
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0) 

# threshold tensor between min and max and set to 1
def treshold_tensor(t,min, max):
    mask = (t>=min) & (t<=max)
    c = torch.zeros(t.shape).to(t.device)
    ones = torch.zeros(t.shape).to(t.device)
    c[mask] = t[mask]
    c[mask] = 1

    return c
#add poisson noise can create more 'intense' noise by giving a noise factor>1
def add_noise(y,seed, noise_factor = 1):
    
    # Data should be >=0 anyway, but add abs just to be safe
    y = torch.abs(y*noise_factor)
    # for i in range(noise_factor):
    y =torch.poisson(y, seed).to(y.device);
    return y

def update_subset_model(F,B,sino,num_subsets,subset_num,umap,full_template,image_template, addittive=None, norm_acf=None):

    subset_lenght = int(sino.shape[sino.dim()-2]/num_subsets)

    if (sino.dim()==4):#2d
        sub_sino = sino[:,:,subset_num*subset_lenght:subset_num*subset_lenght+subset_lenght,:]
    elif (sino.dim()==5):#3d
        sub_sino = sino[:,:,:,subset_num*subset_lenght:subset_num*subset_lenght+subset_lenght,:]
    else:
        sys.exit('Error: the dimenson of the tensor needs to be 4 for 2D or 5 for 3D')

    if(not addittive==None):
        if (sino.dim()==4):#2d
            sub_add = addittive[:,:,subset_num*subset_lenght:subset_num*subset_lenght+subset_lenght,:]
        elif (sino.dim()==5):#3d
            sub_add = addittive[:,:,:,subset_num*subset_lenght:subset_num*subset_lenght+subset_lenght,:]
        else:
            sys.exit('Error: the dimenson of the additive tensor needs to be 4 for 2D or 5 for 3D')

    if(not norm_acf==None):
        if (sino.dim()==4):#2d
            sub_nacf = norm_acf[:,:,subset_num*subset_lenght:subset_num*subset_lenght+subset_lenght,:]
        elif (sino.dim()==5):#3d
            sub_nacf = norm_acf[:,:,:,subset_num*subset_lenght:subset_num*subset_lenght+subset_lenght,:]
        else:
            sys.exit('Error: the dimenson of the additive tensor needs to be 4 for 2D or 5 for 3D')

    temp=full_template.get_subset(range(subset_num*subset_lenght, subset_num*subset_lenght+subset_lenght))
   
    if(not norm_acf==None):
        if (sino.dim()==4):#2d
            acq_model_s, inv_norm_acf_sino = get_acquisition_model_with_normacf(temp, image_template, temp.copy().fill(sub_nacf.detach().cpu().numpy()))
        elif (sino.dim()==5):#3d
            acq_model_s, inv_norm_acf_sino = get_acquisition_model_with_normacf(temp, image_template, temp.copy().fill(sub_nacf[0,...].detach().cpu().numpy()))
    else:
        acq_model_s = get_acquisition_model(temp, umap)   

    acq_model_s.set_up(temp, umap)
    f = F(image_template,temp,acq_model_s).to(device)
    b = B(image_template,temp,acq_model_s).to(device) 

    if (sino.dim()==4):#2d       
        sub_sens = b(torch.ones(sub_sino.shape)).to(device) 
    elif (sino.dim()==5):#3d
        sub_sens = b(torch.ones(sub_sino.shape)).repeat(1,1,1,1,1).to(device) 

    if(not addittive==None):
        return f,b, sub_sens, sub_sino, sub_add
    else:
        return f,b, sub_sens, sub_sino

def save_as(format, tensor, template, filename):
    
    if (format == 'template'):
        save_as_template(tensor, template, filename)
        
    elif (format == 'npy'):
        np.save(filename+'.npy',tensor.detach().cpu().numpy())

    elif (format == 'nii'):
        tnp=tensor[0,:,:,:].detach().cpu().numpy()
        template.fill(tnp)
        parfile=pet.get_STIR_examples_dir()+'/samples/stir_math_ITK_output_file_format.par'
        template.write_par(filename,parfile)
        
def save_as_template(tensor, template, filename):
     tnp=tensor[0,:,:,:].detach().cpu().numpy()
     template.fill(tnp)
     template.write(filename)

def save_npy_as(format, npy, template, filename):
    
    if (format == 'template'):
        save_npy_as_template(npy, template, filename)
        
    elif (format == 'npy'):
        np.save(filename+'.npy',npy)

    elif (format == 'nii'):
        template.fill(npy)
        parfile=pet.get_STIR_examples_dir()+'/samples/stir_math_ITK_output_file_format.par'
        template.write_par(filename,parfile)
        
def save_npy_as_template(npy, template, filename):
     template.fill(npy)
     template.write(filename)