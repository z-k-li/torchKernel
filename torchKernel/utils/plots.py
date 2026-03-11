"""
Usefull function for visualisation

Author: Daniel Deidda

Copyright 2024 National Physical Laboratory.

SPDX-License-Identifier: Apache-2.0
"""

# Import standard extra packages
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import torch
from tqdm.notebook import trange, tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def plot_many_numpys(min, max, slice=0,  colour='jet', list_images=[], list_names=[''],figure_name=""):
    
    cmap = plt.get_cmap(colour, 1000)

    fig, axes = plt.subplots(1,len(list_images),figsize=(len(list_images)*5,4))
    if len(list_images)!= len(list_names):
        print("ERROR: the number of names must be the same as the number of numpy arrays")
        exit(1)

    if (len(list_images)==1):
        im = axes.imshow(list_images.pop()[slice,...],
                    interpolation ='nearest', cmap=colour)
        im.set_clim(vmin = min, vmax = max) 
        # Normalizer 
        norm = mpl.colors.Normalize(vmin=min, vmax=max) 
        # creating ScalarMappable 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
        sm.set_array([]) 
        fig.colorbar(sm,ax=axes)

        axes.set_title(list_names.pop())
        axes.set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(len(list_images)):

        im = axes[i].imshow(list_images[i][slice,...],
                    interpolation ='nearest', cmap=colour)
        im.set_clim(vmin = min, vmax = max) 
        # Normalizer 
        norm = mpl.colors.Normalize(vmin=min, vmax=max) 
        # creating ScalarMappable 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
        sm.set_array([]) 
        fig.colorbar(sm,ax=axes[i])

        axes[i].set_title(list_names[i])
        axes[i].set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(figure_name+".png")

def plot_many_numpys_multislice(min, max, slice=[],  colour='jet', list_images=[], list_names=[''],figure_name=""):
    #is the same as above but accepts a tuple of slices instead of sigle slice. So you can decide for every image which one to show
    cmap = plt.get_cmap(colour, 1000)

    fig, axes = plt.subplots(1,len(list_images),figsize=(len(list_images)*5,4))
    if len(list_images)!= len(list_names) & len(list_images)!=len(slice):
        print("ERROR: the number of names must be the same as the number of numpy arrays and the slices to show")
        exit(1)

    if (len(list_images)==1):
        im = axes.imshow(list_images.pop()[slice[0],...],
                    interpolation ='nearest', cmap=colour)
        im.set_clim(vmin = min, vmax = max) 
        # Normalizer 
        norm = mpl.colors.Normalize(vmin=min, vmax=max) 
        # creating ScalarMappable 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
        sm.set_array([]) 
        fig.colorbar(sm,ax=axes)

        axes.set_title(list_names.pop())
        axes.set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(len(list_images)):

        im = axes[i].imshow(list_images[i][slice[i],...],
                    interpolation ='nearest', cmap=colour)
        im.set_clim(vmin = min, vmax = max) 
        # Normalizer 
        norm = mpl.colors.Normalize(vmin=min, vmax=max) 
        # creating ScalarMappable 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
        sm.set_array([]) 
        fig.colorbar(sm,ax=axes[i])

        axes[i].set_title(list_names[i])
        axes[i].set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(figure_name+".png")

def plot_many_tensors(min=None, max=None, slice=0,  colour='jet', list_images=[], list_names=[''],figure_name=""):
    
    cmap = plt.get_cmap(colour, 1000)

    fig, axes = plt.subplots(1,len(list_images),figsize=(len(list_images)*5,4))
    if len(list_images)!= len(list_names):
        print("ERROR: the number of names must be the same as the number of numpy arrays")
        exit(1)

    if (len(list_images)==1):
        if (list_images[0].dim()==2):
            im = axes.imshow(list_images.pop().detach().cpu().numpy(),
                        interpolation ='nearest', cmap=colour)
        elif (list_images[0].dim()==3):
            im = axes.imshow(list_images.pop()[slice,...].detach().cpu().numpy(),
                            interpolation ='nearest', cmap=colour)
        elif (list_images[0].dim()==4):
            im = axes.imshow(list_images.pop()[0,slice,...].detach().cpu().numpy(),
                            interpolation ='nearest', cmap=colour)
        elif (list_images[0].dim()==5):
            im = axes.imshow(list_images.pop()[0,0,slice,...].detach().cpu().numpy(),
                            interpolation ='nearest', cmap=colour)
        else:
            print("ERROR: the dimension of the tensor must be 2, 3, 4 or 5 no other dimension supported")
            exit(1)

        im.set_clim(vmin = min, vmax = max) 
        # Normalizer 
        if(min==None and max==None):
            norm = mpl.colors.Normalize(vmin=list_images.pop().min(), vmax=list_images.pop().max())
        else: 
            norm = mpl.colors.Normalize(vmin=min, vmax=max) 
        # creating ScalarMappable 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
        sm.set_array([]) 
        fig.colorbar(sm,ax=axes)

        axes.set_title(list_names.pop())
        axes.set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        for i in range(len(list_images)):
            if (list_images[i].dim()==2):
                im = axes[i].imshow(list_images[i].detach().cpu().numpy(),
                        interpolation ='nearest', cmap=colour)
            elif (list_images[i].dim()==3):
                im = axes[i].imshow(list_images[i][slice,...].detach().cpu().numpy(),
                        interpolation ='nearest', cmap=colour)
            elif (list_images[i].dim()==4):
                im = axes[i].imshow(list_images[i][0,slice,...].detach().cpu().numpy(),
                        interpolation ='nearest', cmap=colour)
            elif (list_images[i].dim()==5):
                im = axes[i].imshow(list_images[i][0,0,slice,...].detach().cpu().numpy(),
                        interpolation ='nearest', cmap=colour)
            else:
                print("ERROR: the dimension of the tensor must be 2, 3, 4 or 5 no other dimension supported")
                exit(1)

            im.set_clim(vmin = min, vmax = max) 
            # Normalizer 
            if(min==None and max==None):
                norm = mpl.colors.Normalize(vmin=list_images[i].min(), vmax=list_images[i].max())
            else: 
                norm = mpl.colors.Normalize(vmin=min, vmax=max) 
            # creating ScalarMappable 
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
            sm.set_array([]) 
            fig.colorbar(sm,ax=axes[i])

            axes[i].set_title(list_names[i])
            axes[i].set_axis_off()
            plt.subplots_adjust(wspace=0, hspace=0)
    
    fig.savefig(figure_name+".png")
