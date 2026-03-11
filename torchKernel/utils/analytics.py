"""
Usefull function for analytics

Author: Daniel Deidda

Copyright 2024 National Physical Laboratory.

SPDX-License-Identifier: Apache-2.0

"""
import numpy as np
import torch
import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt
import matplotlib as mpl
from .plots import plot_many_tensors
from .torch_operations import make_cylindrical_mask_tensor
import sirf.STIR as stir
import os.path
import sys
from ignite.metrics import *
from ignite.engine import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from scipy import stats

def check_stat_distributions(data,column, ci, distr_to_show=None, show_plot=None):
    """
      function that takes in input a dataframe at a specific column and checks whether the statistical
      distribution is norm or a selection of others, 'expon', 'lognorm', 'gamma', 'beta', 'weibull_min', 'weibull_max'
      you can choose to show a plot with the fi. ther returned values are the distribution parameter after the fit and
      the name of the distribution. Furthermore, a Kolmogorov-Smirnov Test is performed at a slected confidence interval.
      If distr_to_show=None the function perform the analysis on the one with highest p-value.
    """
    # List of distributions to test
    distributions = ['norm', 'expon', 'lognorm', 'gamma', 'beta', 'weibull_min', 'weibull_max','uniform']

    # Dictionary to store results
    results = {}
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        params = dist.fit(data[column])
        ks_stat, p_value = stats.kstest(data[column], dist_name, args=params)
        results[dist_name] = {'dist':dist_name, 'KS Statistic': ks_stat, 'p-value': p_value}

    
    # Convert results to DataFrame for display
    results_df = pd.DataFrame(results).T.sort_values(by='p-value', ascending=False)
    # print("Kolmogorov-Smirnov Test Results:")
    # print(results_df)
    
    def results(distr_to_show):
        best_dist = getattr(stats, distr_to_show)
        best_params = best_dist.fit(data[column])

        x = np.linspace(min(data[column]), max(data[column]), 100)
        pdf = best_dist.pdf(x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
   
        if (show_plot is not None):
            plt.hist(data[column], bins=10, density=True, alpha=0.5, label='Data')
            plt.plot(x, pdf, label=f'{distr_to_show} fit', color='red')
            plt.legend()
            plt.title('Distribution Fit')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.show()

        norm = results_df[results_df['dist'] == distr_to_show]
        print(f"Kolmogorov-Smirnov Test Statistic: {norm['KS Statistic']}")
        print(f"p-value: {norm['p-value']}")

        # Interpretation
        if float(norm['p-value']) > ci:
            print("The data appears to be distributed as, "+distr_to_show+" (fail to reject H0).")
        else:
            print("The data does not appear to be distributed as, "+distr_to_show+"  (reject H0).")

        return best_params, best_dist

    # Optional: Plot histogram and best-fitting distribution
    best_fit = results_df.index[0]
    if distr_to_show is None:
        best_params, best_dist = results(best_fit)
    else:
        best_params, best_dist = results(distr_to_show)
    return best_params, best_dist




def show_all_seed_images(pref1, iter, pref2, num_replicas, max, slice=0):
    for s in range(num_replicas):
            
        o = torch.from_numpy(np.load(pref1+str(s)+'_'+pref2+str(iter)+'.npy',allow_pickle=True))
        if (o.dim()==5):
            plot_many_tensors(0, max, slice,'jet', [o[0,...]], ['image iter'+str(iter)])
        else:
            plot_many_tensors(0, max, slice,'jet', [o], ['image iter'+str(iter)])

def show_image(pref1, iter, pref2, seed, max, slice=0):
                
        o = torch.from_numpy(np.load(pref1+str(seed)+'_'+pref2+str(iter)+'.npy',allow_pickle=True))
        print(o.max())
        if (o.dim()==5):
            plot_many_tensors(0, max, slice,'jet', [o[0,...]], ['image iter'+str(iter)])
        else:
            plot_many_tensors(0, max, slice,'jet', [o], ['image iter'+str(iter)])

def estimate_MSE_and_save(string1,num_iter, string2, recon_name, reference, skip_iter=0, islog10=True):
    """
    estimates the log of the mse using the true imge as reference
    it appends an mse value for every saved iteration up to num_iter
    the  name of the file needs to be something like string1+iter+string2 
    this only works if the reconstructed image is saved as numpy  (check the relative algorithm for names)
    """
        
    MSE={}
    MSE['mse']=[] 
    MSE['epoch']=[] 

    i=0
    for j in range(int(num_iter)):
            
            string=string1+str(j)+string2#_M1.0_P1.0
            if not os.path.isfile(string):
                #  print('Warning: the following file is not found. Will look for the next iteration ' + string )
                    continue
            else:         
                i=i+1
            if i==0:
                print('Warning: I hve not found the following files ' + string )

            # mask = (reference != 0)

            r = np.load(string,allow_pickle=True)
            r= torch.from_numpy(r).to(reference.device)

            if i==1:
                mask_im = make_cylindrical_mask_tensor(r)

            mask = (mask_im==1)
            
            if islog10:
                MSE['mse'].append(10*(torch.log10(torch.square(torch.sum((reference[mask]-r[mask]))/torch.sum(reference[mask])))).detach().cpu().numpy());
            else:
                MSE['mse'].append(torch.square(torch.sum((reference[mask]-r[mask]))/torch.sum(reference[mask])).detach().cpu().numpy());
            MSE['epoch'].append( j )

    np.save(('MSE_'+string1+str(num_iter)+string2),MSE)  

    min_mse = np.argmin(MSE['mse']) 
    delta=MSE["epoch"][1]
    skip_id=int(skip_iter/delta)
    print('Minimum MSE is: {} at epoch {}'.format(MSE['mse'][min_mse],MSE['epoch'][min_mse]))
    
    # if (recon_name != None):
    #     o = np.load(string1+str(MSE['epoch'][min_mse])+string2,allow_pickle=True)
    #     if (r.dim()==5):
    #         plot_many_tensors(torch.from_numpy(o[0,...]),r[0,...], 'optimum image', 'last_image',0, reference.max())
    #     else:
    #         plot_many_tensors(torch.from_numpy(o),r, 'optimum image', 'last_image',0, reference.max())  
    #     fig, (ax1) = plt.subplots(1,1,figsize=(15,5))#, ax2, ax3)
    #     ax1.plot(MSE['epoch'][skip_id:],MSE['mse'][skip_id:], label=recon_name)
    #     plt.legend( fontsize=18)
    #     ax1.set_title('(log-MSE)', fontsize=18)
    return MSE

def plot_losses(loss_names_pref, skip_iter=0, show_every=1, islog10=False):
    """
    plots the loss function saved during the reconstruction. You can plot as many losses as you like for comparison
    you can decide to skip 'skip_iter' iterations to focus on the final values
    since the loss contains all the epochs value you can decide to use 'show_every' to only show the outer iteration ex: 
    50 for ADMM
    """
    loss = []
    nologloss = []
    # nologloss['loss'] = []
    # nologloss['epoch'] = []
    nologloss_list = []
    
    for i in range(len(loss_names_pref)):
        # loss= np.load(algo_pref+ '_data_log.npy', allow_pickle=True).item()
        if islog10:
            nologloss_list.append(np.load(loss_names_pref[i]+'_data_log.npy', allow_pickle=True).item())
            nologloss.append({'loss': np.log10(nologloss_list[i]['loss']).tolist(), 'epoch': nologloss_list[i]['epoch']})
            # nologloss['epoch']=()
            loss.append((nologloss[i]))
        else:
            loss.append(np.load(loss_names_pref[i]+'_data_log.npy', allow_pickle=True).item())
            # multi_loss['loss'] = (loss['loss'])


    fig, (ax1) = plt.subplots(1,1,figsize=(15,5))#, ax2, ax3)

    for i in range(len(loss_names_pref)):
        ax1.plot(loss[i]['loss'][skip_iter::show_every], label= loss_names_pref[i].replace('_data_log.npy', ''))

    plt.legend()
    ax1.set_title('(Training log-likelihood)', fontsize=18)
    return loss

def plot_losses_old(loss_name1, loss_name2='', loss_name3='',loss_name4='', loss_name5='', skip_iter=0, show_every=1):
    """
    plots the loss function saved during the reconstruction. You can plot up to 5 losses for comparison
    you can decide to skip 'skip_iter' iterations to focus on the final values
    since the loss contains all the epochs value you can decide to use 'show_every' to only show the outer iteration ex: 
    50 for ADMM
    """
    loss1 = np.load(loss_name1, allow_pickle=True).item()

    if (loss_name2 != ''):
        loss2 = np.load(loss_name2, allow_pickle=True).item()
    if (loss_name3 != ''):
        loss3 = np.load(loss_name3, allow_pickle=True).item()
    if (loss_name4 != ''):
        loss4 = np.load(loss_name4, allow_pickle=True).item()
    if (loss_name5 != ''):
        loss5 = np.load(loss_name5, allow_pickle=True).item()

    fig, (ax1) = plt.subplots(1,1,figsize=(15,5))#, ax2, ax3)
    ax1.plot(loss1['loss'][skip_iter::show_every], label= loss_name1.replace('_data_log_dH1', ''))
    if (loss_name2 != ''):
        ax1.plot(loss2['loss'][skip_iter::show_every], label= loss_name2.replace('_data_log_dH1', ''))
    if (loss_name3 != ''):
        ax1.plot(loss3['loss'][skip_iter::show_every], label= loss_name3.replace('_data_log_dH1', ''))
    if (loss_name4 != ''):
        ax1.plot(loss4['loss'][skip_iter::show_every], label= loss_name4.replace('_data_log_dH1', ''))
    if (loss_name5 != ''):
        ax1.plot(loss5['loss'][skip_iter::show_every], label= loss_name5.replace('_data_log_dH1', ''))

    plt.legend()
    ax1.set_title('(Training log-likelihood)', fontsize=18)
    return loss1,

def plot_metrics(metric_name,analytics_list, algo_pref):

    fig, (ax1) = plt.subplots(1,1,figsize=(15,5))#, ax2, ax3)
    ax1.plot(analytics_list["epoch"],analytics_list[metric_name+'_'+str(0)], label= algo_pref.replace('_dit', ' ')+ 'ROI'+str(1))#.replace('_data_log_dH1', '')
    ax1.plot(analytics_list["epoch"],analytics_list[metric_name+'_'+str(1)], label= algo_pref.replace('_dit', ' ')+ 'ROI'+str(2))
    ax1.plot(analytics_list["epoch"],analytics_list[metric_name+'_'+str(2)], label= algo_pref.replace('_dit', ' ')+ 'ROI'+str(3))

    plt.legend()
    ax1.set_title('(Bias vs CoV)', fontsize=18)
    ax1.set_ylabel(metric_name, fontsize=18)
    ax1.set_xlabel('epoch', fontsize=18)

def create_metrics_with_gt_col_ROI_sirf(algo_pref, ROIs, num_epoch, ground_true=torch.zeros(1), seed=None):
    """
    #this takes the filename of the reconstructed image, a list of ROis, the number of epoch/iteration that we want to study
    # a list or true_values if available. The metrics are organised in columns to be easily processed with pandas
    # you can read the output of this function in two ways: out=create_metrics_with_gt_col_ROI(...) or out = pd.read_csv(algo_pref+'analytics.csv')
    """
    analytics = {}
    algo_roi_nz = torch.zeros(ROIs[0].shape).to(ROIs[0].device)
    # plot_many_tensors(0,1,7,'jet', [ROIs[0][0,...],ROIs[1][0,...],ROIs[2][0,...],ROIs[3][0,...]],['0','1','2','3'])
    # for r in range(ROIs.__len__()):
    analytics["algo"] = [ ]
    analytics["ROI"] = [ ]
    analytics["mean"] = [ ]
    analytics["SD"] = [ ]
    if ground_true.dim()>1:
        gt_roi_nz = torch.zeros(ROIs[0].shape).to(ROIs[0].device)
        analytics["bias"] = [ ]
        analytics["gt mean"] = [ ]
        analytics["gt SD"] = [ ]
        analytics["gtCoV"] = [ ]
    #analytics["mse_"+str(r)] = [ ]
    analytics["CoV"] = [ ]
    analytics["CoVb"] = [ ]
    analytics["meanb"] = [ ] #for the estimation of CNR
    analytics["SDb"] = [ ] #for the estimation of CNR
    analytics["epoch"] = [ ]
    analytics["CNR"] = [ ]
    
    if  seed!=None:
        analytics["seed"] = [ ]
    nexist=0
    index=0
    for j in range(num_epoch):
        if not os.path.isfile(algo_pref+str(j)+'.hv'):
            nexist=nexist+1
            #  print('Warning: the following file is not found. Will look for the next iteration ' + string )
            continue
        meanb=0
        sdb=0
        algo = torch.from_numpy(stir.ImageData(algo_pref+str(j)+'.hv').as_array()).to(ROIs[0].device)
        if algo.size==0:
            print('Error: the following file is not found: ' + algo )
            break
        else:
             index = index + 1
        
        for r in range(ROIs.__len__()):
            algo_roi=(algo*ROIs[r]).to(ROIs[0].device)
            mask = ROIs[r]> 0
            algo_roi_nz[mask]=(algo_roi[mask]).to(ROIs[0].device)
            
            mean = float(torch.mean(algo_roi_nz[mask], dtype=torch.float))
            sd = float(torch.std(algo_roi_nz[mask]))
            if mean==0:
                mean=0.000000000000001

            analytics["epoch"].append(j)
            analytics["algo"].append(algo_pref)
            
            analytics["CoV"].append(sd / mean) 
            analytics["mean"].append(mean)
            analytics["SD"].append(sd)
            
            if ground_true.dim()>1:
                gt_roi=(ground_true*ROIs[r])
                gt_roi_nz[mask]=(gt_roi[mask])
                gt_mean=float(torch.mean(gt_roi_nz[mask], dtype=torch.float))
                gt_sd=float(torch.std(gt_roi_nz[mask]))
                analytics["gt mean"].append(gt_mean)
                analytics["bias"].append((mean-gt_mean)/gt_mean )
                analytics["gt SD"].append(gt_sd)
                analytics["gtCoV"].append(gt_sd / gt_mean) 

            analytics["ROI"].append(r+1)
            if  seed!=None:
                analytics["seed"].append(seed)

            if (r == ROIs.__len__()-1):
                meanb=mean
                sdb=sd
                for r in range(ROIs.__len__()):
                    # if ground_true.dim()>1:
                    #     # if(sd>gt_sd):
                    #     #     analytics["CoVb"].append((sd-gt_sd) / mean) 
                    #     # else:
                    #     analytics["CoVb"].append((sd) / mean)

                    # else:
                    analytics["CoVb"].append(sd / mean) 
                    analytics["meanb"].append((meanb)) 
                    analytics["SDb"].append(sdb) 

                    if sdb==0:
                        sdb=0.00000000000000001
                    # print(analytics["mean"][r],meanb,sdb)
                    analytics["CNR"].append(( analytics["mean"][ROIs.__len__()*(index-1)+r]-meanb)/sdb) 

    # if nexist>=num_epoch:
    #     print('Error no file found: ', algo_pref)
    df=pd.DataFrame(analytics)  
    # df.to_csv(algo_pref+'analytics.csv', index=True, header=True)

    return df

def create_metrics_with_gt_col_ROI(algo_pref, ROIs, num_epoch, ground_true=torch.zeros(1), seed=None):
    """
    #this takes the filename of the reconstructed image, a list of ROis, the number of epoch/iteration that we want to study
    # a list or true_values if available. The metrics are organised in columns to be easily processed with pandas
    # you can read the output of this function in two ways: out=create_metrics_with_gt_col_ROI(...) or out = pd.read_csv(algo_pref+'analytics.csv')
    """
    analytics = {}
    algo_roi_nz = torch.zeros(ROIs[0].shape).to(ROIs[0].device)
    # plot_many_tensors(0,1,7,'jet', [ROIs[0][0,...],ROIs[1][0,...],ROIs[2][0,...],ROIs[3][0,...]],['0','1','2','3'])
    # for r in range(ROIs.__len__()):
    analytics["algo"] = [ ]
    analytics["ROI"] = [ ]
    analytics["mean"] = [ ]
    analytics["SD"] = [ ]
    if ground_true.dim()>1:
        gt_roi_nz = torch.zeros(ROIs[0].shape).to(ROIs[0].device)
        analytics["bias"] = [ ]
        analytics["gt mean"] = [ ]
        analytics["gt SD"] = [ ]
        analytics["gtCoV"] = [ ]
    #analytics["mse_"+str(r)] = [ ]
    analytics["CoV"] = [ ]
    analytics["CoVb"] = [ ]
    analytics["meanb"] = [ ] #for the estimation of CNR
    analytics["SDb"] = [ ] #for the estimation of CNR
    analytics["epoch"] = [ ]
    analytics["CNR"] = [ ]
    
    if  seed!=None:
        analytics["seed"] = [ ]
    nexist=0
    index=0
    for j in range(num_epoch):
        if not os.path.isfile(algo_pref+str(j)+'.npy'):
            nexist=nexist+1
            #  print('Warning: the following file is not found. Will look for the next iteration ' + string )
            continue
        meanb=0
        sdb=0
        algo = torch.from_numpy(np.load(algo_pref+str(j)+'.npy', allow_pickle=True)).to(ROIs[0].device)
        if algo.size==0:
            print('Error: the following file is not found: ' + algo )
            break
        else:
             index = index + 1
        
        for r in range(ROIs.__len__()):
            algo_roi=(algo*ROIs[r]).to(ROIs[0].device)
            mask = ROIs[r]> 0
            algo_roi_nz[mask]=(algo_roi[mask]).to(ROIs[0].device)
            
            mean = float(torch.mean(algo_roi_nz[mask], dtype=torch.float))
            sd = float(torch.std(algo_roi_nz[mask]))
            if mean==0:
                mean=0.000000000000001

            analytics["epoch"].append(j)
            analytics["algo"].append(algo_pref)
            
            analytics["CoV"].append(sd / mean) 
            analytics["mean"].append(mean)
            analytics["SD"].append(sd)
            
            if ground_true.dim()>1:
                gt_roi=(ground_true*ROIs[r])
                gt_roi_nz[mask]=(gt_roi[mask])
                gt_mean=float(torch.mean(gt_roi_nz[mask], dtype=torch.float))
                gt_sd=float(torch.std(gt_roi_nz[mask]))
                analytics["gt mean"].append(gt_mean)
                analytics["bias"].append((mean-gt_mean)/gt_mean )
                analytics["gt SD"].append(gt_sd)
                analytics["gtCoV"].append(gt_sd / gt_mean) 

            analytics["ROI"].append(r+1)
            if  seed!=None:
                analytics["seed"].append(seed)

            if (r == ROIs.__len__()-1):
                meanb=mean
                sdb=sd
                for r in range(ROIs.__len__()):
                    # if ground_true.dim()>1:
                    #     # if(sd>gt_sd):
                    #     #     analytics["CoVb"].append((sd-gt_sd) / mean) 
                    #     # else:
                    #     analytics["CoVb"].append((sd) / mean)

                    # else:
                    analytics["CoVb"].append(sd / mean) 
                    analytics["meanb"].append((meanb)) 
                    analytics["SDb"].append(sdb) 

                    if sdb==0:
                        sdb=0.00000000000000001
                    # print(analytics["mean"][r],meanb,sdb)
                    analytics["CNR"].append(( analytics["mean"][ROIs.__len__()*(index-1)+r]-meanb)/sdb) 

    if nexist>=num_epoch:
        print('Error no file found: ', algo_pref)
    df=pd.DataFrame(analytics)  
    df.to_csv(algo_pref+'analytics.csv', index=True, header=True)

    return df

def concatenate_csv(df1_name, df2_name ):
    """
    this takes  filenames in input read into csv file and concatanate them. it returns a dataframe
    """
    df1 = pd.read_csv(df1_name)
    df2 = pd.read_csv(df2_name)
    return pd.concat([df1, df2],axis=0)

def seaborn_plot(df, x, y,  colour, col=None, style=None, height=5, aspect=0.8, kind='line',  
                 every_n_epoch=None, start_epoch=None, end_epoch=None, palette='rainbow'):
    """
    # takes a dataframe in input, decide which column to plot in x and y. 
    # colour: assigns color according to that column ex different colours for different algorithms
    # kind='line' could be 'scatter' or others check seaborn documentation for relplot
    # col: divide the plot is in sublots according to that column ex: one per ROI
    # you can select a range of iteration to plot  like (start_epoch end_epoch) and also to skip every 'every_n_epoch' iterations

    """
    #df = df[(df[x] < 0.5)]
    if (end_epoch!= None):
        df = df[(df["epoch"] <end_epoch)]
    if (start_epoch!= None):
        df = df[(df["epoch"]>start_epoch)]
    if (every_n_epoch!= None):
        df = df[(df['epoch'] % every_n_epoch) == 0]

    #sb.relplot(df, x=x, y=y, hue=colour)
    sb.relplot(df, x=x, y=y, hue=colour, kind=kind, col=col, palette=palette, style=style, height=height, aspect=aspect)#
    #sb.lineplot(data=data, x="epoch", y="bias",hue="ROI")
    #sb.set_style("ticks")
    plt.show()

    

def compare_different_algorithms_metrics(x, y, algo1, algo2, algo3=pd.DataFrame(), algo4=pd.DataFrame(),
                                        algo5=pd.DataFrame(), algo6=pd.DataFrame(), algo7=pd.DataFrame(),
                                        algo8=pd.DataFrame() ):
    """
    # take in input from 2 to 8 pandas dataframe having different algorithm concatenate them in one and make a plot of comparison
    # according to the selected metric in x and y. A plot for every ROI in is generated.
    # There will be an error if the dataframes have not the same number of columns, and ROIs
    """
    data = pd.concat([algo1, algo2],axis=0, ignore_index=True)
    if not algo3.empty:
        data = pd.concat([data, algo3],axis=0, ignore_index=True)
    if not algo4.empty:
        data = pd.concat([data, algo4],axis=0, ignore_index=True)
    if not algo5.empty:
        data = pd.concat([data, algo5],axis=0, ignore_index=True)
    if not algo6.empty:
        data = pd.concat([data, algo6],axis=0, ignore_index=True)
    if not algo7.empty:
        data = pd.concat([data, algo7],axis=0, ignore_index=True)
    if not algo8.empty:
        data = pd.concat([data, algo8],axis=0, ignore_index=True)
        
    data.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
    data[data.index.duplicated()]

    seaborn_plot(data,x=x,y=y,colour="algo",  kind= "scatter", col="ROI", start_epoch=0)


    # # make a mask based on the reference FoV
    # offset_xy=reference.shape[reference.dim()-1]/2

    # mask = torch.zeros(reference.shape)
    # # if (reference.dim()<3):
    # #     for i in range(reference.shape[reference.dim()-3]):
    # #         for j in range(reference.shape[1]):
    # #                 if np.sqrt(np.square(i-offset_xy)+ np.square(j-offset_xy))<reference.shape[1]/2:
    # #                     mask[i,j]=1
    # # else:
    # for i in range(reference.shape[reference.dim()-3]):
    #     for j in range(reference.shape[reference.dim()-2]):
    #         for k in range(reference.shape[reference.dim()-1]):
    #             if np.sqrt(np.square(j-offset_xy)+ np.square(k-offset_xy))<reference.shape[reference.dim()-1]/2:
    #                 if(reference.dim()==3):
    #                     mask[i,j,k]=1
    #                 elif(reference.dim()==4):
    #                     mask[:,i,j,k]=1
    #                 elif(reference.dim()==5):
    #                     mask[:,:,i,j,k]=1

    # return mask

def get_mean_std_bias_images(prefix1, num_iter, prefix2, num_replicas, reference=torch.zeros(1), skip=None, aleatory=False ):
    """
    # returns the tensors for mean, std and bias (if reference tensor is passed) images. It requires the iteration number of the image you want to see
    # the prefix of the file name before and after (but without '.npy') the iteration number;
    # the number of times the algorithm was run. This assumes that the reconstructed images were saved in .npy format
    """

    splus=0

    if aleatory:
        algo0=prefix1+str(0)+'_aseed'+str(0)+'_'+prefix2+str(num_iter)+'.npy'
    else:
        algo0=prefix1+str(0)+'_'+prefix2+str(num_iter)+'.npy'
    if not os.path.isfile(algo0):
        print('Warning: I hve not found the following files ' + algo0 +'I will read seed=1')
        splus=1
        if aleatory:
            algo0=prefix1+str(1)+'_aseed'+str(0)+'_'+prefix2+str(num_iter)+'.npy'
        else:
                algo0=prefix1+str(1)+'_'+prefix2+str(num_iter)+'.npy'
      
    r =  torch.from_numpy(np.load(algo0,allow_pickle=True)).to(reference.device)  
    image=torch.zeros(r.shape).to(reference.device)
    # mask = (r != 0)
    mask_im = make_cylindrical_mask_tensor(r).to(reference.device)
    mask = (mask_im==1).to(reference.device)
    mean_im = torch.zeros(r.shape).to(reference.device)
    unc_map = torch.zeros(r.shape).to(reference.device)
    bias_map = torch.zeros(r.shape).to(reference.device)
    mse_map = torch.zeros(r.shape).to(reference.device)
    image[mask]=r[mask]     
    mean_im =  (mean_im+image)
    num_replicas_div = num_replicas
    for s in range(1,num_replicas):

        if skip  is not  None:
            if s in skip:
                print(skip)
                num_replicas_div = num_replicas_div-1
                continue;
        
        if aleatory:
            algo=prefix1+str(s)+'_aseed'+str(s)+'_'+prefix2+str(num_iter)+'.npy'
        else:
            algo=prefix1+str(s)+'_'+prefix2+str(num_iter)+'.npy'
        if not os.path.isfile(algo):
            print('Warning: I hve not found the following files ' + algo )
        image=torch.zeros(r.shape).to(reference.device)
        r =  torch.from_numpy(np.load(algo,allow_pickle=True)).to(reference.device)  
        image[mask]=r[mask] .to(reference.device)    
        mean_im =  (mean_im+image)
    mean_im = mean_im/num_replicas_div

    for s in range(num_replicas):
        if skip  is not  None:
            if s in skip:
                continue;
        
        algo=prefix1+str(s)+'_'+prefix2+str(num_iter)+'.npy'
        if not os.path.isfile(algo):
            print('Warning: I hve not found the following files ' + algo )
        image=torch.zeros(r.shape).to(reference.device)
        r =  torch.from_numpy(np.load(algo,allow_pickle=True)  ).to(reference.device)    
        image[mask]=r[mask].to(reference.device)     
        unc_map = (unc_map+ torch.square(image-mean_im))
        if reference.dim()>1:
            bias_map = bias_map + (image - reference)
        
    unc_map = torch.sqrt(unc_map/num_replicas_div)
    print("the effective number of replicas is: ",num_replicas_div)
    if reference.dim()>1:
        bias_map = bias_map/num_replicas_div#[mask] = bias_map[mask]/num_replicas/true_brain_np[mask]
        mse_map = mse_map + (torch.square(bias_map)+torch.square(unc_map))
        return mean_im, unc_map, bias_map, mse_map#.unsqueeze_(dim=0)
    else:
        return mean_im, unc_map#.unsqueeze_(dim=0).unsqueeze_(dim=0)  

def create_mean_dataframe_metric_with_uncertainties(data,metric,pref):
    """
    # returns a dataframe with "epoch","ROI", "mean_"+metric, metric+"_std" ,"algo","perc_"+metric+"_single_unc"
    #requires the dataframe obtained from the relative incertainty estimation algorithm, 
    # ex: for neuralKEM uncertainties_neuralKEM.py in algorithms. you can choose which metric to evaluate, atm: 'mean', bias' or 'CoVb'
    #if you want more, like Signal-to-noise-ratio this needs to be added in the create_metrics_with_gt_col_ROI() function
    """
    
    meandata=data.groupby(["epoch","ROI"], as_index=False).agg({metric:['mean','std']})
    meandata.columns = ["epoch","ROI", "mean_"+metric, metric+"_std"]
    meandata["algo"]=pref
    meandata["perc_"+metric+"_single_unc"] = np.abs(meandata[metric+"_std"]/meandata["mean_"+metric]*100)
    return meandata
           
def get_dataframe_with_covariance(total_data, x, y):

    """
    returns a dataframe equal to total_data with the addition of the covariance between the variable x and y
    """
    

    mean_std_data=total_data[['epoch',"ROI",x,y]]#
    covar={}
    covar['epoch']=[]
    covar['ROI']=[]
    covar['covar_'+x+'_'+y]=[]
    for i, m in mean_std_data.groupby(["epoch","ROI"], as_index=False):
       
        covar['covar_'+x+'_'+y].append(m[[x, y]].cov(ddof=0)[x][y])
        covar['epoch'].append(i[0])
        covar['ROI'].append(i[1])
    return  pd.DataFrame(covar)

def get_dataframe_with_uncertainties(dataframe, pref):
    """
    # From a dataframe with all the different seeds creates a dataframe with uncertanties 
    #   and calculates covariance between mean and std
    # this assumes that the BGR ROI is the one with the highest ID value
    """
    
    covar_mean_SD = get_dataframe_with_covariance(dataframe,'mean','SD',)
    covar_meanb_SDb = get_dataframe_with_covariance(dataframe,'meanb','SDb',)
    mean = create_mean_dataframe_metric_with_uncertainties(dataframe,'mean', pref)
    if 'bias' in dataframe.columns:
        bias = create_mean_dataframe_metric_with_uncertainties(dataframe,'bias', pref)

    cov = create_mean_dataframe_metric_with_uncertainties(dataframe,'CoV', pref)

    covb = create_mean_dataframe_metric_with_uncertainties(dataframe,'CoVb', pref)
    SD = create_mean_dataframe_metric_with_uncertainties(dataframe,'SD', pref)

    SDb = create_mean_dataframe_metric_with_uncertainties(dataframe,'SDb', pref)
    meanb = create_mean_dataframe_metric_with_uncertainties(dataframe,'meanb', pref)
    CNR = create_mean_dataframe_metric_with_uncertainties(dataframe,'CNR', pref)

    if 'bias' in dataframe.columns:
        tot_dataframe = pd.concat([mean, meanb, bias, SD, SDb, cov, covb, CNR],axis=1, ignore_index=False)
    else:
        tot_dataframe = pd.concat([mean, meanb, SD, SDb, cov, covb, CNR],axis=1, ignore_index=False)

    tot_dataframe = tot_dataframe.loc[:,~tot_dataframe.columns.duplicated()].copy()# remove duplicates
    tot_dataframe =  pd.merge(covar_mean_SD, tot_dataframe, on=['epoch','ROI'])   

    tot_dataframe =  pd.merge(covar_meanb_SDb, tot_dataframe, on=['epoch','ROI'])   
    
    if 'bias' in dataframe.columns:
        tot_dataframe['trade-off_d'] = np.sqrt(np.square(tot_dataframe['mean_CoVb']) + np.square(tot_dataframe['mean_bias']))
        covar_bias_cov = get_dataframe_with_covariance(tot_dataframe,'mean_bias','mean_CoVb')
        tot_dataframe =  pd.merge(covar_bias_cov, tot_dataframe, on=['epoch','ROI'])
        #, left_on=['A_c1','c2'], right_on = ['B_c1','c2'])
        #calculate uncertanties using the unc propagation rule:
        #  if Bias= (X-T)/T then it's uncertainty Ub=Ux/T assuming a Ut=0 from the ground true
        tot_dataframe['GT'] = (tot_dataframe['mean_mean']/(1+tot_dataframe['mean_bias']))
        tot_dataframe['propagated_Ub']=np.abs(tot_dataframe['mean_std']/
                                        (tot_dataframe['mean_mean']/(1+tot_dataframe['mean_bias'])))#this is the true value got from the bias formula
                                        #  ]
        tot_dataframe['perc_Ub'] =np.abs(tot_dataframe['propagated_Ub']/tot_dataframe['mean_bias']*100)


    #  if CoV= SD/X then it's uncertainty Ucov=sqrt(CoV*(square(Usd/SD) +square(Umean/mean) - 2*(covar_mean_sd)/(SD*mean) assuming a covar=0 ?
    tot_dataframe['propagated_Ucov']=np.abs(np.sqrt((np.square(tot_dataframe['SD_std']/(tot_dataframe['mean_SD'])) + 
                                                        np.square(tot_dataframe['mean_std']/(tot_dataframe['mean_mean'])) 
                                                            # - 2*tot_dataframe['covar_mean_SD']/tot_dataframe['mean_SD']/tot_dataframe['mean_mean']
                                        )))*tot_dataframe['mean_CoV']

    # Propagated uncertanty for CNR
    tot_dataframe['propagated_UCNR']=np.abs(tot_dataframe['mean_CNR']*np.sqrt(
         (np.square(tot_dataframe['mean_std']))/(tot_dataframe['mean_mean'] - tot_dataframe['mean_meanb']) + 
         (np.square(tot_dataframe['meanb_std']))/(tot_dataframe['mean_mean'] - tot_dataframe['mean_meanb'])+
          np.square(tot_dataframe['SDb_std']/(tot_dataframe['mean_SDb'])) #-
                                          #     2*tot_dataframe['mean_mean']*np.square(tot_dataframe['covar_meanb_SDb']/tot_dataframe['mean_SDb'])
                                        ))

    tot_dataframe['perc_Ucov'] =tot_dataframe['propagated_Ucov']/tot_dataframe['mean_CoV']*100
    #we need to have the bgr CoV uncertainty as a column to make a nice plot
    num_ROIs=tot_dataframe['ROI'].nunique()
    U_bgr_cov=create_full_column_from_bgr_ROI(tot_dataframe,'propagated_U_bgr_CoV', num_ROIs, num_ROIs)
    # join with tot_data
    tot_dataframe =  pd.merge(U_bgr_cov, tot_dataframe, on=['epoch','ROI'])

    if 'bias' in dataframe.columns:
    # uncertaninty propagation for f=sqrt(B^2+C^2) => sf =  sqrt((B*sb/f)^2 + (C*sC/f)^2+ 2*(B*C/(f^2))*covar_bc)
        tot_dataframe['propagated_U_d'] = np.sqrt(np.square(tot_dataframe['mean_bias']*tot_dataframe['propagated_Ub'])+
                                                np.square(tot_dataframe['mean_CoVb']*tot_dataframe['propagated_U_bgr_CoV'])+
                                                2*tot_dataframe['mean_bias']*tot_dataframe['mean_CoVb']*tot_dataframe['covar_mean_bias_mean_CoVb']
                                                )/tot_dataframe['trade-off_d']
        
    return tot_dataframe

def plot_metric_with_uncertainty(df,x, y, uy, classes, colour=None, grid=True, title_x=None, title_y=None, legend_loc=None, figure_name=None):
    """
    #plots the relationship between x and y with the uncertainty in y as shadow
    #it divides in sublot according to classes (ex:ROI)
    """
    
    if figure_name==None:
        figure_name=""
    if classes!=None and classes in df.columns:
        ncol=df[classes].value_counts().size
        fig, axes = plt.subplots(1,ncol,figsize=(5*ncol,5))
        

        if ncol == 1:
            print("ERROR: classes needs to have at least two different values. Use: classes=None")
            exit(1)
        else:
            
            if(title_y==None):
                axes[0].set_ylabel(y, fontsize=18)
            else:
                axes[0].set_ylabel(title_y, fontsize=18)
                
        

        if (colour==None):
            for i, m in df.groupby(classes):
                    axes[i-1].grid(grid)
                    if(title_x==None):
                        axes[i-1].set_xlabel(x, fontsize=18)
                    else:
                        axes[i-1].set_xlabel(title_x, fontsize=18)

                    axes[i-1].set_title(classes+': '+str(i), fontsize=18)
                    if (legend_loc is None):
                        axes[i-1].plot(m[x], m[y],label = '_nolegend_')
                    else:
                        axes[i-1].plot(m[x], m[y])
                    axes[i-1].fill_between(m[x],pd.to_numeric( m[y] - m[uy]), pd.to_numeric(m[y] + m[uy]), alpha=0.35)

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), fontsize=18,loc=legend_loc)
        else:
            ii=0
            for i, m in df.groupby(classes):
                for j, mm in m.groupby(colour):
                    axes[ii].grid(grid)
                    if(title_x==None):
                        axes[i-1].set_xlabel(x, fontsize=18)
                    else:
                        axes[i-1].set_xlabel(title_x, fontsize=18)

                    axes[ii].set_title(classes+': '+str(ii+1), fontsize=18)
                    if (legend_loc is None):
                        axes[ii].plot(mm[x], mm[y],label = '_nolegend_')
                    else:
                        axes[ii].plot(mm[x], mm[y], label=mm[colour].values[0])

                    axes[ii].fill_between(mm[x], pd.to_numeric(mm[y] - mm[uy]), pd.to_numeric(mm[y] + mm[uy]), alpha=0.35)
                ii=ii+1

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), fontsize=18,loc=legend_loc)
    else:
        ncol=1
        
        fig, axes = plt.subplots(1,ncol,figsize=(10*ncol,10))
        
        if(title_y==None):
            axes.set_ylabel(y, fontsize=18)
        else:
            axes.set_ylabel(title_y, fontsize=18)
        ii=0
        for i, m in df.groupby(colour):
            # for c in range(ncol):
                axes.grid(grid)
                # axes[i-1].label_outer()
                if(title_x==None):
                    axes.set_xlabel(x, fontsize=18)
                else:
                    axes.set_xlabel(title_x, fontsize=18)


                axes.set_title(y, fontsize=18)
                if (colour==None):
                    if (legend_loc is None):
                        axes.plot(m[x], m[y],label = '_nolegend_')
                    else:
                        axes.plot(m[x], m[y])
                else:
                    if (legend_loc is None):
                        axes.plot(m[x], m[y],label = '_nolegend_')
                    else:
                        axes.plot(m[x], m[y], label=m[colour].values[ii])

                axes.fill_between(m[x],pd.to_numeric( m[y] - m[uy]), pd.to_numeric(m[y] + m[uy]), alpha=0.35)
                ii=ii+1
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=18,loc=legend_loc)
    fig.savefig(figure_name+".png")
      
def plot_metric_with_uncertainty_scatter(df,x, y, classes, ux=None, uy=None,  colour=None, grid=True, title_x=None, title_y=None, legend_loc=None,figure_name=None):

    """
    #plots the relationship between x and y with the uncertainty in y as shadow
    #it divides in sublot according to classes (ex:ROI)
    """
    if figure_name==None:
        figure_name=""
    size=4
    if classes!=None and classes in df.columns:
        ncol=df[classes].value_counts().size
        fig, axes = plt.subplots(1,ncol,figsize=(5*ncol,5))

        if ncol == 1:
            print("ERROR: classes needs to have at least two different values")
            exit(1)
        else:
            if(title_y==None):
                axes[0].set_ylabel(y, fontsize=18)
            else:
                axes[0].set_ylabel(title_y, fontsize=18)

        if (colour==None):
                for i, m in df.groupby(classes):
                    axes[i-1].grid(grid)
                    if(title_x==None):
                        axes[i-1].set_xlabel(x, fontsize=18)
                    else:
                        axes[i-1].set_xlabel(title_x, fontsize=18)

                    axes[i-1].set_title(classes+': '+str(i), fontsize=18)
                    # axes[i-1].plot(m[x], m[y])
                    if (uy==None and ux==None):
                        if (legend_loc is None):
                            axes[i-1].errorbar(m[x], m[y],  ms=size, fmt='o', label = '_nolegend_')
                        else:
                            axes[i-1].errorbar(m[x], m[y],  ms=size, fmt='o',  label=m[colour].values[i])  
                    
                    elif (uy==None):
                        if (legend_loc is None):
                            axes[i-1].errorbar(m[x], m[y],  xerr= [m[ux],m[ux]], linestyle='None', ms=size, fmt='o', label = '_nolegend_')
                        else:
                            axes[i-1].errorbar(m[x], m[y], xerr= [m[ux],m[ux]], linestyle='None', ms=size, fmt='o', label=m[colour].values[i]) 
                    
                    elif (ux==None):
                        if (legend_loc is None):
                            axes[i-1].errorbar(m[x], m[y],  yerr= [m[uy],m[uy]], linestyle='None', ms=size, fmt='o', label = '_nolegend_')
                        else:
                            axes[i-1].errorbar(m[x], m[y], yerr = [m[uy], m[uy]], linestyle='None', ms=size, fmt='o', label=m[colour].values[i]) 
                    
                    else:
                        if (legend_loc is None):
                            axes[i-1].errorbar(m[x], m[y],  yerr= [m[uy],m[uy]], xerr= [m[ux],m[ux]], linestyle='None', ms=size, fmt='o', label = '_nolegend_')
                        else:
                            axes[i-1].errorbar(m[x], m[y], yerr = [m[uy],m[uy]], xerr= [m[ux],m[ux]], linestyle='None', ms=size, fmt='o', label=m[colour].values[i]) 
                
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), fontsize=12,loc=legend_loc)
        else:
            ii=0
            for i, m in df.groupby(classes):
                for j, mm in m.groupby(colour):
                    axes[ii].grid(grid)
                    
                    if(title_x==None):
                        axes[ii].set_xlabel(x, fontsize=18)
                    else:
                        axes[ii].set_xlabel(title_x, fontsize=18)

                    axes[ii].set_title(classes+': '+str(m[classes].head(1)), fontsize=18)
                    # axes[ii].plot(mm[x], mm[y], label=mm[colour].values[0])
                    if (uy==None and ux==None):
                        if (legend_loc is None):
                            axes[ii].errorbar(mm[x], mm[y],  ms=size, fmt='o', label = '_nolegend_')
                        else:
                            axes[ii].errorbar(mm[x], mm[y],  ms=size, fmt='o',  label=mm[colour].values[0])  
                    elif (uy==None):
                        if (legend_loc is None):
                            axes[ii].errorbar(mm[x], mm[y],  xerr= [mm[ux],mm[ux]], linestyle='None', ms=size, fmt='o', label = '_nolegend_')
                        else:
                            axes[ii].errorbar(mm[x], mm[y], xerr= [mm[ux],mm[ux]], linestyle='None', ms=size, fmt='o', label=mm[colour].values[0]) 
                    elif (ux==None):

                        if (legend_loc is None):
                            axes[ii].errorbar(mm[x], mm[y],  yerr= [mm[uy],mm[uy]], linestyle='None', ms=size, fmt='o', label = '_nolegend_')
                        else:
                            axes[ii].errorbar(mm[x], mm[y], yerr = [mm[uy], mm[uy]], linestyle='None', ms=size, fmt='o', label=mm[colour].values[0]) 
                    else:

                        if (legend_loc is None):
                            axes[ii].errorbar(mm[x], mm[y], yerr = [mm[uy],mm[uy]], xerr= [mm[ux],mm[ux]], linestyle='None', ms=size, fmt='o', label = '_nolegend_')
                        else:
                            axes[ii].errorbar(mm[x], mm[y], yerr = [mm[uy],mm[uy]], xerr= [mm[ux],mm[ux]], linestyle='None', ms=size, fmt='o', label=mm[colour].values[0]) 
                
                ii=ii+1
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), fontsize=12,loc=legend_loc)
    else:
        ncol=1
        
        fig, axes = plt.subplots(1,ncol,figsize=(10*ncol,10))
        if(title_y==None):
            axes.set_ylabel(y, fontsize=18)
        else:
            axes.set_ylabel(title_y, fontsize=18)

        ii=0
        for i, m in df.groupby(colour):
            axes.grid(grid)
            
            if(title_x==None):
                axes.set_xlabel(x, fontsize=18)
            else:
                axes.set_xlabel(title_x, fontsize=18)


            axes.set_title(y, fontsize=18)
            # if (colour==None):
            #     axes.plot(m[x], m[y])
            # else:
            #     axes.plot(m[x], m[y], label=m[colour].values[ii])
            # axes.plot(m[x], m[y],label=m[colour].values[i], marker='o')
            
            if (ux==None and uy!=None):
                if (legend_loc is None):
                        axes.errorbar(m[x], m[y], yerr = [m[uy],m[uy]], linestyle='None', ms=size, fmt='o', label = '_nolegend_')
                else:
                        axes.errorbar(m[x], m[y], yerr = [m[uy],m[uy]], linestyle='None', ms=size, fmt='o', label=m[colour].values[ii])  
            
            elif (uy==None and ux!=None):
                if (legend_loc is None):
                        axes.errorbar(m[x], m[y], xerr= [m[ux],m[ux]], linestyle='None', ms=size, fmt='o', label = '_nolegend_')
                else:
                        axes.errorbar(m[x], m[y], xerr= [m[ux],m[ux]], linestyle='None', ms=size, fmt='o', label=m[colour].values[ii]) 
            
            elif (uy==None and ux==None):
                if (legend_loc is None):
                        axes.errorbar(m[x], m[y], ms=size, fmt='o', label = '_nolegend_')
                else:
                        axes.errorbar(m[x], m[y], ms=size, fmt='o', label=m[colour].values[ii]) 
            
            else:
                if (legend_loc is None):
                        axes.errorbar(m[x], m[y], yerr = [m[uy],m[uy]], xerr= [m[ux],m[ux]], linestyle='None', ms=size, fmt='o', label = '_nolegend_')
                else:
                        axes.errorbar(m[x], m[y], yerr = [m[uy],m[uy]], xerr= [m[ux],m[ux]], linestyle='None', ms=size, fmt='o', label=m[colour].values[ii]) 
            
            ii=ii+1
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=12,loc=legend_loc)
    fig.savefig(figure_name+".png")

def create_full_column_from_bgr_ROI(df,column, roi_num, num_rois):
    """
    returns a dataframe epoch, ROI, and column were column has the same value over the ROI but changes with epoch
    """
    
    list={}
    list['epoch']=[]
    list['ROI']=[]
    list[column]=[]
    for i, m in df.groupby(["epoch","ROI"], as_index=False):
        
        list['epoch'].append(i[0])
        list['ROI'].append(i[1])
        if (i[1]==roi_num):
            for i in range(num_rois):
                list[column].append(m.propagated_Ucov.values[0])
    return  pd.DataFrame(list)

def get_multi_mse_dataframe(pref1, pref2,num_epoch, num_replicas, reference, recon_name=None, islog10=True):
    """
    estimate_MSE_and_save(string1,num_iter, string2, recon_name, reference, skip_iter=0, islog10=False)
    """
    df= pd.DataFrame()
    for s in range(num_replicas):
        algo_pref=pref1+str(s)+'_'+pref2
        mse=estimate_MSE_and_save(algo_pref, num_epoch, '.npy', recon_name, reference,islog10=islog10)
        df= pd.concat([df,pd.DataFrame(mse)])
        df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
    #concat
    return df

def get_multi_loss_dataframe(pref1, pref2,num_epoch, num_replicas, islog10=True):
    """
    estimate_MSE_and_save(string1,num_iter, string2, recon_name, reference, skip_iter=0, islog10=False)
    """
    df= pd.DataFrame()

    multi_loss = {}
    multi_loss['loss'] = []
    multi_loss['epoch'] = []
    multi_loss['seed'] = []

    for s in range(num_replicas):
        algo_pref=pref1+str(s)+'_'+pref2
        loss= np.load(algo_pref+ '_data_log.npy', allow_pickle=True).item()
        if islog10:
            multi_loss['loss'] = np.log10(loss['loss'])
        else:
            multi_loss['loss'] = (loss['loss'])

        multi_loss['epoch'] = (loss['epoch'])
        multi_loss['seed'] = s
        df= pd.concat([df,pd.DataFrame(multi_loss)])
        df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')
    #concat
    return df

def create_mean_dataframe_with_uncertainties(data, pref, metric):
    # metric='mse'
    meandata=data.groupby(["epoch"], as_index=False).agg({metric:['mean','std']})
    meandata.columns = ["epoch", "mean_"+metric, metric+"_std"]
    meandata["algo"]=pref
    meandata["perc_"+metric+"_single_unc"] = np.abs(meandata[metric+"_std"]/meandata["mean_"+metric]*100)
    return meandata

def create_mean_dataframe_mse_with_uncertainties(data, pref):
    metric='mse'
    meandata=data.groupby(["epoch"], as_index=False).agg({metric:['mean','std']})
    meandata.columns = ["epoch", "mean_"+metric, metric+"_std"]
    meandata["algo"]=pref
    meandata["perc_"+metric+"_single_unc"] = np.abs(meandata[metric+"_std"]/meandata["mean_"+metric]*100)
    return meandata

def create_SSIM_dataframe(prefix1,num_replicas,num_iter, prefix2, target):
    """
    returns a dataframe with "epoch","SSIM", "seed"
    SSIM is estimated using get_SSIM 
    """
    ssim = { }
    ssim['epoch'] = [ ]
    ssim['SSIM'] = [ ]
    ssim['seed'] = [ ]
     

    for i in range(num_replicas):
        for j in range(num_iter):
            algo_filename = prefix1+str(i)+'_'+prefix2+str(j)+'.npy'
            if not os.path.isfile(algo_filename):
            #  print('Warning: the following file is not found. Will look for the next iteration ' + string )
                continue
            pred = torch.from_numpy(np.load(algo_filename,allow_pickle=True)).to(device) 
            ssim['SSIM'].append(get_SSIM(pred, target))
            ssim['epoch'].append(j)
            ssim['seed'].append(i)
    return pd.DataFrame(ssim)

def get_SSIM(pred, target):
    """
    Estimates SSIM given a predicted image and a target image.  This uses pytprch ignite package.
    """
    if pred.dim()!= target.dim():
        print('Error: prediction and target have different dimensions!')

    if pred.dim()==5 :
        pred = pred[0,...]
        target = target[0,...]

    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)

    metric = SSIM(data_range=1.0)
    metric.attach(default_evaluator, 'ssim')
    
    state = default_evaluator.run([[pred, target]])
    return state.metrics['ssim']

