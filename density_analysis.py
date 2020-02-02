
# calculate surface density of display, divide trials in low and high density
# calculate correlations for each case, to see if it changes outcome

import sys, os

# append Pygaz analyser folder, cloned from https://github.com/esdalmaijer/PyGazeAnalyser.git
sys.path.append(os.getcwd()+'/PyGazeAnalyser/')
import pygazeanalyser
import numpy as np

import pandas as pd
import glob

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
#%matplotlib inline

import json

from utils import *

# paths
base_dir =os.getcwd(); base_dir = os.path.join(base_dir,'Data_50ms_7deg')
output_vs = os.path.join(base_dir,'output_VS')
output_crwd = os.path.join(base_dir,'output_crowding')

# define dir to save plots
plot_dir = os.path.join(base_dir,'plots')
if not os.path.exists(plot_dir):
     os.makedirs(plot_dir)


with open(os.path.join(os.getcwd(),'lab_parameters.json'),'r') as json_file:	
            analysis_params = json.load(json_file)	
        

# Parameters
total_trials = 720
blocks = analysis_params['blk_crw'] #including one practice block
trials_block = total_trials/blocks
trials = total_trials - trials_block #number of trials excluding training
ecc = analysis_params['ecc']
last_trials = 96 #number of trials to calculate critical distance

cut_off = 0.6 # level upon which we don't consider performance (accuracy) to be chance level

# load list of csv and edf files in folder
crwd_csv = [x for _,x in enumerate(os.listdir(output_crwd)) if x.endswith('.csv')]; crwd_csv.sort()
crwd_edf = [x for _,x in enumerate(os.listdir(output_crwd)) if x.endswith('.EDF')]; crwd_edf.sort() 

vs_csv = [x for _,x in enumerate(os.listdir(output_vs)) if x.endswith('.csv')]; vs_csv.sort()
vs_edf = [x for _,x in enumerate(os.listdir(output_vs)) if x.endswith('.EDF')]; vs_edf.sort() 

# p-value thresth
p_value = 0.05


# loop over subjects

all_subs = [] # list with all subject number
missed_trials = [] # list with percentage of missed trials (crowding)
acc_fl = []
acc_nofl = []
acc_comb = []

all_tfr = []
all_cs = []
mean_cs = [] 

acc_vs_ecc = []
acc_vs_all = []

rt_vs = []

excluded_sub = []

for ind,behfile in enumerate(crwd_csv):
    
    EXCLUDE = []
    
    all_subs.append(os.path.splitext(behfile)[0][-2::]) 
    print('analysing pp-%s'%all_subs[ind])
    
    # load csv for sub
    data_sub = pd.read_csv(os.path.join(output_crwd,behfile), sep='\t')
    # choose only actual pratice block (first block (first 144) where used to save training, doesn't count)
    data_sub = data_sub.loc[int(trials_block)::]
    
    # check for missed trials
    missed_trials.append(check_nan_trials(data_sub,exclusion_thresh=0.25)[0])
    EXCLUDE.append(check_nan_trials(data_sub,exclusion_thresh=0.25)[1])
    
    # check for accuracy with and without flankers
    # so actually see if crowding manipulation worked
    accur_crowding = accuracy_crowding(data_sub,ecc,exclusion_thresh=cut_off)
    acc_fl.append(accur_crowding['acc_ecc_flank'])
    acc_nofl.append(accur_crowding['acc_ecc_noflank'])
    acc_comb.append(accur_crowding['overall_acc'])
    EXCLUDE.append(accur_crowding['exclude'])
    
    cs = critical_spacing(data_sub,ecc,num_trl=last_trials)
    all_tfr.append(cs['all_tfr'])
    all_cs.append(cs['all_crit_dis'])
    mean_cs.append(cs['mean_CS'])
    EXCLUDE.append(cs['exclude'])
    
    
    # visual search
    
    # load csv for sub
    df_vs = pd.read_csv(os.path.join(output_vs,vs_csv[ind]), sep='\t')
    
    vs_acc = accuracy_search(df_vs,ecc,exclusion_all_thresh=0.85,exclusion_ecc_thresh=0.75)
    
    acc_vs_ecc.append(vs_acc['acc_ecc'])
    acc_vs_all.append(vs_acc['overall_acc'])
    
    EXCLUDE.append(vs_acc['exclude'])
    
    
    # check exclusion and save pp number
    if any(EXCLUDE)==True:
        print('need to exclude subject %s'%all_subs[ind])
        excluded_sub.append(all_subs[ind])
        ex_sub = 'True'
    else:
        ex_sub = 'False'
    

# then take out excluded participants and save relevant info in new arrays
# for later analysis

#params
rad_roi = 4 # radius around target 
quantile = .5 

# find index to take out from variables
exclusion_ind = [np.where(np.array(all_subs)==np.array(excluded_sub[i]))[0][0] for i in range(len(excluded_sub))] 

counter = 0
test_subs = []
test_all_cs = []
test_mean_cs = []

test_rt_ecc_vs_LOW = []
test_rt_ecc_vs_HIGH = []

test_rt_set_vs_LOW = []
test_rt_set_vs_HIGH = []

test_fix_ecc_vs_LOW = []
test_fix_ecc_vs_HIGH = []

test_fix_set_vs_LOW = []
test_fix_set_vs_HIGH = []

test_onobjfix_ecc_vs_LOW = []
test_onobjfix_ecc_vs_HIGH = []

test_onobjfix_set_vs_LOW = []
test_onobjfix_set_vs_HIGH = []

test_density_thresh = []

for j in range(len(all_subs)):
    
    if j == exclusion_ind[counter]:
        print('skipping sub %s'%all_subs[j])
        if counter<len(exclusion_ind)-1:
            counter+=1   
    else:
        # load data frame and eyedata
        df_vs = pd.read_csv(os.path.join(output_vs,vs_csv[j]), sep='\t')
        _, eye_data = convert2asc(all_subs[j],'vs',output_vs,output_crwd)
        
        # compute density for all trials of subject
        surf_dens_all = surface_density(os.path.join(output_vs,vs_csv[j]),rad_roi)
        
        # plot distribution of display density and 
        median_density = np.nanquantile(surf_dens_all, quantile)

        # make figure to have a look
        fig, axis = plt.subplots(1,1,figsize=(15,7.5),dpi=100)

        sns.distplot(surf_dens_all,bins=10,color='r')
        axis.set_xlabel('surface density',fontsize=14)
        axis.set_xlim(0,)
        axis.axvline(x=median_density,c='k',linestyle='--')

        axis.set_title('Histogram of surface density values sub-%s, median density = %0.2f'%(all_subs[j],median_density))
        fig.savefig(os.path.join(plot_dir,'distribution_surface_density_sub-%s'%all_subs[j]), dpi=100)

        # append density threshold
        test_density_thresh.append(median_density)
        
        test_subs.append(all_subs[j])
        
        # critical spacing
        test_all_cs.append(all_cs[j])
        test_mean_cs.append(mean_cs[j])
        
        # THIS IS WHAT I NEED TO SAVE DIFFERENTLY, ACCORDING TO DENSITY ##########
        # RT per ECC or SET SIZE
        test_rt_ecc_vs_LOW.append(density_mean_RT(df_vs,surf_dens_all,
                                                  type_trial='ecc',density='low',threshold=median_density))
        test_rt_ecc_vs_HIGH.append(density_mean_RT(df_vs,surf_dens_all,
                                                   type_trial='ecc',density='high',threshold=median_density))

        test_rt_set_vs_LOW.append(density_mean_RT(df_vs,surf_dens_all,
                                                  type_trial='set',density='low',threshold=median_density))
        test_rt_set_vs_HIGH.append(density_mean_RT(df_vs,surf_dens_all,
                                                   type_trial='set',density='high',threshold=median_density))
        
        # FIXATIONS per ECC or SET SIZE
        test_fix_ecc_vs_LOW.append(density_meanfix(df_vs,eye_data,surf_dens_all,
                                                   type_trial='ecc',density='low',threshold=median_density))
        test_fix_ecc_vs_HIGH.append(density_meanfix(df_vs,eye_data,surf_dens_all,
                                                    type_trial='ecc',density='high',threshold=median_density))

        test_fix_set_vs_LOW.append(density_meanfix(df_vs,eye_data,surf_dens_all,
                                                   type_trial='set',density='low',threshold=median_density))
        test_fix_set_vs_HIGH.append(density_meanfix(df_vs,eye_data,surf_dens_all,
                                                    type_trial='set',density='high',threshold=median_density))

        # ON-OBJECT FIXATIONS per ECC or SET SIZE
        test_onobjfix_ecc_vs_LOW.append(density_on_objectfix(df_vs,eye_data,surf_dens_all,
                                                             type_trial='ecc',density='low',threshold=median_density,
                                                             radius=analysis_params['siz_gab_deg']/2*1.5))
        test_onobjfix_ecc_vs_HIGH.append(density_on_objectfix(df_vs,eye_data,surf_dens_all,
                                                             type_trial='ecc',density='high',threshold=median_density,
                                                             radius=analysis_params['siz_gab_deg']/2*1.5))

        test_onobjfix_set_vs_LOW.append(density_on_objectfix(df_vs,eye_data,surf_dens_all,
                                                             type_trial='set',density='low',threshold=median_density,
                                                             radius=analysis_params['siz_gab_deg']/2*1.5))
        test_onobjfix_set_vs_HIGH.append(density_on_objectfix(df_vs,eye_data,surf_dens_all,
                                                             type_trial='set',density='high',threshold=median_density,
                                                             radius=analysis_params['siz_gab_deg']/2*1.5))


# PLOTS

# RT VS ECC
fig = plt.figure(figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
 
rt_ecc_vs4plot = pd.DataFrame([])

for k in range(len(ecc)):
    rt_ecc_vs4plot = rt_ecc_vs4plot.append(pd.DataFrame({'RT_LOW': np.array(test_rt_ecc_vs_LOW).T[k],
                                                                 'RT_HIGH': np.array(test_rt_ecc_vs_HIGH).T[k],
                                                                 'ecc':np.tile(ecc[k],len(test_subs))}))

sns.regplot(x=rt_ecc_vs4plot['ecc'],y=rt_ecc_vs4plot['RT_LOW'],color='blue', marker='.',label='LOW')
sns.regplot(x=rt_ecc_vs4plot['ecc'],y=rt_ecc_vs4plot['RT_HIGH'],color='red', marker='+',label='HIGH')

ax = plt.gca()
ax.set(xlabel='eccentricity [dva]', ylabel='RT [s]')
ax.set_title('ecc vs RT %d subs'%len(test_subs))
ax.legend()
plt.savefig(os.path.join(plot_dir,'density_search_ecc_RT_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')
 

# RT VS SET SIZE

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

rt_set_vs4plot = pd.DataFrame([])

for k in range(len(analysis_params['set_size'])):
    rt_set_vs4plot = rt_set_vs4plot.append(pd.DataFrame({'RT_LOW': np.array(test_rt_set_vs_LOW).T[k],
                                                         'RT_HIGH': np.array(test_rt_set_vs_HIGH).T[k],
                                                     'set':np.tile(analysis_params['set_size'][k],len(test_subs))}))

sns.regplot(x=rt_set_vs4plot['set'],y=rt_set_vs4plot['RT_LOW'],color='blue', marker='.',label='LOW')
sns.regplot(x=rt_set_vs4plot['set'],y=rt_set_vs4plot['RT_HIGH'],color='red', marker='+',label='HIGH')
    
ax = plt.gca()
ax.set(xlabel='set size', ylabel='RT [s]')
ax.set_title('set size vs RT %d subs'%len(test_subs))
ax.legend()
plt.savefig(os.path.join(plot_dir,'density_search_setsize_RT_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')
 
# EYETRACKING FOR VISUAL SEARCH

# NUMBER OF FIXATIONS VS ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

fix_ecc_vs4plot = pd.DataFrame([])

for k in range(len(ecc)):
    fix_ecc_vs4plot = fix_ecc_vs4plot.append(pd.DataFrame({'fix_LOW': np.array(test_fix_ecc_vs_LOW).T[k],
                                                           'fix_HIGH': np.array(test_fix_ecc_vs_HIGH).T[k],
                                                     'ecc':np.tile(ecc[k],len(test_subs))}))

sns.regplot(x=fix_ecc_vs4plot['ecc'],y=fix_ecc_vs4plot['fix_LOW'],color='blue', marker='.',label='LOW')
sns.regplot(x=fix_ecc_vs4plot['ecc'],y=fix_ecc_vs4plot['fix_HIGH'],color='red', marker='+',label='HIGH')

ax = plt.gca()
ax.set_title('ecc vs number of fixations %d subs'%len(test_subs))
ax.set(xlabel='eccentricity [dva]', ylabel='# fixations')
ax.axes.set_ylim(0,)
ax.legend()
plt.savefig(os.path.join(plot_dir,'density_search_ecc_numfix_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')
 

# NUMBER OF FIXATIONS VS SET SIZE
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

fix_set_vs4plot = pd.DataFrame([])

for k in range(len(analysis_params['set_size'])):
    fix_set_vs4plot = fix_set_vs4plot.append(pd.DataFrame({'fix_LOW': np.array(test_fix_set_vs_LOW).T[k],
                                                           'fix_HIGH': np.array(test_fix_set_vs_HIGH).T[k],
                                                     'set':np.tile(analysis_params['set_size'][k],len(test_subs))}))

    
sns.regplot(x=fix_set_vs4plot['set'],y=fix_set_vs4plot['fix_LOW'],color='blue', marker='.',label='LOW')
sns.regplot(x=fix_set_vs4plot['set'],y=fix_set_vs4plot['fix_HIGH'],color='red', marker='+',label='HIGH')
    
ax = plt.gca()

ax.set(xlabel='set size', ylabel='# fixations')
ax.set_title('set size vs number fixations %d subs'%len(test_subs))
ax.axes.set_ylim(0,)
ax.legend()
plt.savefig(os.path.join(plot_dir,'density_search_setsize_fix_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')


# PERCENTAGE OF ON OBJECT FIXATIONS VS ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

onobj_ecc_vs4plot = pd.DataFrame([])

for k in range(len(ecc)):
    onobj_ecc_vs4plot = onobj_ecc_vs4plot.append(pd.DataFrame({'onobj_LOW': np.array(test_onobjfix_ecc_vs_LOW).T[k]*100,
                                                               'onobj_HIGH': np.array(test_onobjfix_ecc_vs_HIGH).T[k]*100,
                                                     'ecc':np.tile(ecc[k],len(test_subs))}))

sns.regplot(x=onobj_ecc_vs4plot['ecc'],y=onobj_ecc_vs4plot['onobj_LOW'],color='blue', marker='.',label='LOW')
sns.regplot(x=onobj_ecc_vs4plot['ecc'],y=onobj_ecc_vs4plot['onobj_HIGH'],color='red', marker='+',label='HIGH')    

ax = plt.gca()
ax.set_title('ecc vs on-object fixations %d subs'%len(test_subs))
ax.set(xlabel='eccentricity [dva]', ylabel='On object fixation [%]')
ax.axes.set_ylim(0,)
ax.legend()
plt.savefig(os.path.join(plot_dir,'density_search_ecc_onobjectfix_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')


# PERCENTAGE OF ON OBJECT FIXATIONS VS SET SIZE
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

onobj_set_vs4plot = pd.DataFrame([])

for k in range(len(analysis_params['set_size'])):
    onobj_set_vs4plot = onobj_set_vs4plot.append(pd.DataFrame({'onobj_LOW': np.array(test_onobjfix_set_vs_LOW).T[k]*100,
                                                               'onobj_HIGH': np.array(test_onobjfix_set_vs_HIGH).T[k]*100,
                                                     'set':np.tile(analysis_params['set_size'][k],len(test_subs))}))

sns.regplot(x=onobj_set_vs4plot['set'],y=onobj_set_vs4plot['onobj_LOW'],color='blue', marker='.',label='LOW')
sns.regplot(x=onobj_set_vs4plot['set'],y=onobj_set_vs4plot['onobj_HIGH'],color='red', marker='+',label='HIGH')    
    
ax = plt.gca()
ax.set_title('set size vs on-object fixations %d subs'%len(test_subs))
ax.set(xlabel='set size', ylabel='On object fixation [%]')
ax.axes.set_ylim(0,)
ax.legend()
plt.savefig(os.path.join(plot_dir,'density_search_setsize_onobjectfix_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')
 














