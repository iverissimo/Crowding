
# calculate surface density of display, divide trials in low and high density
# calculate correlations for each case, to see if it changes outcome

import sys, os

# append Pygaz analyser folder, cloned from https://github.com/esdalmaijer/PyGazeAnalyser.git
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'PyGazeAnalyser'))
import pygazeanalyser

import numpy as np

import pandas as pd
import glob

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
#%matplotlib inline

import json

from utils import *

from scipy.stats import linregress


# open params jason file
with open(os.path.join(os.getcwd(),'settings.json'),'r') as json_file:  
            params = json.load(json_file)   
       
# paths
output_vs = params['datadir_vs']
output_crwd = params['datadir_crwd']

# define dir to save plots
plot_dir = os.path.join(os.path.split(output_vs)[0],'plots')
if not os.path.exists(plot_dir):
     os.makedirs(plot_dir)


# Params CROWDING
total_trials = params['num_trials_crwd'] 
blocks = params['num_blocks_crwd'] #including one practice block
trials_block = total_trials/blocks # trials per block

trials = total_trials - trials_block #number of trials excluding training
ecc = params['ecc']
last_trials = params['num_cs_trials'] #number of trials to calculate critical distance

cut_off = params['cut_off_acc_crwd'] # level upon which we don't consider performance (accuracy) to be chance level

p_value = params['p_val_crwd'] # p-value thresh

miss_exclusion_thresh = params['cut_off_miss_trials_crwd'] # percentage of missed trials for crowding
vs_exclusion_acc_thresh= params['cut_off_acc_vs'] # percentage of accurate trials for search
vs_exclusion_acc_ecc_thresh=params['cut_off_acc_ecc_vs'] # percentage of accurate trials for search (per ecc)


# list of csv and edf filenames in folder
crwd_csv = [os.path.join(output_crwd,x) for _,x in enumerate(os.listdir(output_crwd)) if x.endswith('.csv')]; crwd_csv.sort()
crwd_edf = [os.path.join(output_crwd,x) for _,x in enumerate(os.listdir(output_crwd)) if x.endswith('.EDF')]; crwd_edf.sort() 

vs_csv = [os.path.join(output_vs,x) for _,x in enumerate(os.listdir(output_vs)) if x.endswith('.csv')]; vs_csv.sort()
vs_edf = [os.path.join(output_vs,x) for _,x in enumerate(os.listdir(output_vs)) if x.endswith('.EDF')]; vs_edf.sort() 


# check if sub should be excluded
sum_file = os.path.join(os.path.split(output_vs)[0],'plots','summary','sum_measures.npz')

if not os.path.isfile(sum_file):
    sum_file = exclude_subs(crwd_csv,crwd_edf,vs_csv,plot_dir,
                                 trials_block = trials_block,
                                 miss_trials = miss_exclusion_thresh,
                                 acc_cut_off_crwd = cut_off,
                                 ecc = ecc,
                                 num_cs_trials = last_trials,
                                 cut_off_acc_vs = vs_exclusion_acc_thresh,
                                 cut_off_acc_ecc_vs = vs_exclusion_acc_ecc_thresh)

sum_measures = np.load(sum_file) # all relevant measures
  

# then take out excluded participants and save relevant info in new arrays
# for later analysis

# find index to take out from variables
exclusion_ind = [np.where(np.array(sum_measures['all_subs'])==np.array(sum_measures['excluded_sub'][i]))[0][0] for i in range(len(sum_measures['excluded_sub']))] 

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

test_iscrowded = []

# data frames with interesting values divided
test_df_RT_LOW = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
test_df_RT_HIGH = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])

test_df_fix_LOW = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
test_df_fix_HIGH = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])

test_df_trialnum_LOW = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
test_df_trialnum_HIGH = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])


for j in range(len(sum_measures['all_subs'])):
    
    if j == exclusion_ind[counter]:
        print('skipping sub %s'%sum_measures['all_subs'][j])
        if counter<len(exclusion_ind)-1:
            counter+=1   
    else:
        # load behav data
        df_vs = pd.read_csv(vs_csv[j], sep='\t')
        
        # load eye data
        asccii_name = convert2asc(sum_measures['all_subs'][j],'search',output_vs)
        print('loading edf for sub-%s data'%sum_measures['all_subs'][j])
        eye_data = read_edf(asccii_name, 'start_trial', stop='stop_trial', debug=False)
        
        # label trials as crowded or not
        is_crowded = crowded_trials(df_vs)

        # append percentage of crowded trials
        test_iscrowded.append(np.sum(is_crowded==True)/len(is_crowded)*100)
        
        test_subs.append(sum_measures['all_subs'][j])
        
        # critical spacing
        test_all_cs.append(sum_measures['all_cs'][j])
        test_mean_cs.append(sum_measures['mean_cs'][j])
        
        # THIS IS WHAT I NEED TO SAVE DIFFERENTLY, ACCORDING TO DENSITY ##########
        # RT per ECC or SET SIZE
        
        ## reaction time search 
        df_RT_LOW, df_trial_LOW = density_mean_RT(df_vs,is_crowded,
                                       sum_measures['all_subs'][j],density='low')
        df_RT_HIGH, df_trials_HIGH = density_mean_RT(df_vs,is_crowded,
                                       sum_measures['all_subs'][j],density='high')
        # append matrix of combo 
        test_df_RT_LOW = test_df_RT_LOW.append(df_RT_LOW,ignore_index=True)
        test_df_RT_HIGH = test_df_RT_HIGH.append(df_RT_HIGH,ignore_index=True)
        
        test_rt_ecc_vs_LOW.append(np.array([np.nanmean(df_RT_LOW[str(x)+'_ecc'].values) for _,x in enumerate(ecc)]))
        test_rt_ecc_vs_HIGH.append(np.array([np.nanmean(df_RT_HIGH[str(x)+'_ecc'].values) for _,x in enumerate(ecc)]))

        test_rt_set_vs_LOW.append(df_RT_LOW[[str(x)+'_ecc' for _,x in enumerate(ecc)]].mean(axis=1).values)
        test_rt_set_vs_HIGH.append(df_RT_HIGH[[str(x)+'_ecc' for _,x in enumerate(ecc)]].mean(axis=1).values)
        
        # FIXATIONS per ECC or SET SIZE
        ## fixation search 
        df_FIX_LOW, _ = density_meanfix(df_vs,eye_data,is_crowded,
                                        sum_measures['all_subs'][j],density='low')
        df_FIX_HIGH, _ = density_meanfix(df_vs,eye_data,is_crowded,
                                         sum_measures['all_subs'][j],density='high')
        # append matrix of combo 
        test_df_fix_LOW = test_df_fix_LOW.append(df_FIX_LOW,ignore_index=True)
        test_df_fix_HIGH = test_df_fix_HIGH.append(df_FIX_HIGH,ignore_index=True)
        
        test_fix_ecc_vs_LOW.append(np.array([np.nanmean(df_FIX_LOW[str(x)+'_ecc'].values) for _,x in enumerate(ecc)]))
        test_fix_ecc_vs_HIGH.append(np.array([np.nanmean(df_FIX_HIGH[str(x)+'_ecc'].values) for _,x in enumerate(ecc)]))

        test_fix_set_vs_LOW.append(df_FIX_LOW[[str(x)+'_ecc' for _,x in enumerate(ecc)]].mean(axis=1).values)
        test_fix_set_vs_HIGH.append(df_FIX_HIGH[[str(x)+'_ecc' for _,x in enumerate(ecc)]].mean(axis=1).values)
        
        # append number of trials for each, want to check for discrepencies between conditions
        test_df_trialnum_LOW = test_df_trialnum_LOW.append(df_trial_LOW,ignore_index=True)
        test_df_trialnum_HIGH = test_df_trialnum_HIGH.append(df_trials_HIGH,ignore_index=True)
        
        
# PLOTS

fig, axis = plt.subplots(1,1,figsize=(15,7.5),dpi=100)

plt.hist(test_iscrowded,bins=10,color='pink')
axis.set_xlabel('% of crowded trials',fontsize=14)

axis.set_title('Histogram of percentage of crowded trials per participant')
fig.savefig(os.path.join(plot_dir,'distribution_crowded_trials'), dpi=100)


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
plt.savefig(os.path.join(plot_dir,'density_search_ecc_RT_regression.svg'), dpi=100,bbox_inches = 'tight')
 

# RT VS SET SIZE

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

rt_set_vs4plot = pd.DataFrame([])

for k in range(len(params['set_size'])):
    rt_set_vs4plot = rt_set_vs4plot.append(pd.DataFrame({'RT_LOW': np.array(test_rt_set_vs_LOW).T[k],
                                                         'RT_HIGH': np.array(test_rt_set_vs_HIGH).T[k],
                                                     'set':np.tile(params['set_size'][k],len(test_subs))}))

sns.regplot(x=rt_set_vs4plot['set'],y=rt_set_vs4plot['RT_LOW'],color='blue', marker='.',label='LOW')
sns.regplot(x=rt_set_vs4plot['set'],y=rt_set_vs4plot['RT_HIGH'],color='red', marker='+',label='HIGH')
    
ax = plt.gca()
ax.set(xlabel='set size', ylabel='RT [s]')
ax.set_title('set size vs RT %d subs'%len(test_subs))
ax.legend()
plt.savefig(os.path.join(plot_dir,'density_search_setsize_RT_regression.svg'), dpi=100,bbox_inches = 'tight')
 
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
plt.savefig(os.path.join(plot_dir,'density_search_ecc_numfix_regression.svg'), dpi=100,bbox_inches = 'tight')
 

# NUMBER OF FIXATIONS VS SET SIZE
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

fix_set_vs4plot = pd.DataFrame([])

for k in range(len(params['set_size'])):
    fix_set_vs4plot = fix_set_vs4plot.append(pd.DataFrame({'fix_LOW': np.array(test_fix_set_vs_LOW).T[k],
                                                           'fix_HIGH': np.array(test_fix_set_vs_HIGH).T[k],
                                                     'set':np.tile(params['set_size'][k],len(test_subs))}))

    
sns.regplot(x=fix_set_vs4plot['set'],y=fix_set_vs4plot['fix_LOW'],color='blue', marker='.',label='LOW')
sns.regplot(x=fix_set_vs4plot['set'],y=fix_set_vs4plot['fix_HIGH'],color='red', marker='+',label='HIGH')
    
ax = plt.gca()

ax.set(xlabel='set size', ylabel='# fixations')
ax.set_title('set size vs number fixations %d subs'%len(test_subs))
ax.axes.set_ylim(0,)
ax.legend()
plt.savefig(os.path.join(plot_dir,'density_search_setsize_fix_regression.svg'), dpi=100,bbox_inches = 'tight')


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
plt.savefig(os.path.join(plot_dir,'density_search_ecc_onobjectfix_regression.svg'), dpi=100,bbox_inches = 'tight')


# PERCENTAGE OF ON OBJECT FIXATIONS VS SET SIZE
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

onobj_set_vs4plot = pd.DataFrame([])

for k in range(len(params['set_size'])):
    onobj_set_vs4plot = onobj_set_vs4plot.append(pd.DataFrame({'onobj_LOW': np.array(test_onobjfix_set_vs_LOW).T[k]*100,
                                                               'onobj_HIGH': np.array(test_onobjfix_set_vs_HIGH).T[k]*100,
                                                     'set':np.tile(params['set_size'][k],len(test_subs))}))

sns.regplot(x=onobj_set_vs4plot['set'],y=onobj_set_vs4plot['onobj_LOW'],color='blue', marker='.',label='LOW')
sns.regplot(x=onobj_set_vs4plot['set'],y=onobj_set_vs4plot['onobj_HIGH'],color='red', marker='+',label='HIGH')    
    
ax = plt.gca()
ax.set_title('set size vs on-object fixations %d subs'%len(test_subs))
ax.set(xlabel='set size', ylabel='On object fixation [%]')
ax.axes.set_ylim(0,)
ax.legend()
plt.savefig(os.path.join(plot_dir,'density_search_setsize_onobjectfix_regression.svg'), dpi=100,bbox_inches = 'tight')
 

# Correlations between tasks

#The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. 
#Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.

###### CORRELATIONS RELATIVE TO ECC #######


# CS VS RT ACROSS ECC
print('\ncomparing mean CS and mean RT in VS across ecc\n')

density_plot_correlation(np.nanmean(test_rt_ecc_vs_LOW,axis=-1),np.nanmean(test_rt_ecc_vs_HIGH,axis=-1),
                         np.nanmean(test_all_cs,axis=-1),
                         'RT [s]','CS','CS vs RT across ecc',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsRT_across-ecc.svg'),p_value=p_value)


# CS VS RT PER ECC
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean RT in VS for ecc %d\n'%ecc[i])
    
    density_plot_correlation(np.array(test_rt_ecc_vs_LOW).T[i],np.array(test_rt_ecc_vs_HIGH).T[i],
                         np.array(test_all_cs).T[i],
                         'RT [s]','CS','CS vs RT at %d ecc'%ecc[i],
                          os.path.join(plot_dir,'density-HIGH_LOW_CSvsRT_%d-ecc.svg'%ecc[i]),p_value=p_value)


# CS VS NUMBER FIXATIONS ACROSS ECC
print('\ncomparing mean CS and mean number Fixations in VS across ecc \n')

density_plot_correlation(np.nanmean(test_fix_ecc_vs_LOW,axis=-1),np.nanmean(test_fix_ecc_vs_HIGH,axis=-1),
                         np.nanmean(test_all_cs,axis=-1),
                         '# Fixations','CS','CS vs #Fix across ecc',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsFix_across-ecc.svg'),p_value=p_value)


# CS VS NUMBER FIXATIONS PER ECC
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean Number Fixations in VS for ecc %d\n'%ecc[i])
    
    density_plot_correlation(np.array(test_fix_ecc_vs_LOW).T[i],np.array(test_fix_ecc_vs_HIGH).T[i],
                         np.array(test_all_cs).T[i],
                         '# Fixations','CS','CS vs #Fix at %d ecc'%ecc[i],
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsFix_%d-ecc.svg'%ecc[i]),p_value=p_value)


# CS VS On-object FIXATIONS ACROSS ECC
print('\ncomparing mean CS and mean percentage On-Object Fixations in VS across ecc \n')

density_plot_correlation(np.nanmean(test_onobjfix_ecc_vs_LOW,axis=-1)*100,np.nanmean(test_onobjfix_ecc_vs_HIGH,axis=-1)*100,
                         np.nanmean(test_all_cs,axis=-1),
                         'On-obj Fixations [%]','CS','CS vs percentage Onobj across ecc',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsOnobjFix_across-ecc.svg'),p_value=p_value)


# CS VS On-object FIXATIONS PER ECC
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean percentage On-Object Fixations in VS for ecc %d\n'%ecc[i])
    
    density_plot_correlation(np.array(test_onobjfix_ecc_vs_LOW).T[i]*100,np.array(test_onobjfix_ecc_vs_HIGH).T[i]*100,
                         np.array(test_all_cs).T[i],
                         'On-obj Fixations [%]','CS','CS vs percentage Onobj at %d ecc'%ecc[i],
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsOnobjFix_%d-ecc.svg'%ecc[i]),p_value=p_value)


###### CORRELATIONS RELATIVE TO SET SIZE #######

# CS VS RT ACROSS SET SIZE
print('\ncomparing mean CS and mean RT in VS across set size \n')

density_plot_correlation(np.nanmean(test_rt_set_vs_LOW,axis=-1),np.nanmean(test_rt_set_vs_HIGH,axis=-1),
                         np.nanmean(test_all_cs,axis=-1),
                         'RT [s]','CS','CS vs RT across set size',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsRT_across-set.svg'),p_value=p_value)


# CS VS RT PER SET SIZE
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean RT in VS for set size %d\n'%params['set_size'][i])

    density_plot_correlation(np.array(test_rt_set_vs_LOW).T[i],np.array(test_rt_set_vs_HIGH).T[i],
                         np.array(test_all_cs).T[i],
                         'RT [s]','CS','CS vs RT at %d set size'%params['set_size'][i],
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsRT_%d-set.svg'%params['set_size'][i]),p_value=p_value)


# CS VS NUMBER FIXATIONS ACROSS SET SIZE
print('\ncomparing mean CS and mean number Fixations in VS across set size \n')

density_plot_correlation(np.nanmean(test_fix_set_vs_LOW,axis=-1),np.nanmean(test_fix_set_vs_HIGH,axis=-1),
                         np.nanmean(test_all_cs,axis=-1),
                         '# Fixations','CS','CS vs #Fix across set size',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsFix_across-set.svg'),p_value=p_value)


# CS VS NUMBER FIXATIONS PER SET SIZE
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean Number Fixations in VS for set size %d\n'%params['set_size'][i])

    density_plot_correlation(np.array(test_fix_set_vs_LOW).T[i],np.array(test_fix_set_vs_HIGH).T[i],
                         np.array(test_all_cs).T[i],
                         '# Fixations','CS','CS vs #Fix at %d set size'%params['set_size'][i],
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsFix_%d-set.svg'%params['set_size'][i]),p_value=p_value)


# CS VS On-object FIXATIONS ACROSS SET SIZE
print('\ncomparing mean CS and mean percentage On-Object Fixations in VS across set size \n')

density_plot_correlation(np.nanmean(test_onobjfix_set_vs_LOW,axis=-1)*100,np.nanmean(test_onobjfix_set_vs_HIGH,axis=-1)*100,
                         np.nanmean(test_all_cs,axis=-1),
                         'On-obj Fixations [%]','CS','CS vs percentage Onobj across set size',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsOnobjFix_across-set.svg'),p_value=p_value)


# CS VS On-object FIXATIONS PER SET SIZE
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean percentage On-Object Fixations in VS for set size %d\n'%params['set_size'][i])

    density_plot_correlation(np.array(test_onobjfix_set_vs_LOW).T[i]*100,np.array(test_onobjfix_set_vs_HIGH).T[i]*100,
                         np.array(test_all_cs).T[i],
                         'On-obj Fixations [%]','CS','CS vs percentage Onobj at %d set size'%params['set_size'][i],
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsOnobjFix_%d-set.svg'%params['set_size'][i]),p_value=p_value)



# COMPUTE SLOPE VALUES

slope_RT_ecc_LOW = []
slope_RT_ecc_HIGH = []
slope_RT_set_LOW = []
slope_RT_set_HIGH = []

slope_fix_ecc_LOW = []
slope_fix_ecc_HIGH = []
slope_fix_set_LOW = []
slope_fix_set_HIGH = []

slope_onobj_ecc_LOW = []
slope_onobj_ecc_HIGH = []
slope_onobj_set_LOW = []
slope_onobj_set_HIGH = []

for k in range(len(test_subs)):
    # RT/ecc slope
    slope_RT_ecc_LOW.append(linregress(ecc,np.array(test_rt_ecc_vs_LOW)[k])[0])
    slope_RT_ecc_HIGH.append(linregress(ecc,np.array(test_rt_ecc_vs_HIGH)[k])[0])
    # RT/set slope
    slope_RT_set_LOW.append(linregress(params['set_size'],np.array(test_rt_set_vs_LOW)[k])[0])
    slope_RT_set_HIGH.append(linregress(params['set_size'],np.array(test_rt_set_vs_HIGH)[k])[0])
    # fixation/ecc slope
    slope_fix_ecc_LOW.append(linregress(ecc,np.array(test_fix_ecc_vs_LOW)[k])[0])
    slope_fix_ecc_HIGH.append(linregress(ecc,np.array(test_fix_ecc_vs_HIGH)[k])[0])
    # fixation/set slope
    slope_fix_set_LOW.append(linregress(params['set_size'],np.array(test_fix_set_vs_LOW)[k])[0])
    slope_fix_set_HIGH.append(linregress(params['set_size'],np.array(test_fix_set_vs_HIGH)[k])[0])
    # on-object fixation/ecc slope
    slope_onobj_ecc_LOW.append(linregress(ecc,np.array(test_onobjfix_ecc_vs_LOW)[k])[0])
    slope_onobj_ecc_HIGH.append(linregress(ecc,np.array(test_onobjfix_ecc_vs_HIGH)[k])[0])
    # on-object fixation/set slope
    slope_onobj_set_LOW.append(linregress(params['set_size'],np.array(test_onobjfix_set_vs_LOW)[k])[0])
    slope_onobj_set_HIGH.append(linregress(params['set_size'],np.array(test_onobjfix_set_vs_HIGH)[k])[0])


# CS vs RT/ECC SLOPE
print('\ncomparing mean CS and mean RT/ecc slope in VS \n')

density_plot_correlation(slope_RT_ecc_LOW,slope_RT_ecc_HIGH,
                         np.nanmean(test_all_cs,axis=-1),
                         'RT/ECC','CS','CS vs RT/ECC',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsRT_ECC_slope_across-set.svg'),p_value=p_value)


# CS vs RT/SET SLOPE
print('\ncomparing mean CS and mean RT/set slope in VS \n')

density_plot_correlation(slope_RT_set_LOW,slope_RT_set_HIGH,
                         np.nanmean(test_all_cs,axis=-1),
                         'RT/set','CS','CS vs RT/set',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsRT_set_slope_across-set.svg'),p_value=p_value)


# CS vs FIX/ECC SLOPE
print('\ncomparing mean CS and mean fix/ecc slope in VS \n')

density_plot_correlation(slope_fix_ecc_LOW,slope_fix_ecc_HIGH,
                         np.nanmean(test_all_cs,axis=-1),
                         'Fix/ECC','CS','CS vs Fix/ECC',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsFix_ECC_slope_across-set.svg'),p_value=p_value)


# CS vs Fix/SET SLOPE
print('\ncomparing mean CS and mean fix/set slope in VS \n')

density_plot_correlation(slope_fix_set_LOW,slope_fix_set_HIGH,
                         np.nanmean(test_all_cs,axis=-1),
                         'Fix/set','CS','CS vs Fix/set',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsFix_set_slope_across-set.svg'),p_value=p_value)


# CS vs on-object/ECC SLOPE
print('\ncomparing mean CS and mean on-object/ecc slope in VS \n')

density_plot_correlation(slope_onobj_ecc_LOW,slope_onobj_ecc_HIGH,
                         np.nanmean(test_all_cs,axis=-1),
                         'on-object/ECC','CS','CS vs on-object/ECC',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsOnobj_ECC_slope_across-set.svg'),p_value=p_value)



# CS vs on-object/SET SLOPE
print('\ncomparing mean CS and mean on-object/set slope in VS \n')

density_plot_correlation(slope_onobj_set_LOW,slope_onobj_set_HIGH,
                         np.nanmean(test_all_cs,axis=-1),
                         'on-object/set','CS','CS vs on-object/set',
                         os.path.join(plot_dir,'density-HIGH_LOW_CSvsOnobj_set_slope_across-set.svg'),p_value=p_value)



































