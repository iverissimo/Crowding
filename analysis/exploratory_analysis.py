
# exploratory analyses
# not originally included in the pre-registration

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

from scipy.stats import wilcoxon, kstest, spearmanr, linregress, friedmanchisquare,kruskal #pearsonr,  
from statsmodels.stats import weightstats
import seaborn as sns

# set font type for plots globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

import matplotlib.patches as mpatches

from statsmodels.stats.anova import AnovaRM

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
vs_exclusion_acc_set_thresh=params['cut_off_acc_set_vs'] # percentage of accurate trials for search (per ecc)


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
                                 cut_off_acc_set_vs = vs_exclusion_acc_set_thresh)

sum_measures = np.load(sum_file) # all relevant measures


# then take out excluded participants and save relevant info in new arrays
# for later analysis

# find index to take out from variables
exclusion_ind = [np.where(np.array(sum_measures['all_subs'])==np.array(sum_measures['excluded_sub'][i]))[0][0] for i in range(len(sum_measures['excluded_sub']))] 

counter = 0
test_subs = []
test_mean_cs = []

# data frames with interesting values divided
test_df_corr_RT_fix = pd.DataFrame(columns=['ecc','set_size','sub'])

for j in range(len(sum_measures['all_subs'])):
    
    if j == exclusion_ind[counter]: # if excluded, skip
        print('skipping sub %s'%sum_measures['all_subs'][j])
        if counter<len(exclusion_ind)-1:
            counter+=1   
    else:
        # load crowding data
        df_crwd = pd.read_csv(crwd_csv[j], sep='\t')
        # choose only actual pratice block (first block (first 144) where used to save training, doesn't count)
        df_crwd = df_crwd.loc[int(trials_block)::]

        # load behav data search
        df_vs = pd.read_csv(vs_csv[j], sep='\t')

        # load eye data search
        asccii_name = convert2asc(sum_measures['all_subs'][j],'search',output_vs)
        print('loading edf for sub-%s data'%sum_measures['all_subs'][j])
        eye_data = read_edf(asccii_name, 'start_trial', stop='stop_trial', debug=False)

        # save test sub identifier labels
        test_subs.append(sum_measures['all_subs'][j])
        
        # critical spacing 
        test_mean_cs.append(sum_measures['mean_cs'][j])
        
        # reunite RT and fixations for all trials
        df_RT_fix = df_all_trial_RT_fix(df_vs,eye_data,sum_measures['all_subs'][j])

        # dataframe to output values
        df_corr = pd.DataFrame(columns=['ecc','set_size','sub'])

        for k in range(len(df_RT_fix)):

            corr, pval = spearmanr(df_RT_fix['RT'][k],df_RT_fix['fix'][k])

            # compute correlations and save in data frame           
            df_corr = df_corr.append({'corr': corr,
                                    'p_val': pval,
                                    'ecc': df_RT_fix['ecc'][k], 
                                    'set_size': df_RT_fix['set_size'][k],
                                    'sub': sum_measures['all_subs'][j]},ignore_index=True)

        # also get a value across all trials combined 
        rt_append = []
        fix_append = []

        for k in range(len(df_RT_fix)):

            rt_append.append(df_RT_fix['RT'][k]) 
            fix_append.append(df_RT_fix['fix'][k]) 

        rt_append = np.concatenate(rt_append).ravel()
        fix_append = np.concatenate(fix_append).ravel()

        corr, pval = spearmanr(rt_append,fix_append)

        # compute correlations in data frame         
        df_corr = df_corr.append({'corr': corr,
                                'p_val': pval,
                                'ecc': 'across', 
                                'set_size': 'across',
                                'sub': sum_measures['all_subs'][j]},ignore_index=True)
        
        # save all corr df
        test_df_corr_RT_fix = test_df_corr_RT_fix.append(df_corr,ignore_index=True)


# dir to save extra plots, tryouts to not make a mess in output folder
test_plot_dir = os.path.join(plot_dir,'exploratory')
if not os.path.exists(test_plot_dir):
     os.makedirs(test_plot_dir)


df_across_corr = test_df_corr_RT_fix.loc[test_df_corr_RT_fix['ecc'] == 'across']
mean_cor = np.mean(df_across_corr['corr'].values)
std_cor = np.std(df_across_corr['corr'].values)

print('Average correlation across subjects is %.2f with std %.2f'%(mean_cor,std_cor))
print('Correlation range between %.2f and %.2f'%(np.min(df_across_corr['corr'].values),
                                                np.max(df_across_corr['corr'].values)))

corr,pval = plot_correlation(np.array(test_mean_cs).T,df_across_corr['corr'].values,
                    'CS',r'RT-Fixations $\rho$','corr vs CS accross ecc and items',
                     os.path.join(test_plot_dir,'RT-Fix_corr_vs_CS_across_trials.svg'),
                                     p_value=p_value,y_lim = [0.6,1])











