
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
import itertools

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
test_df_fix_dist = pd.DataFrame(columns=['ecc','set_size','sub'])
test_df_fix_dur = pd.DataFrame(columns=['ecc','set_size','sub'])
test_df_fix_times_dur = pd.DataFrame(columns=['ecc','set_size','sub'])


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

        # save average number of fixations times x duration
        # note that sum of empty list is 0, hence exception
        df_fix_dur = df_duration_fixations(df_vs,eye_data,sum_measures['all_subs'][j])
        test_df_fix_times_dur = test_df_fix_times_dur.append(pd.DataFrame({'fix_times_dur': pd.DataFrame([np.mean([np.sum(x) for _,x in enumerate(df_fix_dur['fix_duration'].iloc[ind]) if np.sum(x)!=0])] for ind in range(len(df_fix_dur['fix_duration'])))[0],
                                                            'ecc': df_fix_dur['ecc'], 
                                                            'set_size': df_fix_dur['set_size'],
                                                            'sub': np.tile(sum_measures['all_subs'][j],len(df_fix_dur['ecc']))}),
                                                            ignore_index=True)
       

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

        # save all distances between consecutive fixations
        test_df_fix_dist = test_df_fix_dist.append(df_distance_fixations(df_vs,eye_data,sum_measures['all_subs'][j]),ignore_index=True)
        # same for durations
        test_df_fix_dur = test_df_fix_dur.append(df_fix_dur,ignore_index=True)


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


# compute mean distance between fixations in pixels, for each condition combination
# and subject, to visualize in a boxplot 

new_df4plot = pd.DataFrame(columns=['ecc','set_size','sub'])

for _,sj in enumerate(test_subs):
    
    dist_counter = 0
    
    for _,s in enumerate(params['set_size']):
        for _,e in enumerate(ecc):
            
            list_of_lists = test_df_fix_dist.loc[(test_df_fix_dist['ecc']==e)&\
                     (test_df_fix_dist['set_size']==s) &\
                    (test_df_fix_dist['sub']== sj)]['fix_distance'].values[0]
            flattened_list = [y for x in list_of_lists for y in x]
            
            dist_counter += len(flattened_list)
            
            # compute mean number of fixations and save in data frame           
            new_df4plot = new_df4plot.append({'fix_distance': np.nanmean(flattened_list),
                                    'ecc': e, 
                                    'set_size': s,
                                    'sub': sj},ignore_index=True)
    print(dist_counter)


#sns.boxplot(x="ecc", y="fix_distance", hue="set_size", data=new_df4plot, palette="Set3") 

# MAKE ANOVA TO SEE IF EFFECT OF ECC OR SET SIZE
aovrm2way = AnovaRM(new_df4plot, depvar='fix_distance', subject='sub', within=['set_size', 'ecc'])
res2way = aovrm2way.fit()

print(res2way)

# save it
res2way.anova_table.to_csv(os.path.join(test_plot_dir,'FixDIST_ANOVA.csv'))
# This implementation currently only supports fully balanced designs."
# a couple of participants have conditions with nan, because they didnt 2 two fixations in the condition combination
# so can't compute anova

# For all ecc do separate violin plots
# colors a bit off between legend and violin, so corrected for in inkscape
for k,_ in enumerate(ecc):
    colors_ecc = np.array([['#ff8c8c','#e31a1c','#870000'],
                           ['#faad75','#ff6b00','#cc5702'],
                       ['#fff5a3','#eecb5f','#f4b900']])#['#ffea80','#fff200','#dbbe00']])

    df4plot_vs_FIXdist = new_df4plot.loc[new_df4plot['ecc']==ecc[k]]

    fig = plt.figure(num=None, figsize=(7.5,7.5), dpi=100, facecolor='w', edgecolor='k')
    v1 = sns.violinplot(x='ecc', hue='set_size', y='fix_distance', data=df4plot_vs_FIXdist,
                  cut=0, inner='box', palette=colors_ecc[k],linewidth=3)

    plt.legend().remove()
    plt.xticks([], [])
    #plt.xticks([0,1,2], ('5', '15', '30'))

    v1.set(xlabel=None)
    v1.set(ylabel=None)

    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    if k==0:
        plt.ylabel('Distance between Fixations [pixels]',fontsize=18,labelpad=10)
        plt.xlabel('Set Size [items]',fontsize=18,labelpad=35)
    plt.title('%d dva'%ecc[k],fontsize=22,pad=10)

    plt.ylim(50,450)
    plt.savefig(os.path.join(test_plot_dir,'search_ecc_FIX_dist_violin_%decc.svg'%ecc[k]), dpi=100,bbox_inches = 'tight')
    
## correlate mean CS with mean distance between fixations per condition

for _,s in enumerate(params['set_size']): # loop over set size
    
    for _,e in enumerate(ecc): # loop over eccentricity
        
        df_trim = new_df4plot.loc[(new_df4plot['set_size'] == s)&(new_df4plot['ecc'] == e)]
        
        corr,pval = plot_correlation(test_mean_cs,df_trim['fix_distance'].values,
                    'CS','Distance between Fixations [pixel]','CS vs Fixdistance for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(test_plot_dir,'CSvsDistFIX_%s-ecc_%s-set.svg'%(str(e),str(s))),
                                     p_value=p_value, y_lim=[50,450])
        

# now do same but across trials 
#(so aggregate all values for a subject, and compute mean overall distance between fixations of that sub)
# maybe more sensible since not that many fixations

aggregate_df4plot = pd.DataFrame(columns=['sub'])

for _,sj in enumerate(test_subs):
    
    dist_counter = 0
    
    list_of_lists = [y for x in test_df_fix_dist.loc[(test_df_fix_dist['sub']== sj)]['fix_distance'].values for y in x]
    flattened_list = [y for x in list_of_lists for y in x]

    dist_counter += len(flattened_list)

    # compute mean number of fixations and save in data frame           
    aggregate_df4plot = aggregate_df4plot.append({'fix_distance': np.nanmean(flattened_list),
                            'sub': sj},ignore_index=True)
    print(dist_counter)


corr,pval = plot_correlation(test_mean_cs,aggregate_df4plot['fix_distance'].values,
                    'CS','Distance between Fixations [pixel]','CS vs Fixdistance all',
                     os.path.join(test_plot_dir,'CSvsDistFIX_all.svg'),
                                     p_value=p_value, y_lim=[50,450])
        
# compute mean duration of each fixations in seconds, for each condition combination
# and subject, to visualize in a boxplot 

dur_df4plot = pd.DataFrame(columns=['ecc','set_size','sub'])

for _,sj in enumerate(test_subs):
    
    dist_counter = 0
    
    for _,s in enumerate(params['set_size']):
        for _,e in enumerate(ecc):
            
            list_of_lists = test_df_fix_dur.loc[(test_df_fix_dur['ecc']==e)&\
                     (test_df_fix_dur['set_size']==s) &\
                    (test_df_fix_dur['sub']== sj)]['fix_duration'].values[0]
            flattened_list = [y for x in list_of_lists for y in x]
            
            dist_counter += len(flattened_list)
            
            # compute mean number of fixations and save in data frame           
            dur_df4plot = dur_df4plot.append({'fix_duration': np.nanmean(flattened_list),
                                    'ecc': e, 
                                    'set_size': s,
                                    'sub': sj},ignore_index=True)
    print(dist_counter)


#sns.boxplot(x="ecc", y="fix_duration", hue="set_size", data=dur_df4plot, palette="Set3") 

# MAKE ANOVA TO SEE IF EFFECT OF ECC OR SET SIZE
aovrm2way = AnovaRM(dur_df4plot, depvar='fix_duration', subject='sub', within=['set_size', 'ecc'])
res2way = aovrm2way.fit()

print(res2way)

# save it
res2way.anova_table.to_csv(os.path.join(test_plot_dir,'FixDUR_ANOVA.csv'))

# For all ecc do separate violin plots
# colors a bit off between legend and violin, so corrected for in inkscape
for k,_ in enumerate(ecc):
    colors_ecc = np.array([['#ff8c8c','#e31a1c','#870000'],
                           ['#faad75','#ff6b00','#cc5702'],
                       ['#fff5a3','#eecb5f','#f4b900']])#['#ffea80','#fff200','#dbbe00']])

    df4plot_vs_FIXdur = dur_df4plot.loc[dur_df4plot['ecc']==ecc[k]]

    fig = plt.figure(num=None, figsize=(7.5,7.5), dpi=100, facecolor='w', edgecolor='k')
    v1 = sns.violinplot(x='ecc', hue='set_size', y='fix_duration', data=df4plot_vs_FIXdur,
                  cut=0, inner='box', palette=colors_ecc[k],linewidth=3)

    plt.legend().remove()
    plt.xticks([], [])
    #plt.xticks([0,1,2], ('5', '15', '30'))

    v1.set(xlabel=None)
    v1.set(ylabel=None)

    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    if k==0:
        plt.ylabel('Fixation Duration [s]',fontsize=18,labelpad=10)
        plt.xlabel('Set Size [items]',fontsize=18,labelpad=35)
    plt.title('%d dva'%ecc[k],fontsize=22,pad=10)

    plt.ylim(.125,.350)
    plt.savefig(os.path.join(test_plot_dir,'search_ecc_FIX_dur_violin_%decc.svg'%ecc[k]), dpi=100,bbox_inches = 'tight')
    

## correlate mean CS with mean fixations duration per condition

for _,s in enumerate(params['set_size']): # loop over set size
    
    for _,e in enumerate(ecc): # loop over eccentricity
        
        df_trim = dur_df4plot.loc[(dur_df4plot['set_size'] == s)&(dur_df4plot['ecc'] == e)]
        
        corr,pval = plot_correlation(test_mean_cs,df_trim['fix_duration'].values,
                    'CS','Fixation duration [s]','CS vs Fix duration for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(test_plot_dir,'CSvsDurFIX_%s-ecc_%s-set.svg'%(str(e),str(s))),
                                     p_value=p_value, y_lim=[.125,.350])
        

## TO ADD
# compute average #fix x fix-duration and correlate to CS

for _,s in enumerate(params['set_size']): # loop over set size
    
    for _,e in enumerate(ecc): # loop over eccentricity
        
        df_trim = test_df_fix_times_dur.loc[(test_df_fix_times_dur['set_size'] == s)&(test_df_fix_times_dur['ecc'] == e)]
        
        corr,pval = plot_correlation(test_mean_cs,df_trim['fix_times_dur'].values,
                    'CS','# Fixations * duration [s]','CS vs Fix * duration for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(test_plot_dir,'CSvsFIX_times_Dur_%s-ecc_%s-set.svg'%(str(e),str(s))),
                                     p_value=p_value, y_lim=[.17,2.5])
        






