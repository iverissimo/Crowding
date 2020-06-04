
# full analysis pipeline
# makes some plots
# excludes participants that don't abid by exclusion criteria
# computes some stats on final sample set


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
    sum_file = exclude_subs(crwd_csv,vs_csv,plot_dir,
                                 trials_block = trials_block,
                                 miss_trials = miss_exclusion_thresh,
                                 acc_cut_off_crwd = cut_off,
                                 ecc = ecc,
                                 num_cs_trials = last_trials,
                                 cut_off_acc_vs = vs_exclusion_acc_thresh,
                                 cut_off_acc_ecc_vs = vs_exclusion_acc_ecc_thresh)
else:
    sum_measures = np.load(sum_file) # all relevant measures
  
# plot staircases
# just to see what's going on
out_dir = os.path.join(plot_dir,'staircases') # save here
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
for ind,subs in enumerate(sum_measures['all_subs']): 
    
    # plot staricases divided by ecc
    fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    plt.plot(np.array(sum_measures['all_tfr'][ind]).T)
    plt.ylabel("Target-flanker distance (ratio of ecc)",fontsize=18)
    plt.xlabel("# Trial",fontsize=18)
    plt.title("Target-Flanker ratio per ecc",fontsize=18)
    plt.legend(ecc)

    fig.savefig(os.path.join(out_dir,'crowding_TFR-ecc_sub-{sj}.svg'.format(sj=subs)), dpi=100)

    # plot staircase of whole experiment
    # load crowding csv for sub
    data_sub = pd.read_csv(crwd_csv[ind], sep='\t')
    # choose only actual pratice block (first block (first 144) where used to save training, doesn't count)
    data_sub = data_sub.loc[int(trials_block)::]

    fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    plt.plot(data_sub['target_flank_ratio'].values,alpha=0.8)
    plt.hlines(y=sum_measures['mean_cs'][ind], color='r', linestyle='--',linewidth=2,xmin=trials-last_trials,xmax=trials)
    plt.xlim(0,trials)

    plt.ylabel("Target-flanker distance (ratio of ecc)",fontsize=18)
    plt.xlabel("# Trial",fontsize=18)
    plt.title("Target-Flanker ratio all trials",fontsize=18)
    #plt.show()

    fig.savefig(os.path.join(out_dir,'crowding_TFR_sub-{sj}.svg'.format(sj=subs)), dpi=100)
    
    
# then take out excluded participants and save relevant info in new arrays
# for later analysis

# find index to take out from variables
exclusion_ind = [np.where(np.array(sum_measures['all_subs'])==np.array(sum_measures['excluded_sub'][i]))[0][0] for i in range(len(sum_measures['excluded_sub']))] 

counter = 0
test_subs = []
test_acc_fl = []
test_acc_nofl = []
test_all_cs = []
test_mean_cs = []
test_acc_vs_ecc = []
test_rt_ecc_vs = []
test_rt_set_vs = []
test_fix_ecc_vs = []
test_fix_set_vs = []

# data frames with interesting values divided
test_df_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
test_df_fix = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])


for j in range(len(sum_measures['all_subs'])):
    
    if j == exclusion_ind[counter]: # if excluded, skip
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

        # save test sub identifier labels
        test_subs.append(sum_measures['all_subs'][j])

        #accuracy crowding
        test_acc_fl.append(sum_measures['acc_fl'][j])
        test_acc_nofl.append(sum_measures['acc_nofl'][j])
        # critical spacing
        test_all_cs.append(sum_measures['all_cs'][j])
        test_mean_cs.append(sum_measures['mean_cs'][j])

        # accuracy search
        test_acc_vs_ecc.append(sum_measures['acc_vs_ecc'][j])

        ## reaction time search 
        df_RT, _ = mean_RT_set_ecc_combo(df_vs,sum_measures['all_subs'][j])
        # append matrix of combo 
        test_df_RT = test_df_RT.append(df_RT,ignore_index=True)
        # save values per ecc
        test_rt_ecc_vs.append(np.array([np.mean(df_RT[str(x)+'_ecc'].values) for _,x in enumerate(ecc)]))
        # save values per set size
        test_rt_set_vs.append(df_RT[[str(x)+'_ecc' for _,x in enumerate(ecc)]].mean(axis=1).values)

        ## number of fixations in search
        df_fix, _ = mean_fix_set_ecc_combo(df_vs,eye_data,sum_measures['all_subs'][j])
        # append matrix of combo 
        test_df_fix = test_df_fix.append(df_fix,ignore_index=True)
        # save values per ecc
        test_fix_ecc_vs.append(np.array([np.mean(df_fix[str(x)+'_ecc'].values) for _,x in enumerate(ecc)]))
        # save values per set size
        test_fix_set_vs.append(df_fix[[str(x)+'_ecc' for _,x in enumerate(ecc)]].mean(axis=1).values)



# PLOTS

## CROWDING ####

# ACCURACY CROWDING WITH AND WITHOUT FLANKERS
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
plt.hist(np.array(test_acc_fl))
plt.title('Accuracy with flankers')
plt.legend(['4°','8°','12°'])
plt.xlim(0.5,1)

plt.subplot(1,2,2)
plt.hist(np.array(test_acc_nofl))
plt.title('Accuracy without flankers')
plt.legend(['4°','8°','12°'])
plt.xlim(0.5,1)

fig.savefig(os.path.join(plot_dir,'crowding_accuracy_hist.svg'), dpi=100)


# SAME BUT WITH BOXPLOTS, AND TESTING DISTRIBUTION DIFFERENCE

# wilcoxon test
# non-parametric statistical hypothesis test used to compare two matched samples
# to assess whether their population mean ranks differ
pval_acc_crwd = wilcoxon(np.mean(np.array(test_acc_fl),axis=-1), np.mean(np.array(test_acc_nofl),axis=-1))[-1]

crwd_acc4plot = pd.DataFrame(data=np.array([np.mean(np.array(test_acc_fl),axis=-1),
                                np.mean(np.array(test_acc_nofl),axis=-1)]).T, 
                             columns = ['mean_acc_fl','mean_acc_nofl'])

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

sns.boxplot(x='variable',y='value', data=pd.melt(crwd_acc4plot))

y, h, col = 1, .025, 'k'
plt.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text(.5, y+h+.01, 'p-val = %.3f'%pval_acc_crwd, ha='center', va='bottom', color=col)
plt.ylim(0.5, 1.1)
plt.ylabel('Accuracy')
plt.xlabel('')
plt.title('Accuracy crowding (with and without flankers)')

if pval_acc_crwd<p_value:
    print('Significant difference in accuracy distribution \nbetween flanker and no flanker \nacross ecc (p-val = %.6f)'%pval_acc_crwd)

fig.savefig(os.path.join(plot_dir,'crowding_mean_accuracy_boxplot_wilcoxtest.svg'), dpi=100)


# SAME BUT PER ECC

for g,_ in enumerate(ecc):
    pval_acc_crwd = wilcoxon(np.array(test_acc_fl)[...,g], np.array(test_acc_nofl)[...,g])[-1]
    
    crwd_acc4plot = pd.DataFrame(data=np.array([np.array(test_acc_fl)[...,g],
                                    np.array(test_acc_nofl)[...,g]]).T, 
                                 columns = ['mean_acc_fl','mean_acc_nofl'])

    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    sns.boxplot(x='variable',y='value', data=pd.melt(crwd_acc4plot))

    y, h, col = 1, .025, 'k'
    plt.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text(.5, y+h+.01, 'p-val = %.6f'%pval_acc_crwd, ha='center', va='bottom', color=col)
    plt.ylim(0.5, 1.1)
    plt.title('%d ecc accuracy crowding (with and without flankers)'%ecc[g])
    plt.ylabel('Accuracy')
    plt.xlabel('')
    
    if pval_acc_crwd<p_value:
        print('Significant difference in accuracy distribution \nbetween flanker and no flanker \nfor %d ecc (p-val = %.6f)'%(ecc[g],pval_acc_crwd))
    
    fig.savefig(os.path.join(plot_dir,'crowding_%decc_accuracy_boxplot_wilcoxtest.svg'%(ecc[g])), dpi=100)


# MEAN CRITICAL SPACING, WEIGHTED BY ACCURACY, PER ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
weighted_cs_mean = []
weighted_cs_std = []

crwd_df4plot = pd.DataFrame([])

for w in range(len(ecc)):
    cs_ecc = [test_all_cs[val][w] for val in range(len(test_all_cs))]
    acc_ecc = [test_acc_fl[val][w] for val in range(len(test_acc_fl))]
    
    weighted_cs_mean.append(weightstats.DescrStatsW(cs_ecc,weights=acc_ecc).mean)
    weighted_cs_std.append(weightstats.DescrStatsW(cs_ecc,weights=acc_ecc).std_mean)
    
    crwd_df4plot = crwd_df4plot.append(pd.DataFrame({'ecc': np.tile(ecc[w],len(cs_ecc)),
                                  'cs':cs_ecc,
                                   'acc':acc_ecc,
                                    'sub':np.array(test_subs)}),ignore_index=True)

plt.errorbar(ecc, weighted_cs_mean, yerr=weighted_cs_std)
plt.ylabel('Critical spacing',fontsize=18)
plt.xlabel('ecc',fontsize=18)
plt.title('Mean critical spacing, weighted by accuracy',fontsize=18)
fig.savefig(os.path.join(plot_dir,'crowding_meanCS-weighted_errorbar.svg'), dpi=100)

# SHOW INDIVIDUAL SUB CS DISTRIBITUION PER ECC
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

with sns.color_palette("pastel", len(test_subs)):
    sns.catplot(x="ecc", y="cs", hue="sub", kind="point", data=crwd_df4plot)
plt.title('Critical Spacing, per ecc')
plt.savefig(os.path.join(plot_dir,'crowding_CS_ecc_individual.svg'), dpi=100)

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

sns.lineplot(x='ecc', y='cs', data=crwd_df4plot,estimator='mean')
sns.lineplot(x='ecc', y='cs', data=crwd_df4plot,
                   units='sub', estimator=None, lw=1,color='grey',alpha=0.2)

plt.title('Critical Spacing, per ecc')
fig.savefig(os.path.join(plot_dir,'crowding_CS_ecc_all.svg'), dpi=100)

# BOXPLOTS WITH CS PER ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
sns.boxplot(x='ecc', y='cs', data=crwd_df4plot)
sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25")
fig.savefig(os.path.join(plot_dir,'crowding_meanCS-ecc-weighted_boxplot.svg'), dpi=100)

# do Friedman to see if averages are different 
# The Friedman test tests the null hypothesis that repeated measurements of the same individuals have the same distribution. 
#It is often used to test for consistency among measurements obtained in different ways.

pfriedman = friedmanchisquare(np.array(test_all_cs)[...,0], np.array(test_all_cs)[...,1], np.array(test_all_cs)[...,2])[-1]
if pfriedman<p_value:
    print('CS between ecc is different, friedman with p-value = %.6f'%pfriedman)
else:
    print('CS between ecc is the same')

# do post hoc wilcoxon to test differences between groups
# pairs to compare
ecc_compare = np.array(([4,8],[8,12],[4,12]))

for d in range(len(ecc_compare)):
    pval_wilcox = wilcoxon(np.array(test_all_cs)[...,np.where(np.array(ecc)==ecc_compare[d][0])[0][0]], 
                           np.array(test_all_cs)[...,np.where(np.array(ecc)==ecc_compare[d][1])[0][0]])[-1]
    
    if pval_wilcox<(p_value/3): # bonferroni correction of p-value, right?
        print('SIGNIFICANT difference (p-val %.6f) in CS of ecc pair%s'%(pval_wilcox,str(ecc_compare[d])))

# kolmogorov smir test to see if sample of mean cs (across ecc)
# follows normal distribution - if p<0.05 not normally distributed
pval_crwd_norm = kstest(test_mean_cs, 'norm')[-1]
if pval_crwd_norm<p_value:
    print('mean critical spacing values, across ecc NOT normally distributed (KS with p-value=%.6f)'%pval_crwd_norm)
else:
    print('mean critical spacing values, across ecc normally distributed (KS with p-value=%.6f)'%pval_crwd_norm)


print('Now look at visual search data')

# VISUAL SEARCH

# ACCURACY VISUAL SEARCH
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(1,2,1)
plt.hist(np.array(test_acc_vs_ecc))
plt.title('Accuracy visual search')
plt.legend(['4°','8°','12°'])
plt.xlim(0.85,1)

plt.subplot(1,2,2)
plt.hist(np.array(test_rt_ecc_vs))
plt.title('Reaction times visual search')
plt.legend(['4°','8°','12°'])
#plt.xlim(0.2,0.8)

fig.savefig(os.path.join(plot_dir,'search_accuracy_RT_hist.svg'), dpi=100)


# RT VS ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

rt_ecc_vs4plot = pd.DataFrame([])

for k in range(len(ecc)):
    rt_ecc_vs4plot = rt_ecc_vs4plot.append(pd.DataFrame({'RT': np.array(test_rt_ecc_vs).T[k],
                                                        'ecc':np.tile(ecc[k],len(test_subs)),
                                                        'sub':np.array(test_subs)}))
sns.lineplot(x='ecc', y='RT',data=rt_ecc_vs4plot,
                   units='sub', estimator=None, lw=1,color='grey',alpha=0.1)
ax = sns.lineplot(x='ecc', y='RT',data=rt_ecc_vs4plot,estimator='mean')

ax.set(xlabel='eccentricity [dva]', ylabel='RT [s]')
ax = plt.gca()
ax.set_title('ecc vs RT %d subs'%len(test_subs))
plt.savefig(os.path.join(plot_dir,'search_ecc_RT_regression.svg'), dpi=100,bbox_inches = 'tight')

# dividing by set sizes
df4plot_RT_set = test_df_RT.drop(columns=['sub'])
df4plot_RT_set = pd.melt(df4plot_RT_set, 'set_size', var_name='Target eccentricity [dva]', value_name='RT [s]')

fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
sns.boxplot(x='Target eccentricity [dva]', hue='set_size', y='RT [s]', data=df4plot_RT_set)

ax = plt.gca()
ax.set_title('ecc vs RT %d subs'%len(test_subs))
plt.savefig(os.path.join(plot_dir,'search_ecc_RT_boxplots.svg'), dpi=100,bbox_inches = 'tight')


# RT VS SET SIZE
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

rt_set_vs4plot = pd.DataFrame([])

for k in range(len(params['set_size'])):
    rt_set_vs4plot = rt_set_vs4plot.append(pd.DataFrame({'RT': np.array(test_rt_set_vs).T[k],
                                                     'set':np.tile(params['set_size'][k],len(test_subs)),
                                                     'sub':np.array(test_subs)}))

sns.lineplot(x='set', y='RT',data=rt_set_vs4plot,
                   units='sub', estimator=None, lw=1,color='grey',alpha=0.1)
ax = sns.lineplot(x='set', y='RT',data=rt_set_vs4plot,estimator='mean')

ax.set(xlabel='set size', ylabel='RT [s]')
ax = plt.gca()
ax.set_title('set size vs RT %d subs'%len(test_subs))
plt.savefig(os.path.join(plot_dir,'search_setsize_RT_regression.svg'), dpi=100,bbox_inches = 'tight')
 

# INVERSE EFFICIENCY SCORE
inv_eff = np.array(test_rt_ecc_vs)/np.array(test_acc_vs_ecc)
inv_eff_mean = np.mean(inv_eff,axis=0)
print('mean inverse efficiency score, per ecc is %s'%str(inv_eff_mean))

#inv_eff_set = np.array(test_rt_set_vs)/np.array(test_acc_vs_set)
#inv_eff_set_mean = np.mean(inv_eff_set,axis=0)
#print('mean inverse efficiency score, per set size is %s'%str(inv_eff_set_mean))

# EYETRACKING FOR VISUAL SEARCH

# NUMBER OF FIXATIONS VS ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

fix_ecc_vs4plot = pd.DataFrame([])

for k in range(len(ecc)):
    fix_ecc_vs4plot = fix_ecc_vs4plot.append(pd.DataFrame({'fix': np.array(test_fix_ecc_vs).T[k],
                                                     'ecc':np.tile(ecc[k],len(test_subs)),
                                                     'sub':np.array(test_subs)}))

sns.lineplot(x='ecc', y='fix',data=fix_ecc_vs4plot,
                   units='sub', estimator=None, lw=1,color='grey',alpha=0.1)
ax = sns.lineplot(x='ecc', y='fix',data=fix_ecc_vs4plot,estimator='mean')

ax.set(xlabel='eccentricity [dva]', ylabel='# fixations')
ax = plt.gca()
ax.set_title('ecc vs number of fixations %d subs'%len(test_subs))
ax.axes.set_ylim(0,)
plt.savefig(os.path.join(plot_dir,'search_ecc_numfix_regression.svg'), dpi=100,bbox_inches = 'tight')

# dividing by set sizes
df4plot_fix_set = test_df_fix.drop(columns=['sub'])
df4plot_fix_set = pd.melt(df4plot_fix_set, 'set_size', var_name='Target eccentricity [dva]', value_name='# fixations')

fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
sns.boxplot(x='Target eccentricity [dva]', hue='set_size', y='# fixations', data=df4plot_fix_set)

ax = plt.gca()
ax.set_title('ecc vs #fixations %d subs'%len(test_subs))
plt.savefig(os.path.join(plot_dir,'search_ecc_numfix_boxplots.svg'), dpi=100,bbox_inches = 'tight')
 

# NUMBER OF FIXATIONS VS SET SIZE
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

fix_set_vs4plot = pd.DataFrame([])

for k in range(len(params['set_size'])):
    fix_set_vs4plot = fix_set_vs4plot.append(pd.DataFrame({'fix': np.array(test_fix_set_vs).T[k],
                                                     'set':np.tile(params['set_size'][k],len(test_subs)),
                                                     'sub':np.array(test_subs)}))

sns.lineplot(x='set', y='fix',data=fix_set_vs4plot,
                   units='sub', estimator=None, lw=1,color='grey',alpha=0.1)
ax = sns.lineplot(x='set', y='fix',data=fix_set_vs4plot,estimator='mean')

ax.set(xlabel='set size', ylabel='# fixations')
ax = plt.gca()
ax.set_title('set size vs number fixations %d subs'%len(test_subs))
plt.savefig(os.path.join(plot_dir,'search_setsize_fix_regression.svg'), dpi=100,bbox_inches = 'tight')



# Correlations between tasks

#The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. 
#Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.

###### CORRELATIONS RELATIVE TO ECC #######

# CS VS RT ACROSS ECC
print('\ncomparing mean CS and mean RT in VS across ecc \n')
plot_correlation(np.mean(test_rt_ecc_vs,axis=-1),test_mean_cs,
                'RT [s]','CS','CS vs RT across ecc',
                 os.path.join(plot_dir,'CSvsRT_across-ecc.svg'),p_value=p_value)


# CS VS RT PER ECC
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean RT in VS for ecc %d\n'%ecc[i])
    plot_correlation(np.array(test_rt_ecc_vs).T[i],test_mean_cs,
                    'RT [s]','CS','CS vs RT at %d ecc'%ecc[i],
                     os.path.join(plot_dir,'CSvsRT_%d-ecc.svg'%ecc[i]),p_value=p_value)


# CS VS NUMBER FIXATIONS ACROSS ECC
print('\ncomparing mean CS and mean number Fixations in VS across ecc \n')
plot_correlation(np.mean(test_fix_ecc_vs,axis=-1),test_mean_cs,
                '# Fixations','CS','CS vs #Fix across ecc',
                 os.path.join(plot_dir,'CSvsFix_across-ecc.svg'),p_value=p_value)


# CS VS NUMBER FIXATIONS PER ECC
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean Number Fixations in VS for ecc %d\n'%ecc[i])
    plot_correlation(np.array(test_fix_ecc_vs).T[i],test_mean_cs,
                    '# Fixations','CS','CS vs #Fix at %d ecc'%ecc[i],
                     os.path.join(plot_dir,'CSvsFix_%d-ecc.svg'%ecc[i]),p_value=p_value)



###### CORRELATIONS RELATIVE TO SET SIZE #######

# CS VS RT ACROSS SET SIZE
print('\ncomparing mean CS and mean RT in VS across set size \n')
plot_correlation(np.mean(test_rt_set_vs,axis=-1),test_mean_cs,
                'RT [s]','CS','CS vs RT across set size',
                 os.path.join(plot_dir,'CSvsRT_across-set.svg'),p_value=p_value)


# CS VS RT PER SET SIZE
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean RT in VS for set size %d\n'%params['set_size'][i])
    plot_correlation(np.array(test_rt_set_vs).T[i],test_mean_cs,
                    'RT [s]','CS','CS vs RT at %d set size'%params['set_size'][i],
                     os.path.join(plot_dir,'CSvsRT_%d-set.svg'%params['set_size'][i]),p_value=p_value)

    
# CS VS NUMBER FIXATIONS ACROSS SET SIZE
print('\ncomparing mean CS and mean number Fixations in VS across set size \n')
plot_correlation(np.mean(test_fix_set_vs,axis=-1),test_mean_cs,
                '# Fixations','CS','CS vs #Fix across set size',
                 os.path.join(plot_dir,'CSvsFix_across-set.svg'),p_value=p_value)


# CS VS NUMBER FIXATIONS PER SET SIZE
for i in range(len(np.array(test_all_cs).T)):
    print('\ncomparing mean CS and mean Number Fixations in VS for set size %d\n'%params['set_size'][i])
    plot_correlation(np.array(test_fix_set_vs).T[i],test_mean_cs,
                    '# Fixations','CS','CS vs #Fix at %d set size'%params['set_size'][i],
                     os.path.join(plot_dir,'CSvsFix_%d-set.svg'%params['set_size'][i]),p_value=p_value)


# save correlation and respective p-values
# per ecc and set
# making a correlation 3x3 matrix
# to see how this holds
 
## FOR RT
    
corr_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])
pval_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])

for _,s in enumerate(params['set_size']): # loop over set size
    
    df_trim = test_df_RT.loc[test_df_RT['set_size'] == s]
    df_trim = df_trim.drop(columns=['set_size'])
    
    for _,e in enumerate(ecc): # loop over eccentricity
        
        corr,pval = plot_correlation(df_trim[str(e)+'_ecc'].values,test_mean_cs,
                    'RT [s]','CS','CS vs RT for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(plot_dir,'CSvsRT_%s-ecc_%s-set.svg'%(str(e),str(s))),p_value=p_value)
        
        # save correlation values         
        corr_RT = corr_RT.append({str(e)+'_ecc': corr, 
                                'set_size': s},ignore_index=True)
        
        # save p-values of said correlations          
        pval_RT = pval_RT.append({str(e)+'_ecc': pval, 
                                'set_size': s},ignore_index=True)

# now reshape the dataframe (droping nans and making sure labels are ok)    
corr_RT = corr_RT.apply(lambda x: pd.Series(x.dropna().values))
corr_RT = corr_RT.dropna()

for i in corr_RT.index:
    corr_RT.at[i, 'set_size'] = params['set_size'][i]
    
# now reshape the dataframe (droping nans and making sure labels are ok)    
pval_RT = pval_RT.apply(lambda x: pd.Series(x.dropna().values))
pval_RT = pval_RT.dropna()

for i in pval_RT.index:
    pval_RT.at[i, 'set_size'] = params['set_size'][i]


##### MATRIX #######
# plot values in color (heat) matrix
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
ax = sns.heatmap(corr_RT[['4_ecc','8_ecc','12_ecc']],cmap='OrRd',
                 vmin = 0.25, vmax = 0.35, annot = True,
                xticklabels=ecc, yticklabels=params['set_size'])

ax.set_ylim([0,3]) # trick to make plot look decente, because matplotlib 3.1 broke seaborn heatmap
ax.invert_yaxis() # so set size goes from smaller (top) to biggest (bottom)
plt.title('RT vs CS correlation matrix')
plt.ylabel('Set Size')
plt.xlabel('Eccentricity [dva]')
plt.show()
fig.savefig(os.path.join(plot_dir,'correlation_matrix_ALL_RT.svg'), dpi=100,bbox_inches = 'tight')

# plot values in color (heat) matrix
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
ax = sns.heatmap(pval_RT[['4_ecc','8_ecc','12_ecc']],cmap='bwr_r',#'OrRd_r',
                 vmin = 0.001, vmax = 0.1, center=0.05, annot = True,
                xticklabels=ecc, yticklabels=params['set_size'])

ax.set_ylim([0,3]) # trick to make plot look decente, because matplotlib 3.1 broke seaborn heatmap
ax.invert_yaxis() # so set size goes from smaller (top) to biggest (bottom)
plt.title('RT vs CS correlation matrix - p-values')
plt.ylabel('Set Size')
plt.xlabel('Eccentricity [dva]')
plt.show()
fig.savefig(os.path.join(plot_dir,'correlation_matrix-pval_ALL_RT.svg'), dpi=100,bbox_inches = 'tight')


## FOR FIXATIONS
    
corr_fix = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])
pval_fix = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])

for _,s in enumerate(params['set_size']): # loop over set size
    
    df_trim = test_df_fix.loc[test_df_fix['set_size'] == s]
    df_trim = df_trim.drop(columns=['set_size'])
    
    for _,e in enumerate(ecc): # loop over eccentricity
        
        corr,pval = plot_correlation(df_trim[str(e)+'_ecc'].values,test_mean_cs,
                    '# Fixations','CS','CS vs #Fix for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(plot_dir,'CSvsFix_%s-ecc_%s-set.svg'%(str(e),str(s))),p_value=p_value)
        
        # save correlation values         
        corr_fix = corr_fix.append({str(e)+'_ecc': corr, 
                                'set_size': s},ignore_index=True)
        
        # save p-values of said correlations          
        pval_fix = pval_fix.append({str(e)+'_ecc': pval, 
                                'set_size': s},ignore_index=True)

# now reshape the dataframe (droping nans and making sure labels are ok)    
corr_fix = corr_fix.apply(lambda x: pd.Series(x.dropna().values))
corr_fix = corr_fix.dropna()

for i in corr_fix.index:
    corr_fix.at[i, 'set_size'] = params['set_size'][i]
    
# now reshape the dataframe (droping nans and making sure labels are ok)    
pval_fix = pval_fix.apply(lambda x: pd.Series(x.dropna().values))
pval_fix = pval_fix.dropna()

for i in pval_fix.index:
    pval_fix.at[i, 'set_size'] = params['set_size'][i]


##### MATRIX #######
# plot values in color (heat) matrix
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
ax = sns.heatmap(corr_fix[['4_ecc','8_ecc','12_ecc']],cmap='OrRd',
                 vmin = 0.25, vmax = 0.35, annot = True,
                xticklabels=ecc, yticklabels=params['set_size'])

ax.set_ylim([0,3]) # trick to make plot look decente, because matplotlib 3.1 broke seaborn heatmap
ax.invert_yaxis() # so set size goes from smaller (top) to biggest (bottom)
plt.title('#Fix vs CS correlation matrix')
plt.ylabel('Set Size')
plt.xlabel('Eccentricity [dva]')
plt.show()
fig.savefig(os.path.join(plot_dir,'correlation_matrix_ALL_fixations.svg'), dpi=100,bbox_inches = 'tight')

# plot values in color (heat) matrix
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
ax = sns.heatmap(pval_fix[['4_ecc','8_ecc','12_ecc']],cmap='bwr_r',#'OrRd_r',
                 vmin = 0.001, vmax = 0.1, center=0.05, annot = True,
                xticklabels=ecc, yticklabels=params['set_size'])

ax.set_ylim([0,3]) # trick to make plot look decente, because matplotlib 3.1 broke seaborn heatmap
ax.invert_yaxis() # so set size goes from smaller (top) to biggest (bottom)
plt.title('#Fix vs CS correlation matrix - p-values')
plt.ylabel('Set Size')
plt.xlabel('Eccentricity [dva]')
plt.show()
fig.savefig(os.path.join(plot_dir,'correlation_matrix-pval_ALL_fixations.svg'), dpi=100,bbox_inches = 'tight')



# COMPUTE SLOPE VALUES

slope_RT_ecc = []
slope_RT_set = []

slope_fix_ecc = []
slope_fix_set = []


for k in range(len(test_subs)):
    # RT/ecc slope
    slope_RT_ecc.append(linregress(ecc,np.array(test_rt_ecc_vs)[k])[0])
    # RT/set slope
    slope_RT_set.append(linregress(params['set_size'],np.array(test_rt_set_vs)[k])[0])
    # fixation/ecc slope
    slope_fix_ecc.append(linregress(ecc,np.array(test_fix_ecc_vs)[k])[0])
    # fixation/set slope
    slope_fix_set.append(linregress(params['set_size'],np.array(test_fix_set_vs)[k])[0])
    
  
# CS vs RT/ECC SLOPE
print('\ncomparing mean CS and mean RT/ecc slope in VS \n')
plot_correlation(slope_RT_ecc,np.mean(test_all_cs,axis=-1),
                'RT/ECC','CS','CS vs RT/ECC',
                 os.path.join(plot_dir,'CSvsRT_ECC_slope.svg'),p_value=p_value)

# CS vs RT/SET SLOPE
print('\ncomparing mean CS and mean RT/set slope in VS \n')
plot_correlation(slope_RT_set,np.mean(test_all_cs,axis=-1),
                'RT/set','CS','CS vs RT/set',
                 os.path.join(plot_dir,'CSvsRT_set_slope.svg'),p_value=p_value)

# CS vs FIX/ECC SLOPE
print('\ncomparing mean CS and mean fix/ecc slope in VS \n')
plot_correlation(slope_fix_ecc,np.mean(test_all_cs,axis=-1),
                'Fix/ECC','CS','CS vs Fix/ECC',
                 os.path.join(plot_dir,'CSvsFix_ECC_slope.svg'),p_value=p_value)

# CS vs Fix/SET SLOPE
print('\ncomparing mean CS and mean fix/set slope in VS \n')
plot_correlation(slope_fix_set,np.mean(test_all_cs,axis=-1),
                'Fix/set','CS','CS vs Fix/set',
                 os.path.join(plot_dir,'CSvsFix_set_slope.svg'),p_value=p_value)




##### 
# IF NEEDED, CAN CHOOSE A SUBJECT AND TRIAL TO PLOT THE SCANPATH OR RAW GAZE DATA ##
#
#
## dir to save scanpath plots
#scanpath_dir = os.path.join(plot_dir,'scanpaths')
#if not os.path.exists(scanpath_dir):
#     os.makedirs(scanpath_dir)
        
#draw_scanpath_display('86',400,output_vs,scanpath_dir)

#draw_rawdata_display('86',400,output_vs,scanpath_dir)





    










































































