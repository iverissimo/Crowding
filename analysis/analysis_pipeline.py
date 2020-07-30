
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
test_acc_vs_set = []
test_rt_ecc_vs = []
test_rt_set_vs = []
test_fix_ecc_vs = []
test_fix_set_vs = []

# data frames with interesting values divided
test_df_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
test_df_fix = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])

test_df_crowding_RT_NOfl = pd.DataFrame(columns=['sub','ecc','RT'])
test_df_crowding_RT_FL = pd.DataFrame(columns=['sub','ecc','RT'])


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

        #accuracy crowding
        test_acc_fl.append(sum_measures['acc_fl'][j])
        test_acc_nofl.append(sum_measures['acc_nofl'][j])
        # critical spacing
        test_all_cs.append(sum_measures['all_cs'][j])
        test_mean_cs.append(sum_measures['mean_cs'][j])

        # accuracy search
        test_acc_vs_set.append(list(sum_measures['acc_vs_set'][j]))

        ## reaction time search 
        df_RT, _ = mean_RT_set_ecc_combo(df_vs,sum_measures['all_subs'][j])
        # append matrix of combo 
        test_df_RT = test_df_RT.append(df_RT,ignore_index=True)
        # save values per ecc
        test_rt_ecc_vs.append(np.array([np.nanmean(df_RT[str(x)+'_ecc'].values) for _,x in enumerate(ecc)]))
        # save values per set size
        test_rt_set_vs.append(df_RT[[str(x)+'_ecc' for _,x in enumerate(ecc)]].mean(axis=1).values)

        ## number of fixations in search
        df_fix, _ = mean_fix_set_ecc_combo(df_vs,eye_data,sum_measures['all_subs'][j])
        # append matrix of combo 
        test_df_fix = test_df_fix.append(df_fix,ignore_index=True)
        # save values per ecc
        test_fix_ecc_vs.append(np.array([np.nanmean(df_fix[str(x)+'_ecc'].values) for _,x in enumerate(ecc)]))
        # save values per set size
        test_fix_set_vs.append(df_fix[[str(x)+'_ecc' for _,x in enumerate(ecc)]].mean(axis=1).values)

        # save RT crowding
        test_df_crowding_RT_NOfl = test_df_crowding_RT_NOfl.append(RT_crowding(df_crwd,
                                                                               sum_measures['all_subs'][j],
                                                                               flanker=False,ecc=ecc),
                                                                   ignore_index=True)
        test_df_crowding_RT_FL = test_df_crowding_RT_FL.append(RT_crowding(df_crwd,
                                                                           sum_measures['all_subs'][j],
                                                                           flanker=True,ecc=ecc),
                                                                   ignore_index=True)


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
pval_acc_crwd = wilcoxon(np.nanmean(np.array(test_acc_fl),axis=-1), np.nanmean(np.array(test_acc_nofl),axis=-1))[-1]

crwd_acc4plot = pd.DataFrame(data=np.array([np.nanmean(np.array(test_acc_fl),axis=-1),
                                np.nanmean(np.array(test_acc_nofl),axis=-1)]).T, 
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


# SHOW INDIVIDUAL SUB CS DISTRIBITUION PER ECC

crwd_df4plot = pd.DataFrame([])

for w in range(len(ecc)):
    cs_ecc = [test_all_cs[val][w] for val in range(len(test_all_cs))]
    crwd_df4plot = crwd_df4plot.append(pd.DataFrame({'ecc': np.tile(ecc[w],len(cs_ecc)),
                                  'cs':cs_ecc,
                                    'sub':np.array(test_subs)}),ignore_index=True)

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

sns.lineplot(x='ecc', y='cs', data=crwd_df4plot,estimator='mean')
sns.lineplot(x='ecc', y='cs', data=crwd_df4plot,
                   units='sub', estimator=None, lw=1,color='grey',alpha=0.2)

plt.title('Critical Spacing, per ecc')
fig.savefig(os.path.join(plot_dir,'crowding_CS_ecc_all.svg'), dpi=100)

# BOXPLOTS WITH CS PER ECC
fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
v1 = sns.violinplot(x='ecc', y='cs', data=crwd_df4plot,cut=0, inner='box', palette='YlOrRd_r',linewidth=3)#ax.margins(y=0.05)
v1.set(xlabel=None)
v1.set(ylabel=None)
#plt.margins(y=0.025)
#sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25",alpha=0.5)
#plt.plot(np.array(test_all_cs).T,'r-o',c='k',alpha=0.1)#c='grey',alpha=0.5)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

#plt.title('Critical spacing distribution across eccentricity',fontsize=18,pad=10)
plt.xlabel('Eccentricity [dva]',fontsize=30,labelpad=16)
plt.ylabel('Critical Spacing',fontsize=30,labelpad=16)
plt.ylim(0.2,0.8)

fig.savefig(os.path.join(plot_dir,'crowding_meanCS-ecc-violinplot.svg'), dpi=100)

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


# PLOT RT FOR NO FLANKER CONDITION
# to see if differences between ecc

df4plot_RT_NOfl = test_df_crowding_RT_NOfl.drop(columns=['sub'])

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

sns.boxplot(x='ecc',y='RT', data=df4plot_RT_NOfl)
sns.swarmplot(x="ecc", y="RT", data=df4plot_RT_NOfl, color=".3")
plt.ylabel('RT [s]')
plt.xlabel('ECC [dva]')
plt.title('RT crowding (without flankers)')
fig.savefig(os.path.join(plot_dir,'crowding_RT_NOflanker.svg'), dpi=100)

# do Friedman to see if averages are different 
# The Friedman test tests the null hypothesis that repeated measurements of the same individuals have the same distribution. 
#It is often used to test for consistency among measurements obtained in different ways.

pfriedman = friedmanchisquare(df4plot_RT_NOfl.loc[(df4plot_RT_NOfl['ecc']==4)]['RT'].values, 
                              df4plot_RT_NOfl.loc[(df4plot_RT_NOfl['ecc']==8)]['RT'].values, 
                              df4plot_RT_NOfl.loc[(df4plot_RT_NOfl['ecc']==12)]['RT'].values)[-1]
if pfriedman<p_value:
    print('RT without flankers between ecc is different, friedman with p-value = %.6f'%pfriedman)
else:
    print('RT without flankers between ecc is the same, p-value = %.6f'%pfriedman)


df4plot_RT_FL = test_df_crowding_RT_FL.drop(columns=['sub'])

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

sns.boxplot(x='ecc',y='RT', data=df4plot_RT_FL)
sns.swarmplot(x="ecc", y="RT", data=df4plot_RT_FL, color=".3")
plt.ylabel('RT [s]')
plt.xlabel('ECC [dva]')
plt.title('RT crowding (with flankers)')
fig.savefig(os.path.join(plot_dir,'crowding_RT_flanker.svg'), dpi=100)

# do Friedman to see if averages are different 
# The Friedman test tests the null hypothesis that repeated measurements of the same individuals have the same distribution. 
#It is often used to test for consistency among measurements obtained in different ways.

pfriedman = friedmanchisquare(df4plot_RT_FL.loc[(df4plot_RT_FL['ecc']==4)]['RT'].values, 
                              df4plot_RT_FL.loc[(df4plot_RT_FL['ecc']==8)]['RT'].values, 
                              df4plot_RT_FL.loc[(df4plot_RT_FL['ecc']==12)]['RT'].values)[-1]
if pfriedman<p_value:
    print('RT with flankers between ecc is different, friedman with p-value = %.6f'%pfriedman)
else:
    print('RT with flankers between ecc is the same, p-value = %.6f'%pfriedman)

##### 

# VISUAL SEARCH

# ACCURACY VISUAL SEARCH
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(1,2,1)
plt.hist(np.array(test_acc_vs_set))
plt.title('Accuracy visual search')
plt.legend(['5 items','15 items','30 items'])
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

# For all ecc do separate violin plots
# colors a bit off between legend and violin, so corrected for in inkscape
for k,_ in enumerate(ecc):
    colors_ecc = np.array([['#ff8c8c','#e31a1c','#870000'],
                           ['#faad75','#ff6b00','#cc5702'],
                           ['#fff5a3','#eecb5f','#f4b900']])#['#ffea80','#fff200','#dbbe00']])

    columns2drop = np.array([['sub','8_ecc','12_ecc'],
                           ['sub','4_ecc','12_ecc'],
                           ['sub','4_ecc','8_ecc']])

    df4plot_RT_set_ecc = test_df_RT.drop(columns=columns2drop[k])
    df4plot_RT_set_ecc = pd.melt(df4plot_RT_set_ecc, 'set_size', var_name='Target eccentricity [dva]', value_name='RT [s]')

    fig = plt.figure(num=None, figsize=(7.5,7.5), dpi=100, facecolor='w', edgecolor='k')
    v1 = sns.violinplot(x='Target eccentricity [dva]', hue='set_size', y='RT [s]', data=df4plot_RT_set_ecc,
                  cut=0, inner='box', palette=colors_ecc[k],linewidth=3)

    plt.legend().remove()
    plt.xticks([], [])
    #plt.xticks([0,1,2], ('5', '15', '30'))

    v1.set(xlabel=None)
    v1.set(ylabel=None)

    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    if k==0:
        plt.ylabel('RT [s]',fontsize=30,labelpad=16)
    elif k==1:
        plt.xlabel('Set Size [items]',fontsize=30,labelpad=40)
    plt.title('%d dva'%ecc[k],fontsize=30,pad=12)

    plt.ylim(0.25,3)
    plt.savefig(os.path.join(plot_dir,'search_ecc_RT_violin_%decc.svg'%ecc[k]), dpi=100,bbox_inches = 'tight')
    
    
    
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

# For all ecc do separate plots
# colors a bit off between legend and violin, so corrected for in inkscape
for k,_ in enumerate(ecc):
    colors_ecc = np.array([['#ff8c8c','#e31a1c','#870000'],
                           ['#faad75','#ff6b00','#cc5702'],
                           ['#fff5a3','#eecb5f','#f4b900']])#['#ffea80','#fff200','#dbbe00']])

    columns2drop = np.array([['sub','8_ecc','12_ecc'],
                           ['sub','4_ecc','12_ecc'],
                           ['sub','4_ecc','8_ecc']])

    df4plot_fix_set_ecc = test_df_fix.drop(columns=columns2drop[k])
    df4plot_fix_set_ecc = pd.melt(df4plot_fix_set_ecc, 'set_size', var_name='Target eccentricity [dva]', value_name='# Fixations')

    fig = plt.figure(num=None, figsize=(7.5,7.5), dpi=100, facecolor='w', edgecolor='k')
    v1 = sns.violinplot(x='Target eccentricity [dva]', hue='set_size', y='# Fixations', data=df4plot_fix_set_ecc,
                  cut=0, inner='box', palette=colors_ecc[k],linewidth=3)

    plt.legend().remove()
    plt.xticks([], [])
    #plt.xticks([0,1,2], ('5', '15', '30'))

    v1.set(xlabel=None)
    v1.set(ylabel=None)

    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    if k==0:
        plt.ylabel('# Fixations',fontsize=30,labelpad=16)
    elif k==1:
        plt.xlabel('Set Size [items]',fontsize=30,labelpad=40)
    plt.title('%d dva'%ecc[k],fontsize=30,pad=12)

    plt.ylim(0,10)
    plt.savefig(os.path.join(plot_dir,'search_ecc_numfix_violin_%decc.svg'%ecc[k]), dpi=100,bbox_inches = 'tight')
    
    

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


## IMPLEMENT ANOVA WITH STATSMODEL
## FOR RT ##

# reshape data frame
new_DF = pd.DataFrame(columns=['sub','set_size','ecc','RT'])

for _,sj in enumerate(test_subs): # loop over subject
    df_trim = test_df_RT.loc[test_df_RT['sub'] == sj]
    
    for _,s in enumerate(params['set_size']): # loop over set size
        
        new_df_trim = df_trim.loc[df_trim['set_size'] == s]

        for e_ind,e in enumerate(ecc): # loop over eccentricity
            
             new_DF = new_DF.append({
                                'sub': sj,
                                'set_size': s,
                                'ecc': e,
                                'RT': new_df_trim[str(e)+'_ecc'].values[0] 
                                 },ignore_index=True)
                
aovrm2way = AnovaRM(new_DF, depvar='RT', subject='sub', within=['set_size', 'ecc'])
res2way = aovrm2way.fit()

print(res2way)

# save it
res2way.anova_table.to_csv(os.path.join(plot_dir,'RT_ANOVA.csv'))

#sns.boxplot(x="set_size", y="RT", hue="ecc", data=new_DF, palette="Set3") 


## FOR NUMBER OF FIXATIONS ##

# reshape data frame
new_DF = pd.DataFrame(columns=['sub','set_size','ecc','Fixations'])

for _,sj in enumerate(test_subs): # loop over subject
    df_trim = test_df_fix.loc[test_df_RT['sub'] == sj]
    
    for _,s in enumerate(params['set_size']): # loop over set size
        
        new_df_trim = df_trim.loc[df_trim['set_size'] == s]

        for e_ind,e in enumerate(ecc): # loop over eccentricity
            
             new_DF = new_DF.append({
                                'sub': sj,
                                'set_size': s,
                                'ecc': e,
                                'Fixations': new_df_trim[str(e)+'_ecc'].values[0] 
                                 },ignore_index=True)
                
aovrm2way = AnovaRM(new_DF, depvar='Fixations', subject='sub', within=['set_size', 'ecc'])
res2way = aovrm2way.fit()

print(res2way)

# save it
res2way.anova_table.to_csv(os.path.join(plot_dir,'Fix_ANOVA.csv'))

#sns.boxplot(x="set_size", y="Fixations", hue="ecc", data=new_DF, palette="Set3") 



# Correlations between tasks

#The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. 
#Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.

## FOR RT
    
corr_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])
pval_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])

for _,s in enumerate(params['set_size']): # loop over set size
    
    df_trim = test_df_RT.loc[test_df_RT['set_size'] == s]
    df_trim = df_trim.drop(columns=['set_size'])
    
    for _,e in enumerate(ecc): # loop over eccentricity
        
        corr,pval = plot_correlation(test_mean_cs,df_trim[str(e)+'_ecc'].values,
                    'CS','RT [s]','CS vs RT for %s ecc and %s items'%(str(e),str(s)),
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
#plt.show()
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
#plt.show()
fig.savefig(os.path.join(plot_dir,'correlation_matrix-pval_ALL_RT.svg'), dpi=100,bbox_inches = 'tight')


## FOR FIXATIONS
    
corr_fix = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])
pval_fix = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])

for _,s in enumerate(params['set_size']): # loop over set size
    
    df_trim = test_df_fix.loc[test_df_fix['set_size'] == s]
    df_trim = df_trim.drop(columns=['set_size'])
    
    for _,e in enumerate(ecc): # loop over eccentricity
        
        corr,pval = plot_correlation(test_mean_cs,df_trim[str(e)+'_ecc'].values,
                    'CS','# Fixations','CS vs #Fix for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(plot_dir,'CSvsFix_%s-ecc_%s-set.svg'%(str(e),str(s))),p_value=p_value,
                     y_lim = [0,8])
        
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
#plt.show()
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
#plt.show()
fig.savefig(os.path.join(plot_dir,'correlation_matrix-pval_ALL_fixations.svg'), dpi=100,bbox_inches = 'tight')



##### COMPUTE SLOPE VALUES # 


# to save all slope values of RT/set (rt in seconds, so slope in s/set)
# thus we can divide it per ecc or across ecc

df_slope_RT_set = pd.DataFrame(columns=ecc+['all'])
df_slope_Fix_set = pd.DataFrame(columns=ecc+['all'])


for _,e in enumerate(ecc): # loop over eccentricity
    
    df_trim = test_df_RT[[str(e)+'_ecc']].values
    df_trim_fix = test_df_fix[[str(e)+'_ecc']].values
    
    w = 0 # counter 
    for k in range(len(test_subs)): # loop over all subjects
        
        slope_1set = linregress(params['set_size'],np.array([df_trim[w][0],df_trim[w+1][0],df_trim[w+2][0]])).slope
        slope_1set_fix = linregress(params['set_size'],np.array([df_trim_fix[w][0],df_trim_fix[w+1][0],df_trim_fix[w+2][0]])).slope
        
        if e==ecc[0]:
            slope_across_set = linregress(params['set_size'],np.array(test_rt_set_vs)[k]).slope
            slope_across_set_fix = linregress(params['set_size'],np.array(test_fix_set_vs)[k]).slope
        
            # append values in matrix, including across       
            df_slope_RT_set = df_slope_RT_set.append({e: slope_1set, 
                                    'all': slope_across_set},ignore_index=True)
            df_slope_Fix_set = df_slope_Fix_set.append({e: slope_1set_fix, 
                                    'all': slope_across_set_fix},ignore_index=True)
        else:
            # append values in matrix, without across (to avoid duplicated values in DF)        
            df_slope_RT_set = df_slope_RT_set.append({e: slope_1set},ignore_index=True)
            df_slope_Fix_set = df_slope_Fix_set.append({e: slope_1set_fix},ignore_index=True)
            
        w+=3 # increment counter to move to vals of next subject (shitty coding, but works)

# now reshape the dataframe (droping nans and making sure labels are ok)    
df_slope_RT_set = df_slope_RT_set.apply(lambda x: pd.Series(x.dropna().values))
df_slope_RT_set = df_slope_RT_set.dropna()

df_slope_Fix_set = df_slope_Fix_set.apply(lambda x: pd.Series(x.dropna().values))
df_slope_Fix_set = df_slope_Fix_set.dropna()


## correlations for RT/set [s/item?]
    
corr_slope_RT_set = pd.DataFrame(columns=ecc+['all'])
pval_slope_RT_set = pd.DataFrame(columns=ecc+['all'])
  
for _,e in enumerate(ecc+['all']): # loop over eccentricity

    corr,pval = plot_correlation(test_mean_cs,df_slope_RT_set[e].values,
                'CS','RT/set size [s/item]','CS vs RT/set size for %s ecc'%(str(e)),
                 os.path.join(plot_dir,'CSvsRT_SET_SLOPE_%s-ecc.svg'%(str(e))),p_value=p_value,
                 x_lim = [0.2,0.8],y_lim = [0,0.05],decimals=1)

    # save correlation values         
    corr_slope_RT_set = corr_slope_RT_set.append({e: corr},ignore_index=True)

    # save p-values of said correlations          
    pval_slope_RT_set = pval_slope_RT_set.append({e: pval},ignore_index=True)


## correlations for Fix/set [fix/item?]
    
corr_slope_Fix_set = pd.DataFrame(columns=ecc+['all'])
pval_slope_Fix_set = pd.DataFrame(columns=ecc+['all'])
  
for _,e in enumerate(ecc+['all']): # loop over eccentricity

    corr,pval = plot_correlation(test_mean_cs,df_slope_Fix_set[e].values,
                'CS','# Fixations/set size [fix/item]','CS vs # Fixations/set size for %s ecc'%(str(e)),
                 os.path.join(plot_dir,'CSvsFix_SET_SLOPE_%s-ecc.svg'%(str(e))),p_value=p_value,
                 x_lim = [0.2,0.8],y_lim = [0,0.25],decimals=1)

    # save correlation values         
    corr_slope_Fix_set = corr_slope_Fix_set.append({e: corr},ignore_index=True)

    # save p-values of said correlations          
    pval_slope_Fix_set = pval_slope_Fix_set.append({e: pval},ignore_index=True)



## add analysis were I do the correlations per condition
# 3 x 3 grid
# using different CS values (per eccentricity)
# for appendix figures

## FOR RT
    
corr_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])
pval_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])

for _,s in enumerate(params['set_size']): # loop over set size
    
    df_trim = test_df_RT.loc[test_df_RT['set_size'] == s]
    df_trim = df_trim.drop(columns=['set_size'])
    
    for e_ind,e in enumerate(ecc): # loop over eccentricity (using the specific CS for that eccentricity)
        
        corr,pval = plot_correlation(np.array(test_all_cs).T[e_ind],df_trim[str(e)+'_ecc'].values,
                    'CS','RT [s]','CS vs RT for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(plot_dir,'CSvsRT_%s-ecc_%s-set_CS-%sdva.svg'%(str(e),str(s),str(e))),
                                     p_value=p_value,x_lim = [.19,.81])
        
        # save correlation values         
        corr_RT = corr_RT.append({str(e)+'_ecc': corr, 
                                'set_size': s},ignore_index=True)
        
        # save p-values of said correlations          
        pval_RT = pval_RT.append({str(e)+'_ecc': pval, 
                                'set_size': s},ignore_index=True)


## FOR FIXATIONS
    
corr_fix = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])
pval_fix = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])

for _,s in enumerate(params['set_size']): # loop over set size
    
    df_trim = test_df_fix.loc[test_df_fix['set_size'] == s]
    df_trim = df_trim.drop(columns=['set_size'])
    
    for e_ind,e in enumerate(ecc): # loop over eccentricity
        
        corr,pval = plot_correlation(np.array(test_all_cs).T[e_ind],df_trim[str(e)+'_ecc'].values,
                    'CS','# Fixations','CS vs #Fix for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(plot_dir,'CSvsFix_%s-ecc_%s-set_CS-%sdva.svg'%(str(e),str(s),str(e))),
                                     p_value=p_value,y_lim = [0,8],x_lim = [.19,.81])
        
        # save correlation values         
        corr_fix = corr_fix.append({str(e)+'_ecc': corr, 
                                'set_size': s},ignore_index=True)
        
        # save p-values of said correlations          
        pval_fix = pval_fix.append({str(e)+'_ecc': pval, 
                                'set_size': s},ignore_index=True)



#### NOW FOR SLOPES #####

## correlations for RT/set [s/item?]
    
corr_slope_RT_set = pd.DataFrame(columns=ecc+['all'])
pval_slope_RT_set = pd.DataFrame(columns=ecc+['all'])
  
for e_ind,e in enumerate(ecc): # loop over eccentricity

    corr,pval = plot_correlation(np.array(test_all_cs).T[e_ind],df_slope_RT_set[e].values,
                'CS','RT/set size [s/item]','CS vs RT/set size for %s ecc'%(str(e)),
                 os.path.join(plot_dir,'CSvsRT_SET_SLOPE_%s-ecc_CS-%sdva.svg'%(str(e),str(e))),p_value=p_value,
                 x_lim = [0.19,0.81],y_lim = [0,0.05],decimals=1)

    # save correlation values         
    corr_slope_RT_set = corr_slope_RT_set.append({e: corr},ignore_index=True)

    # save p-values of said correlations          
    pval_slope_RT_set = pval_slope_RT_set.append({e: pval},ignore_index=True)


## correlations for Fix/set [fix/item?]
    
corr_slope_Fix_set = pd.DataFrame(columns=ecc+['all'])
pval_slope_Fix_set = pd.DataFrame(columns=ecc+['all'])
  
for e_ind,e in enumerate(ecc): # loop over eccentricity

    corr,pval = plot_correlation(np.array(test_all_cs).T[e_ind],df_slope_Fix_set[e].values,
                'CS','# Fixations/set size [fix/item]','CS vs # Fixations/set size for %s ecc'%(str(e)),
                 os.path.join(plot_dir,'CSvsFix_SET_SLOPE_%s-ecc_CS-%sdva.svg'%(str(e),str(e))),p_value=p_value,
                 x_lim = [0.19,0.81],y_lim = [0,0.25],decimals=1)

    # save correlation values         
    corr_slope_Fix_set = corr_slope_Fix_set.append({e: corr},ignore_index=True)

    # save p-values of said correlations          
    pval_slope_Fix_set = pval_slope_Fix_set.append({e: pval},ignore_index=True)


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





    










































































