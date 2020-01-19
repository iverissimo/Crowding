
# full analysis pipeline
# makes some plots
# excludes participants that don't abid by exclusion criteria
# computes some stats on final sample set


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

from scipy.stats import wilcoxon, kstest, spearmanr, linregress, friedmanchisquare,kruskal #pearsonr,  
from statsmodels.stats import weightstats
import seaborn as sns

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


# Params CROWDING
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
    
    # plot staircase just to see what's going on
    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    plt.plot(np.array(all_tfr[ind]).T)
    plt.ylabel("Target-flanker distance (ratio of ecc)",fontsize=18)
    plt.xlabel("# Trial",fontsize=18)
    plt.title("Target-Flanker ratio per ecc",fontsize=18)
    plt.legend(analysis_params['ecc'])
    fig.savefig(os.path.join(plot_dir,'crowding_TFR_per_ecc_sub-{sj}.svg'.format(sj=all_subs[ind])), dpi=100)
    
    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    plt.plot(data_sub['target_flank_ratio'].values,alpha=0.8)
    plt.hlines(y=mean_cs[ind], color='r', linestyle='--',linewidth=2,xmin=len(data_sub)-last_trials,xmax=len(data_sub))
    plt.xlim(0,len(data_sub))

    plt.ylabel("Target-flanker distance (ratio of ecc)",fontsize=18)
    plt.xlabel("# Trial",fontsize=18)
    plt.title("Target-Flanker ratio all trials",fontsize=18)
    #plt.show()
    fig.savefig(os.path.join(plot_dir,'crowding_TFR_all_sub-{sj}.svg'.format(sj=all_subs[ind])), dpi=100)


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
    
    # save relevant variables from above in descriptive tsv
    summary_df = pd.DataFrame({'sub':all_subs[ind],
                                'nan_trials_pct': missed_trials[ind],
                                'accuracy_flankers_pct':acc_fl[ind]*100,
                                'accuracy_noflankers_pct':acc_nofl[ind]*100,
                                'crit_spacing_all':all_cs[ind],
                                'crit_spacing_mean':mean_cs[ind],
                                'accuracy_vs_all_pct':acc_vs_ecc[ind]*100,
                                'accuracy_vs_mean_pct':acc_vs_all[ind]*100,
                                'exclude':ex_sub
                              })

    summary_df.to_csv(os.path.join(plot_dir,'summary_pp_'+all_subs[ind]+'.csv'), sep='\t')
   
    

# then take out excluded participants and save relevant info in new arrays
# for later analysis

# find index to take out from variables
exclusion_ind = [np.where(np.array(all_subs)==np.array(excluded_sub[i]))[0][0] for i in range(len(excluded_sub))] 

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
test_onobjfix_ecc_vs = []
test_onobjfix_set_vs = []

for j in range(len(all_subs)):
    
    if j == exclusion_ind[counter]:
        print('skipping sub %s'%all_subs[j])
        if counter<len(exclusion_ind)-1:
            counter+=1   
    else:
        df_vs = pd.read_csv(os.path.join(output_vs,vs_csv[j]), sep='\t')
        _, eye_data = convert2asc(all_subs[j],'vs',output_vs,output_crwd)
        
        test_subs.append(all_subs[j])
        
        test_acc_fl.append(acc_fl[j])
        test_acc_nofl.append(acc_nofl[j])
        test_all_cs.append(all_cs[j])
        test_mean_cs.append(mean_cs[j])
        
        test_acc_vs_ecc.append(acc_vs_ecc[j])
        test_rt_ecc_vs.append(mean_RT(df_vs,ecc))
        test_rt_set_vs.append(mean_RT_setsize(df_vs,analysis_params['set_size']))
        
        test_fix_ecc_vs.append(meanfix_ecc(df_vs,eye_data,ecc))
        test_fix_set_vs.append(meanfix_setsize(df_vs,eye_data,analysis_params['set_size']))

        test_onobjfix_ecc_vs.append(on_objectfix_ecc(df_vs,eye_data,ecc,analysis_params['siz_gab_deg']/2*1.5))
        test_onobjfix_set_vs.append(on_objectfix_set(df_vs,eye_data,analysis_params['set_size'],analysis_params['siz_gab_deg']/2*1.5))

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

fig.savefig(os.path.join(plot_dir,'crowding_accuracy_hist_%d-subs.svg'%len(test_subs)), dpi=100)


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

fig.savefig(os.path.join(plot_dir,'crowding_mean_accuracy_boxplot_wilcoxtest_%d-subs.svg'%len(test_subs)), dpi=100)


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
    
    fig.savefig(os.path.join(plot_dir,'crowding_%decc_accuracy_boxplot_wilcoxtest_%d-subs.svg'%(ecc[g],len(test_subs))), dpi=100)


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
fig.savefig(os.path.join(plot_dir,'crowding_meanCS-weighted_errorbar_%d-subs.svg'%len(test_subs)), dpi=100)

# SHOW INDIVIDUAL SUB CS DISTRIBITUION PER ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

with sns.color_palette("pastel", len(test_subs)):
    sns.catplot(x="ecc", y="cs", hue="sub", kind="point", data=crwd_df4plot)
plt.title('Critical Spacing, per ecc')
fig.savefig(os.path.join(plot_dir,'crowding_CS_ecc_individual_%d-subs.svg'%len(test_subs)), dpi=100)

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

sns.lineplot(x='ecc', y='cs', data=crwd_df4plot,estimator='mean')
sns.lineplot(x='ecc', y='cs', data=crwd_df4plot,
                   units='sub', estimator=None, lw=1,color='grey',alpha=0.2)

plt.title('Critical Spacing, per ecc')
fig.savefig(os.path.join(plot_dir,'crowding_CS_ecc_all_%d-subs.svg'%len(test_subs)), dpi=100)

# BOXPLOTS WITH CS PER ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
sns.boxplot(x='ecc', y='cs', data=crwd_df4plot)
sns.swarmplot(x='ecc', y='cs', data=crwd_df4plot,color=".25")
fig.savefig(os.path.join(plot_dir,'crowding_meanCS-ecc-weighted_boxplot_%d-subs.svg'%len(test_subs)), dpi=100)

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

fig.savefig(os.path.join(plot_dir,'search_accuracy_RT_hist_%d-subs.svg'%len(test_subs)), dpi=100)


# RT VS ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

rt_ecc_vs4plot = pd.DataFrame([])

for k in range(len(ecc)):
    rt_ecc_vs4plot = rt_ecc_vs4plot.append(pd.DataFrame({'RT': np.array(test_rt_ecc_vs).T[k],
                                                     'ecc':np.tile(ecc[k],len(test_subs))}))
ax = sns.lmplot(x='ecc', y='RT',data=rt_ecc_vs4plot)
ax.set(xlabel='eccentricity [dva]', ylabel='RT [s]')
ax = plt.gca()
ax.set_title('ecc vs RT %d subs'%len(test_subs))
plt.savefig(os.path.join(plot_dir,'search_ecc_RT_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')
 

# RT VS SET SIZE
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

rt_set_vs4plot = pd.DataFrame([])

for k in range(len(analysis_params['set_size'])):
    rt_set_vs4plot = rt_set_vs4plot.append(pd.DataFrame({'RT': np.array(test_rt_set_vs).T[k],
                                                     'set':np.tile(analysis_params['set_size'][k],len(test_subs))}))
ax = sns.lmplot(x='set', y='RT',data=rt_set_vs4plot)
ax.set(xlabel='set size', ylabel='RT [s]')
ax = plt.gca()
ax.set_title('set size vs RT %d subs'%len(test_subs))
plt.savefig(os.path.join(plot_dir,'search_setsize_RT_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')
 

# INVERSE EFFICIENCY SCORE
inv_eff = np.array(test_rt_ecc_vs)/np.array(test_acc_vs_ecc)
inv_eff_mean = np.mean(inv_eff,axis=0)
print('mean inverse efficiency score, per ecc is %s'%str(inv_eff_mean))


# EYETRACKING FOR VISUAL SEARCH

# NUMBER OF FIXATIONS VS ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

fix_ecc_vs4plot = pd.DataFrame([])

for k in range(len(ecc)):
    fix_ecc_vs4plot = fix_ecc_vs4plot.append(pd.DataFrame({'fix': np.array(test_fix_ecc_vs).T[k],
                                                     'ecc':np.tile(ecc[k],len(test_subs))}))
ax = sns.lmplot(x='ecc', y='fix',data=fix_ecc_vs4plot)
ax.set(xlabel='eccentricity [dva]', ylabel='# fixations')
ax = plt.gca()
ax.set_title('ecc vs number of fixations %d subs'%len(test_subs))
ax.axes.set_ylim(0,)
plt.savefig(os.path.join(plot_dir,'search_ecc_numfix_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')
 

# NUMBER OF FIXATIONS VS SET SIZE
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

fix_set_vs4plot = pd.DataFrame([])

for k in range(len(analysis_params['set_size'])):
    fix_set_vs4plot = fix_set_vs4plot.append(pd.DataFrame({'fix': np.array(test_fix_set_vs).T[k],
                                                     'set':np.tile(analysis_params['set_size'][k],len(test_subs))}))
ax = sns.lmplot(x='set', y='fix',data=fix_set_vs4plot)
ax.set(xlabel='set size', ylabel='number fixations')
ax = plt.gca()
ax.set_title('set size vs number fixations %d subs'%len(test_subs))
plt.savefig(os.path.join(plot_dir,'search_setsize_fix_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')

# PERCENTAGE OF ON OBJECT FIXATIONS VS ECC
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

onobj_ecc_vs4plot = pd.DataFrame([])

for k in range(len(ecc)):
    onobj_ecc_vs4plot = onobj_ecc_vs4plot.append(pd.DataFrame({'onobj': np.array(test_onobjfix_ecc_vs).T[k],
                                                     'ecc':np.tile(ecc[k],len(test_subs))}))
ax = sns.lmplot(x='ecc', y='onobj',data=onobj_ecc_vs4plot)
ax.set(xlabel='eccentricity [dva]', ylabel='On object fixation [%]')
ax = plt.gca()
ax.set_title('ecc vs on-object fixations %d subs'%len(test_subs))
ax.axes.set_ylim(0,)
plt.savefig(os.path.join(plot_dir,'search_ecc_onobjectfix_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')

# PERCENTAGE OF ON OBJECT FIXATIONS VS SET SIZE
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

onobj_set_vs4plot = pd.DataFrame([])

for k in range(len(analysis_params['set_size'])):
    onobj_set_vs4plot = onobj_set_vs4plot.append(pd.DataFrame({'onobj': np.array(test_onobjfix_set_vs).T[k],
                                                     'set':np.tile(analysis_params['set_size'][k],len(test_subs))}))
ax = sns.lmplot(x='set', y='onobj',data=onobj_set_vs4plot)
ax.set(xlabel='set size', ylabel='On object fixation [%]')
ax = plt.gca()
ax.set_title('set size vs on-object fixations %d subs'%len(test_subs))
plt.savefig(os.path.join(plot_dir,'search_setsize_onobjectfix_regression_%d-subs.svg'%len(test_subs)), dpi=100,bbox_inches = 'tight')
 

# Correlations between tasks

#The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. 
#Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.

# CS VS RT ACROSS ECC
cor_CSmean_RT, pval_cor_CSmean_RT = spearmanr(np.mean(test_all_cs,axis=-1),np.mean(test_rt_ecc_vs,axis=-1))
print('\ncomparing critical distance mean across ecc and mean RT in VS across ecc \n')
print('correlation = %.6f, p-value = %.6f'%(cor_CSmean_RT,pval_cor_CSmean_RT))
if pval_cor_CSmean_RT<p_value:
    print('SIGNIFICANT CORRELATION')
    
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

CS_RT_mean_df4plot = pd.DataFrame({'CS': np.mean(test_all_cs,axis=-1),
            'RT': np.mean(test_rt_ecc_vs,axis=-1)})

ax = sns.lmplot(x='RT', y='CS',data=CS_RT_mean_df4plot)
ax.set(ylabel='CS [dva]', xlabel='RT [s]')
ax = plt.gca()
ax.set_title('CS vs RT (rho=%.2f,pval=%.3f)'%(cor_CSmean_RT,pval_cor_CSmean_RT))
plt.savefig(os.path.join(plot_dir,'CSvsRT_across-ecc_%d-subs.svg'%(len(test_subs))), dpi=100,bbox_inches = 'tight')


# CS VS RT PER ECC
for i in range(len(np.array(test_all_cs).T)):
    cor_CSecc_RT, pval_cor_CSecc_RT = spearmanr(np.array(test_all_cs).T[i],np.array(test_rt_ecc_vs).T[i])
    print('\ncomparing critical distance and mean RT in VS for ecc %d\n'%ecc[i])
    print('correlation = %.6f, p-value = %.6f'%(cor_CSecc_RT,pval_cor_CSecc_RT))

    if pval_cor_CSecc_RT<p_value:
        print('SIGNIFICANT CORRELATION')

    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
    cs_rt_4plot = pd.DataFrame({'RT': np.array(test_rt_ecc_vs).T[i],
                                'CS':np.array(test_all_cs).T[i]})
    
    ax = sns.lmplot(x='RT', y='CS',data=cs_rt_4plot)
    ax.set(ylabel='CS [dva]', xlabel='RT [s]')
    ax = plt.gca()
    ax.set_title('CS vs RT at %d ecc (rho=%.2f,pval=%.3f)'%(ecc[i],cor_CSecc_RT,pval_cor_CSecc_RT))
    plt.savefig(os.path.join(plot_dir,'CSvsRT_%d-ecc_%d-subs.svg'%(ecc[i],len(test_subs))), dpi=100,bbox_inches = 'tight') 


# CS VS INVERSE EFFICACY ACROSS ECC
cor_CSmean_inveff, pval_cor_CSmean_inveff = spearmanr(np.mean(test_all_cs,axis=-1),np.mean(inv_eff,axis=-1))
print('\ncomparing critical distance mean across ecc and mean inverse efficiency in VS across ecc \n')
print('correlation = %.6f, p-value = %.6f'%(cor_CSmean_inveff,pval_cor_CSmean_inveff))

if pval_cor_CSmean_inveff<p_value:
    print('SIGNIFICANT CORRELATION')
    
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

CS_RT_mean_df4plot = pd.DataFrame({'CS': np.mean(test_all_cs,axis=-1),
            'invEffic': np.mean(inv_eff,axis=-1)})

ax = sns.lmplot(x='invEffic', y='CS',data=CS_RT_mean_df4plot)
ax.set(ylabel='CS [dva]', xlabel='invEffic [a.u.]')
ax = plt.gca()
ax.set_title('CS vs RT (rho=%.2f,pval=%.3f)'%(cor_CSmean_inveff,pval_cor_CSmean_inveff))
plt.savefig(os.path.join(plot_dir,'CSvsinvEffic_across-ecc_%d-subs.svg'%(len(test_subs))), dpi=100,bbox_inches = 'tight')
    

# CS VS INVERSE EFFICACY PER ECC
for i in range(len(np.array(test_all_cs).T)):
    cor_CSecc_inveff, pval_cor_CSecc_inveff = spearmanr(np.array(test_all_cs).T[i],np.array(inv_eff).T[i])
    print('\ncomparing critical distance mean and mean inverse efficiency in VS for ecc %d\n'%ecc[i])
    print('correlation = %.6f, p-value = %.6f'%(cor_CSecc_inveff,pval_cor_CSecc_inveff))
    
    if pval_cor_CSecc_inveff<p_value:
        print('SIGNIFICANT CORRELATION')

    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
    cs_inveff_4plot = pd.DataFrame({'invEffic': np.array(inv_eff).T[i],
                                    'CS':np.array(test_all_cs).T[i]})
    
    ax = sns.lmplot(x='invEffic', y='CS',data=cs_inveff_4plot)
    ax.set(ylabel='CS [dva]', xlabel='inverse efficacy [a.u.]')
    ax = plt.gca()
    ax.set_title('CS vs invEffic at %d ecc (rho=%.2f,pval=%.3f)'%(ecc[i],cor_CSecc_inveff,pval_cor_CSecc_inveff))
    plt.savefig(os.path.join(plot_dir,'CSvsinvEffic_%d-ecc_%d-subs.svg'%(ecc[i],len(test_subs))), dpi=100,bbox_inches = 'tight') 



# COMPUTE SLOPE VALUES

## check later, slope values seem a bit low... ##

slope_RT_ecc = []
slope_RT_set = []

slope_fix_ecc = []
slope_fix_set = []

slope_onobj_ecc = []
slope_onobj_set = []

for k in range(len(test_subs)):
    # RT/ecc slope
    slope_RT_ecc.append(linregress(ecc,np.array(test_rt_ecc_vs)[k])[0])
    # RT/set slope
    slope_RT_set.append(linregress(analysis_params['set_size'],np.array(test_rt_set_vs)[k])[0])
    # fixation/ecc slope
    slope_fix_ecc.append(linregress(ecc,np.array(test_fix_ecc_vs)[k])[0])
    # fixation/set slope
    slope_fix_set.append(linregress(analysis_params['set_size'],np.array(test_fix_set_vs)[k])[0])
    # on-object fixation/ecc slope
    slope_onobj_ecc.append(linregress(ecc,np.array(test_onobjfix_ecc_vs)[k])[0])
    # on-object fixation/set slope
    slope_onobj_set.append(linregress(analysis_params['set_size'],np.array(test_onobjfix_set_vs)[k])[0])

  
# CS vs RT/ECC SLOPE
cor_CSmean_RTecc_slope, pval_cor_CSmean_RTecc_slope = spearmanr(slope_RT_ecc,np.mean(test_all_cs,axis=-1))
print('\ncomparing critical distance mean across ecc and mean RT/ecc slope \n')
print('correlation = %.6f, p-value = %.6f'%(cor_CSmean_RTecc_slope,pval_cor_CSmean_RTecc_slope))

if pval_cor_CSmean_RTecc_slope<p_value:
    print('SIGNIFICANT CORRELATION')

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

CS_RTECC_slope_df4plot = pd.DataFrame({'CS': np.mean(test_all_cs,axis=-1),
            'RT/ECC': slope_RT_ecc})

ax = sns.lmplot(x='RT/ECC', y='CS',data=CS_RTECC_slope_df4plot)
ax.set(ylabel='CS [dva]', xlabel='RT/ECC [a.u.]')
ax = plt.gca()
ax.set_title('CS vs RT/ECC (rho=%.2f,pval=%.3f)'%(cor_CSmean_RTecc_slope,pval_cor_CSmean_RTecc_slope))
plt.savefig(os.path.join(plot_dir,'CSvsRT_ECC_slope_across-ecc_%d-subs.svg'%(len(test_subs))), dpi=100,bbox_inches = 'tight')
    

# CS vs RT/SET SLOPE
cor_CSmean_RTset_slope, pval_cor_CSmean_RTset_slope = spearmanr(slope_RT_set,np.mean(test_all_cs,axis=-1))
print('\ncomparing critical distance mean across set and mean RT/set slope \n')
print('correlation = %.6f, p-value = %.6f'%(cor_CSmean_RTset_slope,pval_cor_CSmean_RTset_slope))

if pval_cor_CSmean_RTset_slope<p_value:
    print('SIGNIFICANT CORRELATION')

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

CS_RTset_slope_df4plot = pd.DataFrame({'CS': np.mean(test_all_cs,axis=-1),
            'RT/set': slope_RT_set})

ax = sns.lmplot(x='RT/set', y='CS',data=CS_RTset_slope_df4plot)
ax.set(ylabel='CS [dva]', xlabel='RT/set [a.u.]')
ax = plt.gca()
ax.set_title('CS vs RT/set (rho=%.2f,pval=%.3f)'%(cor_CSmean_RTset_slope,pval_cor_CSmean_RTset_slope))
plt.savefig(os.path.join(plot_dir,'CSvsRT_set_slope_across-ecc_%d-subs.svg'%(len(test_subs))), dpi=100,bbox_inches = 'tight')
    

# CS vs FIX/ECC SLOPE
cor_CSmean_fixecc_slope, pval_cor_CSmean_fixecc_slope = spearmanr(slope_fix_ecc,np.mean(test_all_cs,axis=-1))
print('\ncomparing critical distance mean across ecc and mean fix/ecc slope \n')
print('correlation = %.6f, p-value = %.6f'%(cor_CSmean_fixecc_slope,pval_cor_CSmean_fixecc_slope))

if pval_cor_CSmean_fixecc_slope<p_value:
    print('SIGNIFICANT CORRELATION')

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

CS_fixECC_slope_df4plot = pd.DataFrame({'CS': np.mean(test_all_cs,axis=-1),
            'fix/ECC': slope_fix_ecc})

ax = sns.lmplot(x='fix/ECC', y='CS',data=CS_fixECC_slope_df4plot)
ax.set(ylabel='CS [dva]', xlabel='fix/ECC [a.u.]')
ax = plt.gca()
ax.set_title('CS vs fix/ECC (rho=%.2f,pval=%.3f)'%(cor_CSmean_fixecc_slope,pval_cor_CSmean_fixecc_slope))
plt.savefig(os.path.join(plot_dir,'CSvsfix_ECC_slope_across-ecc_%d-subs.svg'%(len(test_subs))), dpi=100,bbox_inches = 'tight')
  

# CS vs fix/SET SLOPE
cor_CSmean_fixset_slope, pval_cor_CSmean_fixset_slope = spearmanr(slope_fix_set,np.mean(test_all_cs,axis=-1))
print('\ncomparing critical distance mean across set and mean fix/set slope \n')
print('correlation = %.6f, p-value = %.6f'%(cor_CSmean_fixset_slope,pval_cor_CSmean_fixset_slope))

if pval_cor_CSmean_fixset_slope<p_value:
    print('SIGNIFICANT CORRELATION')

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

CS_fixset_slope_df4plot = pd.DataFrame({'CS': np.mean(test_all_cs,axis=-1),
            'fix/set': slope_fix_set})

ax = sns.lmplot(x='fix/set', y='CS',data=CS_fixset_slope_df4plot)
ax.set(ylabel='CS [dva]', xlabel='fix/set [a.u.]')
ax = plt.gca()
ax.set_title('CS vs fix/set (rho=%.2f,pval=%.3f)'%(cor_CSmean_fixset_slope,pval_cor_CSmean_fixset_slope))
plt.savefig(os.path.join(plot_dir,'CSvsfix_set_slope_across-ecc_%d-subs.svg'%(len(test_subs))), dpi=100,bbox_inches = 'tight')
  

# CS vs on-object/ECC SLOPE
cor_CSmean_onobjecc_slope, pval_cor_CSmean_onobjecc_slope = spearmanr(slope_onobj_ecc,np.mean(test_all_cs,axis=-1))
print('\ncomparing critical distance mean across ecc and mean onobj/ecc slope \n')
print('correlation = %.6f, p-value = %.6f'%(cor_CSmean_onobjecc_slope,pval_cor_CSmean_onobjecc_slope))

if pval_cor_CSmean_onobjecc_slope<p_value:
    print('SIGNIFICANT CORRELATION')

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

CS_onobjECC_slope_df4plot = pd.DataFrame({'CS': np.mean(test_all_cs,axis=-1),
            'onobj/ECC': slope_onobj_ecc})

ax = sns.lmplot(x='onobj/ECC', y='CS',data=CS_onobjECC_slope_df4plot)
ax.set(ylabel='CS [dva]', xlabel='onobj/ECC [a.u.]')
ax = plt.gca()
ax.set_title('CS vs onobj/ECC (rho=%.2f,pval=%.3f)'%(cor_CSmean_onobjecc_slope,pval_cor_CSmean_onobjecc_slope))
plt.savefig(os.path.join(plot_dir,'CSvsonobj_ECC_slope_across-ecc_%d-subs.svg'%(len(test_subs))), dpi=100,bbox_inches = 'tight')
   
# CS vs on-object/SET SLOPE
cor_CSmean_onobjset_slope, pval_cor_CSmean_onobjset_slope = spearmanr(slope_onobj_set,np.mean(test_all_cs,axis=-1))
print('\ncomparing critical distance mean across set and mean onobj/set slope \n')
print('correlation = %.6f, p-value = %.6f'%(cor_CSmean_onobjset_slope,pval_cor_CSmean_onobjset_slope))

if pval_cor_CSmean_onobjset_slope<p_value:
    print('SIGNIFICANT CORRELATION')

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

CS_onobjset_slope_df4plot = pd.DataFrame({'CS': np.mean(test_all_cs,axis=-1),
            'onobj/set': slope_onobj_set})

ax = sns.lmplot(x='onobj/set', y='CS',data=CS_onobjset_slope_df4plot)
ax.set(ylabel='CS [dva]', xlabel='onobj/set [a.u.]')
ax = plt.gca()
ax.set_title('CS vs onobj/set (rho=%.2f,pval=%.3f)'%(cor_CSmean_onobjset_slope,pval_cor_CSmean_onobjset_slope))
plt.savefig(os.path.join(plot_dir,'CSvsonobj_set_slope_across-ecc_%d-subs.svg'%(len(test_subs))), dpi=100,bbox_inches = 'tight')
    


##### Things to add later ######

# also make some scanpath plots? anoying in terms of memory, because several trials
# should check if everything ok

# think of the density, how to calculate? define ROI and count amout of distractors? 
# Calculate area of ROI that has distractors? 

####################


##### 
# IF NEEDED, CAN CHOOSE A SUBJECT AND TRIAL TO PLOT THE SCANPATH OR RAW GAZE DATA ##
#
#
## dir to save scanpath plots
#scanpath_dir = os.path.join(plot_dir,'scanpaths')
#if not os.path.exists(scanpath_dir):
#     os.makedirs(scanpath_dir)
        
#draw_scanpath_display('86',400,output_vs,output_vs,scanpath_dir)

#draw_rawdata_display('86',400,output_vs,output_vs,scanpath_dir)





    










































































