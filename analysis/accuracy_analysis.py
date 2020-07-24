# accuracy analysis, to make extra figures/tables
# table for crowding accuracy
# accuracy ANOVA visual search
# correlations with crowding accuracy and RT/fixations

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
plot_dir = os.path.join(os.path.split(output_vs)[0],'plots','accuracy')
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
test_df_vs_accuracy = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])

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

        ## accuracy search 
        df_vs_ACC, _ = mean_ACC_set_ecc_combo(df_vs,sum_measures['all_subs'][j])
        # append matrix of combo 
        test_df_vs_accuracy = test_df_vs_accuracy.append(df_vs_ACC,ignore_index=True)
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


# make table with crowding accuracy
# with and without flankers
# Initialise data to lists. 
data_ACC_CRWD = [{'4 dva': np.mean([test_acc_fl[i][0] for i in range(len(test_acc_fl))]), 
                  '8 dva': np.mean([test_acc_fl[i][1] for i in range(len(test_acc_fl))]), 
                  '12 dva': np.mean([test_acc_fl[i][2] for i in range(len(test_acc_fl))]),
                  'Combined': np.mean(np.array(test_acc_fl).ravel())}, 
                 {'4 dva': np.mean([test_acc_nofl[i][0] for i in range(len(test_acc_nofl))]), 
                  '8 dva': np.mean([test_acc_nofl[i][1] for i in range(len(test_acc_nofl))]), 
                  '12 dva': np.mean([test_acc_nofl[i][2] for i in range(len(test_acc_nofl))]),
                  'Combined': np.mean(np.array(test_acc_nofl).ravel())}] 
  
# Creates padas DataFrame by passing  
# Lists of dictionaries and row index. 
df_ACC_CRWD = pd.DataFrame(data_ACC_CRWD, index =['Flankers', 'No Flankers']) 

df_ACC_CRWD.to_csv(os.path.join(plot_dir,'accuracy_crowding.csv'))


#pval_acc_crwd = wilcoxon(np.nanmean(np.array(test_acc_fl),axis=-1), np.nanmean(np.array(test_acc_nofl),axis=-1))[-1]

crwd_acc_no_fl = pd.DataFrame(data=np.array(test_acc_nofl), 
                             columns = ['4_dva','8_dva','12_dva'])

fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

sns.boxplot(x='variable',y='value', data=pd.melt(crwd_acc_no_fl))
sns.swarmplot(x="variable", y="value", data=pd.melt(crwd_acc_no_fl), color=".3")
plt.ylabel('Accuracy')
plt.xlabel('')
plt.title('Accuracy crowding (without flankers)')

# do Friedman to see if averages are different 
# The Friedman test tests the null hypothesis that repeated measurements of the same individuals have the same distribution. 
#It is often used to test for consistency among measurements obtained in different ways.

pfriedman = friedmanchisquare(np.array(test_acc_nofl)[...,0], 
                              np.array(test_acc_nofl)[...,1], 
                              np.array(test_acc_nofl)[...,2])[-1]
if pfriedman<p_value:
    print('Accuracy without flankers between ecc is different, friedman with p-value = %.6f'%pfriedman)
else:
    print('Accuracy without flankers between ecc is the same, p-value = %.6f'%pfriedman)


# For all ecc do separate violin plots
# colors a bit off between legend and violin, so corrected for in inkscape
for k,_ in enumerate(ecc):
    colors_ecc = np.array([['#ff8c8c','#e31a1c','#870000'],
                           ['#faad75','#ff6b00','#cc5702'],
                           ['#fff5a3','#eecb5f','#f4b900']])#['#ffea80','#fff200','#dbbe00']])

    columns2drop = np.array([['sub','8_ecc','12_ecc'],
                           ['sub','4_ecc','12_ecc'],
                           ['sub','4_ecc','8_ecc']])

    df4plot_vs_accuracy = test_df_vs_accuracy.drop(columns=columns2drop[k])
    df4plot_vs_accuracy = pd.melt(df4plot_vs_accuracy, 'set_size', var_name='Target eccentricity [dva]', value_name='Accuracy')

    fig = plt.figure(num=None, figsize=(7.5,7.5), dpi=100, facecolor='w', edgecolor='k')
    v1 = sns.violinplot(x='Target eccentricity [dva]', hue='set_size', y='Accuracy', data=df4plot_vs_accuracy,
                  cut=0, inner='box', palette=colors_ecc[k],linewidth=3)

    plt.legend().remove()
    plt.xticks([], [])
    #plt.xticks([0,1,2], ('5', '15', '30'))

    v1.set(xlabel=None)
    v1.set(ylabel=None)

    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    if k==0:
        plt.ylabel('Accuracy',fontsize=18,labelpad=10)
        plt.xlabel('Set Size [items]',fontsize=18,labelpad=35)
    plt.title('%d dva'%ecc[k],fontsize=22,pad=10)

    plt.ylim(0.7,1)
    plt.savefig(os.path.join(plot_dir,'search_ecc_ACC_violin_%decc.svg'%ecc[k]), dpi=100,bbox_inches = 'tight')
    

## IMPLEMENT ANOVA WITH STATSMODEL
## FOR ACCURACY ##

## reformat data frame

new_DF = pd.DataFrame(columns=['sub','set_size','ecc','accuracy'])

for _,sj in enumerate(test_subs): # loop over subject
    df_trim = test_df_vs_accuracy.loc[test_df_vs_accuracy['sub'] == sj]
    
    for _,s in enumerate(params['set_size']): # loop over set size
        
        new_df_trim = df_trim.loc[df_trim['set_size'] == s]

        for e_ind,e in enumerate(ecc): # loop over eccentricity
            
             new_DF = new_DF.append({
                                'sub': sj,
                                'set_size': s,
                                'ecc': e,
                                'accuracy': new_df_trim[str(e)+'_ecc'].values[0] 
                                 },ignore_index=True)
                
aovrm2way = AnovaRM(new_DF, depvar='accuracy', subject='sub', within=['set_size', 'ecc'])
res2way = aovrm2way.fit()

print(res2way)

res2way.anova_table.to_csv(os.path.join(plot_dir,'Accuracy_ANOVA.csv'))

#sns.boxplot(x="set_size", y="accuracy", hue="ecc", data=new_DF, palette="Set3") 

# Correlations between tasks
# Using crowding Accuracy of NO FLANKER (for appendix)

# The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets. 
# Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.


## FOR RT ##

corr_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])
pval_RT = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size'])

for _,s in enumerate(params['set_size']): # loop over set size
    
    df_trim = test_df_RT.loc[test_df_RT['set_size'] == s]
    df_trim = df_trim.drop(columns=['set_size'])
    
    for e_ind,e in enumerate(ecc): # loop over eccentricity (using the specific accuracy of NO FLANKERS for that eccentricity)
        
        corr,pval = plot_correlation(np.array(test_acc_nofl).T[e_ind],df_trim[str(e)+'_ecc'].values,
                    'Accuracy','RT [s]','Acc vs RT for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(plot_dir,'Acc_NOFLANK_vsRT_%s-ecc_%s-set.svg'%(str(e),str(s))),
                                     p_value=p_value,y_lim=[0.5,2.5],x_lim = [.6,1.01]) #[.4,1.01])
        
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
        
        corr,pval = plot_correlation(np.array(test_acc_nofl).T[e_ind],df_trim[str(e)+'_ecc'].values,
                    'Accuracy','# Fixations','Acc vs #Fix for %s ecc and %s items'%(str(e),str(s)),
                     os.path.join(plot_dir,'Acc_NOFLANK_vsFix_%s-ecc_%s-set.svg'%(str(e),str(s))),
                                     p_value=p_value,y_lim = [0,8],x_lim = [.6,1.01]) #[.6,1.01])
        
        # save correlation values         
        corr_fix = corr_fix.append({str(e)+'_ecc': corr, 
                                'set_size': s},ignore_index=True)
        
        # save p-values of said correlations          
        pval_fix = pval_fix.append({str(e)+'_ecc': pval, 
                                'set_size': s},ignore_index=True)











