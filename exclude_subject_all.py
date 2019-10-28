
# script to decide if we should exclude subject or not

# not finished, should add option for single sub vs all

# to produce plots for single subject or for all subject (that are not excluded)


import sys, os

# append Pygaz analyser folder, cloned from https://github.com/esdalmaijer/PyGazeAnalyser.git
sys.path.append(os.getcwd()+'/PyGazeAnalyser/')
import pygazeanalyser
import numpy as np
from pygazeanalyser.edfreader import read_edf
import pandas as pd
import glob

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
#%matplotlib inline

import json

#from utils import *

def ang2pix(dist_in_deg,h,d,r): 
    import math 
    
    deg_per_px = math.degrees(math.atan2(0.5*h,d))/(0.5*r)
    dist_in_px = dist_in_deg/deg_per_px
    return dist_in_px 

# paths
#base_dir =os.getcwd(); base_dir = os.path.join(base_dir,'Data2convert','Data7deg')

data = '5deg'

if data == '5deg':
    base_dir = r'C:\Users\Steffi\Desktop\Peripheral Vision Project\Data_pilot'
    subjects = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','16','17','18','19','20','21','21','22','23','24','25','26','27','28','29','30','31','32','33','34']

elif data == '7deg':
    base_dir = r'C:\Users\Steffi\Desktop\Peripheral Vision Project\Data'
    subjects = ['17','18','20','21','22','23','24','25','26','27']


output_vs = os.path.join(base_dir,'output_VS')
output_crwd = os.path.join(base_dir,'output_crowding')

#Dictionary to save exclusion data
dict_exclusion = {}
exclusion_reasons = {}

# open json parameter file
with open(os.path.join(os.getcwd(),'lab_parameters.json'),'r') as json_file:	
            analysis_params = json.load(json_file)	

# define dir to save plots
plot_dir = os.path.join(output_crwd,'plots')
if not os.path.exists(plot_dir):
     os.makedirs(plot_dir)


vRes = analysis_params['vRes']
hRes = analysis_params['hRes']


# get eyetracking data
# Load subjects
for sj in subjects:
    filename = os.path.join(output_crwd,'pp-%s'%sj,'eyedata_crowding_pp_%s.asc' %sj)
    eye_data_crowding = read_edf(filename, 'start_trial', stop='stop_trial', debug=False)
    print(sj, filename)
    
    EXCLUDE = False
    message = []
    # first get duration of each trial from start of trial to end 
    #(we'll check that by seeing the number of gaze points in a trial - ex:300 samples = 300 ms duration)
    first_trial_indx = 72 # first 72 trials were training block, doesn't count
    all_trial_dur = []
    for trl in range(first_trial_indx,len(eye_data_crowding)): 
        all_trial_dur.append(len(eye_data_crowding[trl]['trackertime']))
    
    
    # plot duration of all crowding trials 
    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(all_trial_dur)
    plt.xlabel('Trial #',fontsize=18)
    plt.ylabel('Time (ms)',fontsize=18)
    plt.axhline(y=np.median(all_trial_dur), color='r', linestyle='--')
    plt.title('Trial duration (median trial duration is %.2f ms)'%np.median(all_trial_dur))
    
    fig.savefig(os.path.join(plot_dir,'all_trials_duration_sub-{sj}.svg'.format(sj=str(sj).zfill(2))), dpi=100)
    
    exclusion_radius_deg = 1.5
    exclusion_radius_pix = ang2pix(exclusion_radius_deg,analysis_params['screenHeight'],analysis_params['screenDis'],analysis_params['vRes'])
    total_trials = 576
    
    
    # Look at saccades and see which have amplitude bigger than exclusion criteria ecc
    # events Esac - list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
    eye_trl_indx = []
    excluded_fix_radius = []
    
    for index in range(first_trial_indx,len(eye_data_crowding)):
        
        trl_counter = True # counter to update good trials
        
        if eye_data_crowding[index]['events']['Ssac']: # if saccade list not empty
            sacc_dist = [] # to save saccade distances for trial
            for _,sac in enumerate(eye_data_crowding[index]['events']['Esac']): # for each saccade in trial
                # x distance = endx - startx
                x_dist = sac[-2]-sac[-4]
                # y distance = endy - starty
                y_dist = sac[-1]-sac[-3]
                
                sacc_dist.append(np.sqrt(x_dist**2 + y_dist**2)) # pitagoras
                #print('saccade distance is %.2f'%sacc_dist)
            if sacc_dist and max(sacc_dist)>exclusion_radius_pix: # if hipotenusa bigger than exclusion distance
                #print('exclude trial %d'%index)
                excluded_fix_radius.append(max(sacc_dist))
                trl_counter = False
        
        if trl_counter: #if good trial counter still true        
            eye_trl_indx.append(index)
    
    # Calculate % of trials when they were not fixating properly
    exclud_trial_per = len(excluded_fix_radius)/total_trials * 100
    if exclud_trial_per>10:
        print('%.2f %% of trials were excluded for subject %s based on saccade info \n' \
                'EXCLUDE SUBJECT'%(exclud_trial_per,sj))
        EXCLUDE = True
        message.append('excluded saccades')
    else:
        print('%.2f %% of trials were excluded for subject %s based on saccade info \n' \
                'continuing analysis for crowding'%(exclud_trial_per,sj))
    
    
    # plot scatter of excluded trials where saccade info is bigger than 1 deg
    # and include lines where target eccentricities are and exclusion criteria too
    ecc_trgt_pix = [ang2pix(ecc,analysis_params['screenHeight'],analysis_params['screenDis'],analysis_params['vRes']) 
                    for _,ecc in enumerate(analysis_params['ecc'])]
    
    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
    
    plt.scatter(range(len(excluded_fix_radius)),excluded_fix_radius)
    plt.axhline(y=exclusion_radius_pix, color='black', linestyle='-')
    plt.axhline(y=ecc_trgt_pix[0], color='r', linestyle='--')
    plt.axhline(y=ecc_trgt_pix[1], color='m', linestyle='--')
    plt.axhline(y=ecc_trgt_pix[2], color='g', linestyle='--')
    
    plt.xlabel('excluded trials',fontsize=18)
    plt.ylabel('eccentricity (pixel)',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['exclusion line','4deg','8deg','12deg'], fontsize=10)
    #plt.show()
    
    fig.savefig(os.path.join(plot_dir,'excluded_sacc_amplitude_sub-{sj}.png'.format(sj=str(sj).zfill(2))), dpi=100)
    
    #first block (first 144) where used to save training, doesn't count
    behv_start_index = 144
    
    
    # now check from the percentage of missed trials - i.e. that they didn't reply
    # read dataframe for subject
    df_crwd = pd.read_csv(os.path.join(output_crwd,'data_crowding_pp_%s.csv'%sj), sep='\t')
    alltrials_df = df_crwd.loc[behv_start_index::] #filter training block out from behavioural data
    
    # check amount of trials where key pressed = nan
    miss = alltrials_df['key_pressed'].isna().sum()  
    missed_trials = miss/total_trials
    
    if missed_trials>0.25:
        print('Subject %s missed %.2f %% of trials (no response) based on behavioural data \n'\
              'EXCLUDE SUBJECT'%(sj,missed_trials*100))
        EXCLUDE = True
        message.append('missed trials')
    else:
        print('Subject %s missed %.2f %% of trials (no response) based on behavioural data \n'\
              'continuing analysis for crowding'%(sj,missed_trials*100))
    
    
    # Calcutate accuracy for crowding 
    cut_off = 0.6 # level upon which we don't consider performance to be chance level
    
    # index of missed trials from alltrials_df 
    miss_index = np.where(alltrials_df['key_pressed'].isna())[0]
    
    flank_corr = np.zeros((len(analysis_params['ecc']))) # counter for correct answers, with flanker trials
    flank_ecc = np.zeros((len(analysis_params['ecc']))) # ecc counter for with flanker trials
    noflank_corr = np.zeros((len(analysis_params['ecc']))) # counter for correct answers, no flanker trials
    noflank_ecc = np.zeros((len(analysis_params['ecc']))) # ecc counter for no flanker trials
    
    ratio_ecc = np.zeros((len(analysis_params['ecc']),160)) # counter for ratio (staircased val) per ecc
    counter = np.zeros((len(analysis_params['ecc']))) # counter to help in loop
    
    for i in range(len(alltrials_df)): # for all actual trials (576) 
        
        # get index of ecc val for this trial ex: ecc=4 -> ecc_idx = 0
        ecc_idx = np.where(np.array(analysis_params['ecc'])==int(alltrials_df['target_ecc'][i+behv_start_index]))[0][0]
        
        if alltrials_df['flanker_presence'][i+behv_start_index] == 'flankers': # if flanker trial
            
            flank_ecc[ecc_idx] += 1 #increment overall counter for right ecc
            # save ratio of target flanker distance per ecc, for later 
            ratio_ecc[ecc_idx][int(counter[ecc_idx])]=alltrials_df['target_flank_ratio'][i+behv_start_index]
            counter[ecc_idx] += 1 #increment counter
            
            if i in miss_index: # if missed trial skip 
                print('Trial %d/%d was miss trial'%(i,total_trials))
            
            # if key pressed and target orientation the same  
            elif alltrials_df['key_pressed'][i+behv_start_index]==alltrials_df['target_orientation'][i+behv_start_index]:
                flank_corr[ecc_idx] += 1 #increment correct answer counter for right ecc
    
        else: #no flanker trial
            
            noflank_ecc[ecc_idx] += 1 #increment overall counter for right ecc
            
            if i in miss_index: # if missed trial skip 
                print('Trial %d/%d was miss trial'%(i,total_trials))
            
            # if key pressed and target orientation the same  
            elif alltrials_df['key_pressed'][i+behv_start_index]==alltrials_df['target_orientation'][i+behv_start_index]:
                noflank_corr[ecc_idx] += 1 #increment correct answer counter for right ecc
    
    acc_ecc_flank = flank_corr/flank_ecc # accuracy with flankers, per ecc
    acc_ecc_noflank = noflank_corr/noflank_ecc # accuracy withou flankers, per ecc
    overall_acc = (flank_corr.sum()+noflank_corr.sum())/(flank_ecc.sum()+noflank_ecc.sum()) # overall accuracy
    print('Accuracy with flankers is %s %% for ecc [4,8,12]'%str(acc_ecc_flank*100))
    print('Accuracy without flankers is %s %% for ecc [4,8,12]'%str(acc_ecc_noflank*100))
    if overall_acc<cut_off:
        print('Subject %s has overall accuracy of %.2f %% \n'\
              'EXCLUDE SUBJECT'%(sj,overall_acc*100))
        EXCLUDE = True
        message.append('accuracy crowding')
    else:
        print('Subject %s has overall accuracy of %.2f %% \n' \
              'continuing analysis for crowding'%(sj,overall_acc*100))
    
    
    # plot staircase just to see what's going on
    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
    
    plt.plot(ratio_ecc[0])
    plt.plot(ratio_ecc[1])
    plt.plot(ratio_ecc[2])
    
    plt.ylabel("Target-flanker distance (ratio of ecc)",fontsize=18)
    plt.xlabel("# Trial",fontsize=18)
    plt.title("Target-Flanker ratio per ecc",fontsize=18)
    plt.legend(analysis_params['ecc'])
    #plt.show()
    
    fig.savefig(os.path.join(plot_dir,'TFR_per_ecc_sub-{sj}.svg'.format(sj=str(sj).zfill(2))), dpi=100)
    
    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
    
    plt.plot(alltrials_df['target_flank_ratio'])
    
    plt.ylabel("Target-flanker distance (ratio of ecc)",fontsize=18)
    plt.xlabel("# Trial",fontsize=18)
    plt.title("Target-Flanker ratio all trials",fontsize=18)
    #plt.show()
    
    fig.savefig(os.path.join(plot_dir,'TFR_all_sub-{sj}.svg'.format(sj=str(sj).zfill(2))), dpi=100)
    
    
    # Calculate critical spacing value for participant
    last_trls = 96 # number of last trials to use to compute critical spacing value
    
    crit_dis = []
    for i in range(len(analysis_params['ecc'])):
        crit_dis.append(np.median(ratio_ecc[i][-last_trls:])) #median of last 96 trials
    crit_dis = np.array(crit_dis)
    
    mean_CS = np.mean(crit_dis)
    if np.round_(mean_CS,decimals=3) == 0.200 or np.round_(mean_CS,decimals=3) == 0.800:
        print('Mean critical spacing is %.3f for subject %s so bottom/ceiling effect\n EXCLUDE SUBJECT'%(mean_CS,sj))
        EXCLUDE = True
        message.append('bottom/ceiling CS')
    else:
        print('Mean critical spacing is %.3f for subject %s'%(mean_CS,sj))
    
    
    # Now look at Visual search
    print('loading behavioural data for subject %s visual search'%(sj))
    df_vs = pd.read_csv(os.path.join(output_vs,'data_visualsearch_pp_%s.csv'%sj), sep='\t')
    
    # accuracy per ecc
    corr_ecc_vs = np.zeros((len(analysis_params['ecc']))) # correct ecc counter
    for _,i in enumerate(np.where(df_vs['key_pressed']==df_vs['target_orientation'])[0]):
        # get index of ecc val for this trial ex: ecc=4 -> ecc_idx = 0
        ecc_idx = np.where(np.array(analysis_params['ecc'])==df_vs['target_ecc'][i])[0][0]
        corr_ecc_vs[ecc_idx] += 1
    acc_vs_ecc = corr_ecc_vs/(len(df_vs)/len(analysis_params['ecc']))
    
    if min(acc_vs_ecc)<0.75:
        print('Accuracy for visual search is %s %% for subject %s \n EXCLUDE SUBJECT'%(str(acc_vs_ecc*100),sj))
        EXCLUDE = True
        message.append('accuracy VS ecc')
    else:
        print('Accuracy for visual search is %s %% for subject %s'%(str(acc_vs_ecc*100),sj))
        
    # overall accuracy
    #acc_vs = len(np.where(df_vs['key_pressed']==df_vs['target_orientation'])[0])/len(df_vs)
    acc_vs = np.mean(acc_vs_ecc)
    if acc_vs<0.85:
        print('Accuracy for visual search is %.2f %% for subject %s \n EXCLUDE SUBJECT'%(acc_vs*100,sj))
        EXCLUDE = True
        message.append('accuracy VS overall')
    else:
        print('Accuracy for visual search is %.2f %% for subject %s'%(acc_vs*100,sj))

    
    #Add exclusion parameter to dictionary
    if EXCLUDE == True:
        dict_exclusion.update({sj:'excluded'})
        exclusion_reasons.update({sj:message})
    else:
        dict_exclusion.update({sj:'included'})

print(dict_exclusion)
print(exclusion_reasons)
#Save exclusion dict
#excl_sj = pd.DataFrame(dict_exclusion)
#excl_sj.to_csv(base_dir, sep='\t')  










