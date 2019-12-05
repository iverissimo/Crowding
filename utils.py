
# to save common functions

# then I don't need to reapeat all the time


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

# functions

# convert edf file to asc file
def convert2asc(sj,taskname,outVS,outCRWD):
    
    import os, glob, shutil
    from pygazeanalyser.edfreader import read_edf
    
    if type(sj) == int: #if sub num integer
        sj = str(sj).zfill(2) #turn to string
    
    #list of absolute paths to all edf files in that session for that subject
    #single hdf5 file that contains all eye data for the runs of that session
    if taskname=='vs':
        edf_file = glob.glob(os.path.join(outVS, 'eyedata_visualsearch_pp_%s.edf' %sj))[0]
        asc_dir = outVS+'/pp-{sj}/'.format(sj=sj)
    elif taskname=='crowding':
        edf_file = glob.glob(os.path.join(outCRWD, 'eyedata_crowding_pp_%s.edf' %sj))[0]
        asc_dir = outCRWD+'/pp-{sj}/'.format(sj=sj)
           
    if not os.path.exists(asc_dir): # check if path to save hdf5 files exists
        os.makedirs(asc_dir)     # if not create it
    
    os.system("edf2asc %s" % edf_file)
    
    asc_file = os.path.split(edf_file)[1].replace('.edf','.asc')
    shutil.move(os.path.join(os.path.split(edf_file)[0],asc_file),asc_dir+asc_file)
    
    asc_filename = asc_dir+asc_file
    edfdata = read_edf(asc_filename, 'start_trial', stop='stop_trial', debug=False)
    
    return asc_filename, edfdata #name of asccii, actual gaze data

# turn visual angle in degrees
def ang2pix(dist_in_deg,h,d,r): 
    import math 
    
    deg_per_px = math.degrees(math.atan2(0.5*h,d))/(0.5*r)
    dist_in_px = dist_in_deg/deg_per_px
    return dist_in_px 

# function to draw all gaze points in trial on top of display
def draw_rawdata_display(sj,trial,filename,target,distr,gazedata,hRes,vRes,radius):
        
    import matplotlib.pyplot as plt
    import os
    COLS ={"aluminium": ['#eeeeec',
                    '#d3d7cf',
                    '#babdb6',
                    '#888a85',
                    '#555753',
                    '#2e3436'],
            }

    # set figure and main axis
    fig, ax = plt.subplots()
    # change default range 
    ax.set_xlim((0, hRes))
    ax.set_ylim((0, vRes))
    
    # add target as red circle
    trg_circle = plt.Circle((target[0], target[1]), radius, color='r')
    ax.add_artist(trg_circle)

    # add distractors as blue circles
    for i in range(len(distr)):
        distr_circle = plt.Circle((distr[i][0], distr[i][1]), radius, color='b')
        ax.add_artist(distr_circle)

    # add new axis that can be inverted, for gaze data
    # invert y axis, as (0,0) is top left on a display
    ax1 = ax.twinx()
    ax1.set_xlim((0, hRes))
    ax1.set_ylim((0, vRes))
    ax1.invert_yaxis()
    ax1.get_yaxis().set_visible(False) #but don't show it
    ax1.get_xaxis().set_visible(False)

    # plot raw data points
    x = gaze[0]
    y = gaze[1] # did this to invert y axis, as (0,0) is top left on a display (#ax.invert_yaxis() will do whole thing, dont want that)

    ax1.plot(gazedata[0], gazedata[1], 'o', color=COLS['aluminium'][0], markeredgecolor=COLS['aluminium'][5])

    plt.show()
    
    absfile = os.path.join(os.path.split(filename)[0], "rawdata_pp-%s_trial-%s.png" % (sj,str(trial).zfill(3)))
    if not os.path.exists(absfile): # if file not in dir, save
        fig.savefig(absfile, dpi=1000)

    
# function to draw fixation locations and saccade path in trial on top of display   
def draw_scanpath_display(sj,trial,filename,target,distr,fixation,saccades,hRes,vRes,radius):
    
    import matplotlib.pyplot as plt
    import os
    COLS ={"aluminium": ['#eeeeec',
                    '#d3d7cf',
                    '#babdb6',
                    '#888a85',
                    '#555753',
                    '#2e3436'],
            "chameleon": ['#8ae234',
                '#73d216',
                '#4e9a06']
            }
    
    # set figure and main axis
    fig, ax = plt.subplots()
    # change default range 
    ax.set_xlim((0, hRes))
    ax.set_ylim((0, vRes))
    
    # add target as red circle
    trg_circle = plt.Circle((target[0], target[1]), radius, color='r')
    ax.add_artist(trg_circle)

    # add distractors as blue circles
    for i in range(len(distr)):
        distr_circle = plt.Circle((distr[i][0], distr[i][1]), radius, color='b')
        ax.add_artist(distr_circle)


    # ADD DISPLAY
    # add new axis with different orientation of gaze data
    ax1 = ax.twinx()
    ax1.set_xlim((0, hRes))
    ax1.set_ylim((0, vRes))
    ax1.get_yaxis().set_visible(False) #but don't show it
    ax1.get_xaxis().set_visible(False)
    # invert the y axis, as (0,0) is top left on a display
    ax1.invert_yaxis()

    alpha = 0.5 # alpha level for scanpath drawings
    
    # FIXATIONS
    # parse fixations
    from pygazeanalyser.gazeplotter import parse_fixations
    fix = parse_fixations(fixations)
    # draw fixations, size of dot depends on duration
    ax1.scatter(fix['x'],fix['y'], s=fix['dur'], c=COLS['chameleon'][2], marker='o', cmap='jet', alpha=alpha, edgecolors='none')
    # draw annotations (fixation numbers)
    for i in range(len(fixations)):
        ax1.annotate(str(i+1), (fix['x'][i],fix['y'][i]), color=COLS['aluminium'][5], alpha=1, horizontalalignment='center', verticalalignment='center', multialignment='center')

    # SACCADES
    if saccades:
        # loop through all saccades
        for st, et, dur, sx, sy, ex, ey in saccades:
            # draw an arrow between every saccade start and ending
            ax1.arrow(sx, sy, ex-sx, ey-sy, alpha=alpha, fc=COLS['aluminium'][0], ec=COLS['aluminium'][5], fill=True, shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)
    
    absfile = os.path.join(os.path.split(filename)[0], "scanpath_pp-%s_trial-%s.png" % (sj,str(trial).zfill(3)))
    if not os.path.exists(absfile): # if file not in dir, save
        fig.savefig(absfile, dpi=1000)



def check_nan_trials(data,exclusion_thresh=0.25):
    # function to check for missed trials (no response)
    ### inputs ###
    # data - behav dataframe
    # exclusion_thresh - percentage of trials on which to exclude subject
    ### outputs ###
    # missed_trials - percentage of missed trials
    # exclude - bool saying if we should exclude sub or not
    
    exclude = False
    miss = data['key_pressed'].isna().sum() #easier way to count NaNs in dataframe
    # check for missed trials (trials where they didn't respond)
    missed_trials = miss/len(data)#len(miss)/len(data_sub)
    
    if missed_trials>exclusion_thresh:
        print('Subject missed %.2f %% of trials (no response) based on behavioural data \n'\
          'EXCLUDE SUBJECT'%(missed_trials*100))
        exclude = True
    else:
        print('Subject missed %.2f %% of trials (no response) based on behavioural data \n'\
              'continuing analysis for crowding'%(missed_trials*100))

    return missed_trials*100, exclude


def accuracy_crowding(data,ecc,exclusion_thresh=0.6):
    # function to check accuracy for crowding
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # ecc - list with eccs used in task
    # exclusion_thresh - accuracy value on which to exclude subject
    #
    # OUTPUTS #
    # acc_ecc_flank - accuracy for all ecc, with flankers condition
    # acc_ecc_noflank - accuracy for all ecc, without flankers condition
    # overall_acc - overall accuracy
    # exclude - bool saying if we should exclude sub or not
    
    exclude = False
    
    # initialize counters
    flank_corr = np.zeros((len(ecc))) # counter for correct answers, with flanker trials
    flank_ecc = np.zeros((len(ecc))) # ecc counter for with flanker trials
    noflank_corr = np.zeros((len(ecc))) # counter for correct answers, no flanker trials
    noflank_ecc = np.zeros((len(ecc))) # ecc counter for no flanker trials
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of strings indicating flanker presence/absent
    flnk_pres = data['flanker_presence'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    
    for i in range(len(data)): # for all actual trials (576) 
        
        # ecc index for this trial
        ecc_idx = np.where(np.array(ecc)==int(target_ecc[i]))[0][0]
        
        if flnk_pres[i]=='flankers': # if flanker trial
            flank_ecc[ecc_idx] += 1 #increment overall counter for right ecc
            
            # if key pressed and target orientation the same  
            if key_or[i]==target_or[i]:
                flank_corr[ecc_idx] += 1 #increment correct answer counter for right ecc
            
        
        else: #no flanker trial
            noflank_ecc[ecc_idx] += 1 #increment overall counter for right ecc

            # if key pressed and target orientation the same  
            if key_or[i]==target_or[i]:
                noflank_corr[ecc_idx] += 1 #increment correct answer counter for right ecc
            
    
    acc_ecc_flank = flank_corr/flank_ecc # accuracy with flankers, per ecc
    acc_ecc_noflank = noflank_corr/noflank_ecc # accuracy withou flankers, per ecc
    overall_acc = (flank_corr.sum()+noflank_corr.sum())/(flank_ecc.sum()+noflank_ecc.sum()) # overall accuracy
    print('Accuracy with flankers is %s %% for ecc %s'%(str(acc_ecc_flank*100),str(ecc)))
    print('Accuracy without flankers is %s %% for ecc %s'%(str(acc_ecc_noflank*100),str(ecc)))
    
    if overall_acc<exclusion_thresh:
        print('Subject has overall accuracy of %.2f %% \n'\
              'EXCLUDE SUBJECT'%(overall_acc*100))
        exclude = True
    else:
        print('Subject has overall accuracy of %.2f %% \n' \
              'continuing analysis for crowding'%(overall_acc*100))

    return {'acc_ecc_flank': acc_ecc_flank, 'acc_ecc_noflank': acc_ecc_noflank ,
            'overall_acc': overall_acc,'exclude':exclude} 


def critical_spacing(data,ecc,num_trl=96):
    # function to calculate critical spacing
    
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # ecc - list with eccs used in task
    # num_trl - number of trials to use for CS
    #
    # OUTPUTS #
    # all_tfr - all target flanker rations, per ecc
    # all_crit_dis - all critical distances, per ecc
    # mean_CS - mean critical spacing
    # exclude - bool saying if we should exclude sub or not
    
    exclude = False
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of strings indicating flanker presence/absent
    flnk_pres = data['flanker_presence'].values
    # list of values with the target flanker ratio
    target_fl_ratio = data['target_flank_ratio'].values
    
    all_tfr = [] # list to append all ecc tfr
    all_crit_dis = [] # list of critical distances for all ecc
    
    for _,j in enumerate(ecc):
        
        tfr = []
    
        for i in range(len(data)): # for all actual trials (576) 
            if flnk_pres[i]=='flankers' and int(target_ecc[i])==j: # if flanker trial and coreect ecc
                
                tfr.append(target_fl_ratio[i]) # append target flanker ratio
                
        all_tfr.append(np.array(tfr))
        all_crit_dis.append(np.median(tfr[-num_trl:]))
        
        
    mean_CS = np.mean(np.array(all_crit_dis))
    if np.round_(mean_CS,decimals=3) == 0.200 or np.round_(mean_CS,decimals=3) == 0.800:
        print('Mean critical spacing is %.3f for subjectso bottom/ceiling effect\n EXCLUDE SUBJECT'%(mean_CS))
        exclude = True
    else:
        print('Mean critical spacing is %.3f for subject'%(mean_CS))
        
    
    return {'all_tfr': all_tfr, 'all_crit_dis': all_crit_dis ,
            'mean_CS': mean_CS,'exclude':exclude} 



def accuracy_search(data,ecc,exclusion_all_thresh=0.85,exclusion_ecc_thresh=0.75):
    # function to check accuracy for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # ecc - list with eccs used in task
    # exclusion_thresh - accuracy value on which to exclude subject
    #
    # OUTPUTS #
    # acc_ecc - accuracy for all ecc
    # overall_acc - overall accuracy
    # exclude - bool saying if we should exclude sub or not
    
    exclude = False
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    
    # initialize counters
    ans_corr = np.zeros((len(ecc))) # counter for correct answers, with flanker trials
    ans_ecc = np.zeros((len(ecc))) # ecc counter for with flanker trials
        
    for i in range(len(data)): # for all actual trials 
        
        # ecc index for this trial
        ecc_idx = np.where(np.array(ecc)==int(target_ecc[i]))[0][0]
        
        ans_ecc[ecc_idx] += 1 #increment overall counter for right ecc
                    
        if key_or[i]==target_or[i]: # if key press = target orientation
            ans_corr[ecc_idx] += 1 #increment correct answer counter for right ecc

    acc_ecc = ans_corr/ans_ecc # accuracy, per ecc
    overall_acc = ans_corr.sum()/ans_ecc.sum() # overall accuracy
    print('Accuracy is %s %% for ecc %s'%(str(acc_ecc*100),str(ecc)))
    
    if overall_acc<exclusion_all_thresh or min(acc_ecc)<exclusion_ecc_thresh:
        print('Accuracy for visual search is %s %% for subject \n'\
              'EXCLUDE SUBJECT'%(str(acc_ecc*100)))
        exclude = True
    else:
        print('Accuracy for visual search is %s %% for subject'%(str(acc_ecc*100)))

    return {'acc_ecc': acc_ecc,
            'overall_acc': overall_acc,'exclude':exclude} 
        
                


def mean_RT(data,ecc):
    # function to check RT as function of ecc for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # RT_all - mean RT for all ecc
        
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    # list RT
    RT = data['RT'].values
    
    RT_all = [] # RT for all ecc
    
    for _,j in enumerate(ecc):
        
        RT_ecc = []
        
        for i in range(len(data)): # for all actual trials 

            if key_or[i]==target_or[i] and int(target_ecc[i])==j: # if key press = target orientation and correct ecc
                RT_ecc.append(RT[i]) #append RT value
        
        RT_all.append(np.mean(RT_ecc))
    
    return RT_all
        
 
def mean_RT_setsize(data,setsize):
    # function to check RT as function of set size for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # setsize - list with set sizes used in task
    #
    # OUTPUTS #
    # RT_all - mean RT for all set size
        
    # list of values with target ecc
    target_set = data['set_size'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    # list RT
    RT = data['RT'].values
    
    RT_all = [] # RT for all ecc
    
    for _,j in enumerate(setsize):
        
        RT_set = []
        
        for i in range(len(data)): # for all actual trials 

            if key_or[i]==target_or[i] and int(target_set[i])==j: # if key press = target orientation and correct ecc
                RT_set.append(RT[i]) #append RT value
        
        RT_all.append(np.mean(RT_set))
    
    return RT_all
           
    
def meanfix_ecc(data,eyedata,ecc):
    # function to check mean number of fixations as function of ecc for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # eyedata - eyetracking data
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # fix_all - mean fix for all ecc
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    
    fix_all = [] # RT for all ecc
    
    for _,j in enumerate(ecc):
        
        fix_ecc = []
        
        for i in range(len(data)): # for all actual trials 

            if key_or[i]==target_or[i] and int(target_ecc[i])==j: # if key press = target orientation and correct ecc
                fix_ecc.append(len(eyedata[i]['events']['Efix'])) #append RT value
        
        fix_all.append(np.mean(fix_ecc))
    
    return fix_all
    

def meanfix_setsize(data,eyedata,setsize):
    # function to check mean number of fixations as function of set size for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # eyedata - eyetracking data
    # setsize - list with set sizes used in task
    #
    # OUTPUTS #
    # fix_all - mean fix for all ecc

    # list of values with target set size
    target_set = data['set_size'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    
    fix_all = [] # RT for all ecc
    
    for _,j in enumerate(setsize):
        
        fix_set = []
        
        for i in range(len(data)): # for all actual trials 

            if key_or[i]==target_or[i] and int(target_set[i])==j: # if key press = target orientation and correct ecc
                fix_set.append(len(eyedata[i]['events']['Efix'])) #append RT value
        
        fix_all.append(np.mean(fix_set))
    
    return fix_all
    
    