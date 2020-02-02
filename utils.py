
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
from pygazeanalyser.gazeplotter import parse_fixations
from scipy.stats import spearmanr
import seaborn as sns

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

def draw_rawdata_display(sj,trial,df_vs_dir,eyedata_vs_dir,outdir,
                          rad_gab=1.1,fix_line=0.25,screenHeight=30,screenDis=57,
                         vRes = 1050,hRes=1680):
    
    # function to draw raw gaze for a specific trial
    # will draw display (target and distractor position)
    # and x-y gaze throughout time
    #
    #### INPUTS #####
    # sj - subject number (str)
    # trial - trial number (int)
    # df_vs - absolute path dataframe with behaviour data
    # eyedata_vs - absolute path eye data for visual search
    
    behvfile = [x for _,x in enumerate(os.listdir(df_vs_dir)) if x.endswith('.csv') and sj in x]
    eyefile = [x for _,x in enumerate(os.listdir(eyedata_vs_dir)) if x.endswith('.EDF') and sj in x]
    # load csv for sub
    df_vs = pd.read_csv(os.path.join(df_vs_dir,behvfile[0]), sep='\t')
    # load eye data
    _, eyedata_vs = convert2asc(sj,'vs',eyedata_vs_dir,eyedata_vs_dir)
    
    # NOTE - all positions in pixels
    
    # get target and distractor positions as strings in list
    target_pos = df_vs['target_position'][trial].replace(']','').replace('[','').split(' ')
    distr_pos = df_vs['distractor_position'][trial].replace(']','').replace('[','').replace(',','').split(' ')

    # convert to list of floats
    target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])
    distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])

    # save distractor positions in pairs (x,y)
    alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])
    
    f, s = plt.subplots(1, 1, figsize=(8,8))

    # set radius of circles
    r_gabor = ang2pix(rad_gab,screenHeight,
                       screenDis,vRes)
    r_fix = ang2pix(fix_line,screenHeight,
                       screenDis,vRes)
    
    # add target
    s.add_artist(plt.Circle((target_pos[0], target_pos[1]), radius=r_gabor, color='r', fill=True))
    s.set_xlim([-hRes/2,hRes/2])
    s.set_ylim([-vRes/2,vRes/2])
    s.set_aspect('equal') # to not make circles elipses
    
    # add disctractors
    for w in range(len(alldistr_pos)):
        s.add_artist(plt.Circle((alldistr_pos[w][0], alldistr_pos[w][1]), radius=r_gabor, color='b', fill=True))

    # add fixation cross
    s.add_artist(plt.Circle((0, 0), radius=r_fix, color='k', fill=True))

    # add circle of contingency 
    s.add_artist(plt.Circle((0, 0),radius=ang2pix(1,screenHeight,screenDis,vRes), color='k', fill=False))
    
    # plot raw data points
    x = eyedata_vs[trial]['x'] - hRes/2
    y = eyedata_vs[trial]['y'] - vRes/2 
    y = -y # have to invert y, because eyelink considers 0 to be upper corner

    s.plot(x, y, 'o', color='#eeeeec', markeredgecolor='#2e3436',alpha=0.5)

    f.savefig(os.path.join(outdir,'rawgaze_pp-%s_trial-%s.png' % (sj,str(trial).zfill(3))), dpi=1000)
    
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
    ax.plot(np.linspace(0,df_vs['RT'][trial],num=len(x)),x,color='k',label='x')
    ax.plot(np.linspace(0,df_vs['RT'][trial],num=len(y)),y,color='orange',label='y')
    
    plt.title('raw gaze trial %d sub %s'%(trial,sj))
    ax.set(ylabel='position [pixels]', xlabel='time [s]')
    plt.legend(loc='upper left')
    
    fig.savefig(os.path.join(outdir,'rawgaze_xy_pp-%s_trial-%s.png' % (sj,str(trial).zfill(3))), dpi=1000)
    

    
def draw_scanpath_display(sj,trial,df_vs_dir,eyedata_vs_dir,outdir,
                          rad_gab=1.1,fix_line=0.25,screenHeight=30,screenDis=57,
                         vRes = 1050,hRes=1680):
    # function to draw scanpath for a specific trial
    # will draw display (target and distractor position)
    # fixation positions and saccade path
    #
    #### INPUTS #####
    # sj - subject number (str)
    # trial - trial number (int)
    # df_vs - absolute path dataframe with behaviour data
    # eyedata_vs - absolute path eye data for visual search

    
    behvfile = [x for _,x in enumerate(os.listdir(df_vs_dir)) if x.endswith('.csv') and sj in x]
    eyefile = [x for _,x in enumerate(os.listdir(eyedata_vs_dir)) if x.endswith('.EDF') and sj in x]
    # load csv for sub
    df_vs = pd.read_csv(os.path.join(df_vs_dir,behvfile[0]), sep='\t')
    # load eye data
    _, eyedata_vs = convert2asc(sj,'vs',eyedata_vs_dir,eyedata_vs_dir)
    
    # NOTE - all positions in pixels
    
    # get target and distractor positions as strings in list
    target_pos = df_vs['target_position'][trial].replace(']','').replace('[','').split(' ')
    distr_pos = df_vs['distractor_position'][trial].replace(']','').replace('[','').replace(',','').split(' ')

    # convert to list of floats
    target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])
    distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])

    # save distractor positions in pairs (x,y)
    alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])
    
    f, s = plt.subplots(1, 1, figsize=(8,8))

    # set radius of circles
    r_gabor = ang2pix(rad_gab,screenHeight,
                       screenDis,vRes)
    r_fix = ang2pix(fix_line,screenHeight,
                       screenDis,vRes)
    
    # add target
    s.add_artist(plt.Circle((target_pos[0], target_pos[1]), radius=r_gabor, color='r', fill=True))
    s.set_xlim([-hRes/2,hRes/2])
    s.set_ylim([-vRes/2,vRes/2])
    s.set_aspect('equal') # to not make circles elipses
    
    # add disctractors
    for w in range(len(alldistr_pos)):
        s.add_artist(plt.Circle((alldistr_pos[w][0], alldistr_pos[w][1]), radius=r_gabor, color='b', fill=True))

    # add fixation cross
    s.add_artist(plt.Circle((0, 0), radius=r_fix, color='k', fill=True))

    # add circle of contingency 
    s.add_artist(plt.Circle((0, 0),radius=ang2pix(1,screenHeight,screenDis,vRes), color='k', fill=False))


    # FIXATIONS
    # parse fixations
    # fix is dictionary with list of x,y, and duration of fixations for trial
    fix = parse_fixations(eyedata_vs[trial]['events']['Efix'])

    # SACCADES
    if eyedata_vs[trial]['events']['Esac']:
        # loop through all saccades
        for st, et, dur, sx, sy, ex, ey in eyedata_vs[trial]['events']['Esac']:

            # make positions compatible with display
            sx = sx - hRes/2 # start x pos
            ex = ex - hRes/2 # end x pos
            sy = sy - vRes/2; sy = -sy #start y pos
            ey = ey - vRes/2; ey = -ey #end y pos

            # draw an arrow between every saccade start and ending
            s.arrow(sx, sy, ex-sx, ey-sy, alpha=0.5, fc='k', 
                    fill=True, shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)

    # make positions compatible with display
    fix['x'] = fix['x'] - hRes/2
    fix['y'] = fix['y'] - vRes/2; fix['y'] = -fix['y']

    # draw fixations, size of dot depends on duration
    s.scatter(fix['x'],fix['y'], s=fix['dur'], c='grey', marker='o', cmap='jet', alpha=0.5, edgecolors='none')

    # draw annotations (fixation numbers)
    for z in range(len(fix['x'])):
        s.annotate(str(z+1), (fix['x'][z],fix['y'][z]), 
                   color='w', alpha=1, horizontalalignment='center', verticalalignment='center', multialignment='center')
    
    f.savefig(os.path.join(outdir,'scanpath_pp-%s_trial-%s.png' % (sj,str(trial).zfill(3))), dpi=1000)





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
    if np.round_(mean_CS,decimals=3) < 0.250 or np.round_(mean_CS,decimals=3) > 0.750:
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
           
    
def meanfix_ecc(data,eyedata,ecc,hRes=1680,vRes=1050,screenHeight=30,screenDis=57,size_gab=2.2):
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
    # list of values of RT
    RT = data['RT'].values
    # radius of gabor in pixels
    r_gabor = ang2pix(size_gab/2,screenHeight,
                       screenDis,
                       vRes)
    # number of samples in 150ms (we'll not count with fixations prior to 150ms after stim display)
    sample_thresh = 1000*0.150 # 1000Hz * time in seconds
    
    fix_all = [] # RT for all ecc
    
    for _,j in enumerate(ecc):
        
        fix_ecc = []
        
        for i in range(len(data)): # for all actual trials 
            
            if key_or[i]==target_or[i] and int(target_ecc[i])==j: # if key press = target orientation and correct ecc
                
                # index for moment when display was shown
                idx_display = np.where(np.array(eyedata[i]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                # eye tracker sample time of display
                smp_display = eyedata[i]['events']['msg'][idx_display][0]

                # get target positions as strings in list
                target_pos = data['target_position'][i].replace(']','').replace('[','').split(' ')
                # convert to list of floats
                target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])
                
                num_fix = 0
                for k,fix in enumerate(eyedata[i]['events']['Efix']):
                    
                    # if fixations between 150ms after display and key press time
                    if (fix[0] > (smp_display+sample_thresh) and fix[0] < np.round(smp_display + RT[i]*1000)):
                        
                        #if fixation not on target (not within target radius)
                        fix_x = fix[-2] - hRes/2
                        fix_y = fix[-1] - vRes/2; fix_y = - fix_y

                        if np.sqrt((fix_x-target_pos[0])**2+(fix_y-target_pos[1])**2) > r_gabor:
                            num_fix += 1
   
                fix_ecc.append(num_fix) #append number of fixations for that trial value
        
        fix_all.append(np.mean(fix_ecc)) # append mean number of fixations per ecc
    
    return fix_all
    

def meanfix_setsize(data,eyedata,setsize,hRes=1680,vRes=1050,screenHeight=30,screenDis=57,size_gab=2.2):
    # function to check mean number of fixations as function of set size for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # eyedata - eyetracking data
    # setsize - list with set sizes used in task
    #
    # OUTPUTS #
    # fix_all - mean fix for all set sizes

    # list of values with target set size
    target_set = data['set_size'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    # list of values of RT
    RT = data['RT'].values
    # radius of gabor in pixels
    r_gabor = ang2pix(size_gab/2,screenHeight,
                       screenDis,
                       vRes)
    # number of samples in 150ms (we'll not count with fixations prior to 150ms after stim display)
    sample_thresh = 1000*0.150 # 1000Hz * time in seconds
    
    fix_all = [] # fications for all set sizes
    
    for _,j in enumerate(setsize):
        
        fix_set = []
        
        for i in range(len(data)): # for all actual trials 

            if key_or[i]==target_or[i] and int(target_set[i])==j: # if key press = target orientation and correct set size
                
                # index for moment when display was shown
                idx_display = np.where(np.array(eyedata[i]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                # eye tracker sample time of display
                smp_display = eyedata[i]['events']['msg'][idx_display][0]

                # get target positions as strings in list
                target_pos = data['target_position'][i].replace(']','').replace('[','').split(' ')
                # convert to list of floats
                target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])
                
                num_fix = 0
                for k,fix in enumerate(eyedata[i]['events']['Efix']):
                    
                    # if fixations between 150ms after display and key press time
                    if (fix[0] > (smp_display+sample_thresh) and fix[0] < np.round(smp_display + RT[i]*1000)):
                        
                        #if fixation not on target (not within target radius)
                        fix_x = fix[-2] - hRes/2
                        fix_y = fix[-1] - vRes/2; fix_y = - fix_y

                        if np.sqrt((fix_x-target_pos[0])**2+(fix_y-target_pos[1])**2) > r_gabor:
                            num_fix += 1
                
            
                fix_set.append(num_fix) #append number of fixations for that trial value
        
        fix_all.append(np.mean(fix_set)) # append mean number of fixations per ecc
    
    return fix_all
    
    
def on_objectfix_ecc(data,eyedata,ecc,radius,hRes=1680,vRes=1050,screenHeight=30,screenDis=57):
    # function to check percentage of on object fixations as function of ecc size for visual search
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
    # list of values of RT
    RT = data['RT'].values
    # radius of gabor in pixels
    radius_pix = ang2pix(radius,screenHeight,
                       screenDis,
                       vRes)
        
    # number of samples in 150ms (we'll not count with fixations prior to 150ms after stim display)
    sample_thresh = 1000*0.150 # 1000Hz * time in seconds
    
    fix_all = [] # fix for all ecc
    
    for _,j in enumerate(ecc):
        
        fix_ecc = []
        
        for i in range(len(data)): # for all actual trials 
            
            if key_or[i]==target_or[i] and int(target_ecc[i])==j: # if key press = target orientation and correct ecc
                
                # index for moment when display was shown
                idx_display = np.where(np.array(eyedata[i]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                # eye tracker sample time of display
                smp_display = eyedata[i]['events']['msg'][idx_display][0]
                
                num_fix = 0
                for k,fix in enumerate(eyedata[i]['events']['Efix']):
                    
                    # if fixations between 150ms after display and key press time
                    if (fix[0] > (smp_display+sample_thresh) and fix[0] < np.round(smp_display + RT[i]*1000)):
                        
                        #if fixation not on target (not within target radius)
                        fix_x = fix[-2] - hRes/2
                        fix_y = fix[-1] - vRes/2; fix_y = - fix_y
                        
                        # get distractor positions as strings in list
                        distr_pos = data['distractor_position'][i].replace(']','').replace('[','').replace(',','').split(' ')
                        # convert to list of floats
                        distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])
                        # save distractor positions in pairs (x,y)
                        alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])

                        # if fixation within radius of any of the distractors
                        for n in range(len(alldistr_pos)): 
                            if np.sqrt((fix_x-alldistr_pos[n][0])**2+(fix_y-alldistr_pos[n][1])**2) < radius_pix:
                                num_fix += 1 # save fixation
                
                if len(eyedata[i]['events']['Efix'])==0:   # if empty, to avoid division by 0
                    on_obj_per = 0
                else:
                    on_obj_per = num_fix/len(eyedata[i]['events']['Efix'])

                fix_ecc.append(on_obj_per) #append percentage of on object fixations of trial

        fix_all.append(np.mean(fix_ecc)) # append mean percentage of on object fixations per ecc
    
    return fix_all



def on_objectfix_set(data,eyedata,setsize,radius,hRes=1680,vRes=1050,screenHeight=30,screenDis=57):
    # function to check percentage of on object fixations as function of set size for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # eyedata - eyetracking data
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # fix_all - mean fix for all ecc
    
    # list of values with target set size
    target_set = data['set_size'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    # list of values of RT
    RT = data['RT'].values
    # radius of gabor in pixels
    radius_pix = ang2pix(radius,screenHeight,
                       screenDis,
                       vRes)
        
    # number of samples in 150ms (we'll not count with fixations prior to 150ms after stim display)
    sample_thresh = 1000*0.150 # 1000Hz * time in seconds
    
    fix_all = [] # fix for all ecc
    
    for _,j in enumerate(setsize):
        
        fix_set = []
        
        for i in range(len(data)): # for all actual trials 
            
            if key_or[i]==target_or[i] and int(target_set[i])==j: # if key press = target orientation and correct set size
                
                # index for moment when display was shown
                idx_display = np.where(np.array(eyedata[i]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                # eye tracker sample time of display
                smp_display = eyedata[i]['events']['msg'][idx_display][0]
                
                num_fix = 0
                for k,fix in enumerate(eyedata[i]['events']['Efix']):
                    
                    # if fixations between 150ms after display and key press time
                    if (fix[0] > (smp_display+sample_thresh) and fix[0] < np.round(smp_display + RT[i]*1000)):
                        
                        #if fixation not on target (not within target radius)
                        fix_x = fix[-2] - hRes/2
                        fix_y = fix[-1] - vRes/2; fix_y = - fix_y
                        
                        # get distractor positions as strings in list
                        distr_pos = data['distractor_position'][i].replace(']','').replace('[','').replace(',','').split(' ')
                        # convert to list of floats
                        distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])
                        # save distractor positions in pairs (x,y)
                        alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])

                        # if fixation within radius of any of the distractors
                        for n in range(len(alldistr_pos)): 
                            if np.sqrt((fix_x-alldistr_pos[n][0])**2+(fix_y-alldistr_pos[n][1])**2) < radius_pix:
                                num_fix += 1 # save fixation
                
                if len(eyedata[i]['events']['Efix'])==0:   # if empty, to avoid division by 0
                    on_obj_per = 0
                else:
                    on_obj_per = num_fix/len(eyedata[i]['events']['Efix'])

                fix_set.append(on_obj_per) #append percentage of on object fixations of trial

        fix_all.append(np.mean(fix_set)) # append mean percentage of on object fixations per set size
    
    return fix_all

def plot_correlation(arr_x,arr_y,label_x,label_y,plt_title,outfile,p_value=0.05):
    
    corr, pval = spearmanr(arr_x,arr_y)

    print('correlation = %.6f, p-value = %.6f'%(corr,pval))
    if pval<p_value:
        print('SIGNIFICANT CORRELATION')

    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    df4plot = pd.DataFrame({'xx': arr_x,
                            'yy': arr_y})

    ax = sns.lmplot(x='xx', y='yy',data=df4plot)
    ax.set(ylabel = label_y, xlabel = label_x)
    ax = plt.gca()
    ax.set_title(plt_title+' (rho=%.2f,pval=%.3f)'%(corr,pval))
    plt.savefig(outfile, dpi=100,bbox_inches = 'tight')


def surface_density(df_vs_dir,rad_roi,
                    rad_gab=1.1,screenHeight=30,screenDis=57,
                    vRes = 1050,hRes=1680):
    
    # function to calculate surface density values for all trials of visual search for subject
    #### INPUTS #####
    # df_vs - absolute path dataframe with behaviour data
    # rad_roi - radius around target, will be the ROI
    
    # returns
    # surf_density for all trials
    
    df_vs = pd.read_csv(df_vs_dir, sep='\t')
    
    # set radius around target to compute density (ROI)
    radius_roi = ang2pix(rad_roi,screenHeight,
                       screenDis,vRes)
    # radius of each gabor
    radius_gabor = ang2pix(rad_gab,screenHeight,
                       screenDis,vRes)

    # area of ROI to analyse
    roi_area = np.pi*(radius_roi)**2 - np.pi*(radius_gabor)**2 # subtract area of target 
    
    surf_density = []
    
    for trial in range(len(df_vs)): # for all trials
        
        # NOTE - all positions in pixels
    
        # get target and distractor positions as strings in list
        target_pos = df_vs['target_position'][trial].replace(']','').replace('[','').split(' ')
        distr_pos = df_vs['distractor_position'][trial].replace(']','').replace('[','').replace(',','').split(' ')

        # convert to list of floats
        target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])
        distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])

        # save distractor positions in pairs (x,y)
        alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])
        
        # convert positions to make everything positive and hence my life easier
        # (changing coordinate axis to have center position in bottom left of display 
        # instead of center position in center of screen)

        new_target_pos = target_pos + [hRes/2,vRes/2]
        new_alldistr_pos = alldistr_pos + [hRes/2,vRes/2]
        
        # compute total distractor area within ROI
        intersect_area = []

        for i in range(len(new_alldistr_pos)): # for all distractors
            # calculate distance between center of target and center of distractor
            distance = np.sqrt((new_target_pos[0]-new_alldistr_pos[i][0])**2+(new_target_pos[1]-new_alldistr_pos[i][1])**2)

            if distance >= radius_roi+radius_gabor: # circles not intersecting
                inters = 0

            elif distance + radius_gabor <= radius_roi: # distractor fully within ROI
                inters = np.pi*(radius_gabor)**2

            else: # distractor partially intersecting ROI
                # whole description of formula in https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6
                d1 = ((radius_roi)**2-(radius_gabor)**2+distance**2)/(2*distance)
                d2 = ((radius_gabor)**2-(radius_roi)**2+distance**2)/(2*distance)

                A1 = ((radius_roi)**2)*np.arccos(d1/radius_roi) - d1*np.sqrt((radius_roi)**2-d1**2)
                A2 = ((radius_gabor)**2)*np.arccos(d2/radius_gabor) - d2*np.sqrt((radius_gabor)**2-d2**2)

                inters = A1+A2

            intersect_area.append(inters)

        # sum of distractor areas
        dist_area = sum(intersect_area)

        if dist_area>roi_area: # if sum of distractor area bigger than ROI throw error, something is off
            raise ValueError('sum of disctractor area bigger than ROI for trial %d!'%trial)

        # surface density for trial    
        surf_density.append(dist_area/roi_area)

    return np.array(surf_density)


def density_mean_RT(data,density_arr,type_trial='ecc',density='high',threshold=0.03, ecc=[4,8,12],set_size=[5,15,30]):
    # function to check RT as function of ecc or set size for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # density_arr - surface density array for all trials
    # type_trial - get RT values in function of 'ecc' or 'set'
    # density - select high or low density trials
    # threshold - threshold density value
    #
    # OUTPUTS #
    # RT_all - mean RT for all groups
        
        
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    # list RT
    RT = data['RT'].values
    
    if type_trial=='ecc':
        # list of values with target ecc
        target_type = data['target_ecc'].values
        val = ecc # values to loop over
    elif type_trial=='set':
        # list of values with target set size
        target_type = data['set_size'].values
        val = set_size # values to loop over
    
    RT_all = [] # RT for all ecc
    
    for _,j in enumerate(val):
        
        RT_per_group = []
        
        for i in range(len(data)): # for all actual trials 

            if key_or[i]==target_or[i] and int(target_type[i])==j: # if key press = target orientation and correct ecc/set size
                
                # save depending on density of trial
                if density=='high' and density_arr[i]>=threshold:
                    RT_per_group.append(RT[i]) #append RT value
                elif density=='low' and density_arr[i]<threshold:
                    RT_per_group.append(RT[i]) #append RT value
        
        RT_all.append(np.mean(RT_per_group))
        #print('number of trials for group %d is %d'%(j,len(RT_per_group)))
    
    return RT_all
    
def density_meanfix(data,eyedata,density_arr,type_trial='ecc',density='high',
                    threshold=0.03, ecc=[4,8,12],set_size=[5,15,30],
                    hRes=1680,vRes=1050,screenHeight=30,screenDis=57,size_gab=2.2):
    
    # function to check mean number of fixations as function of ecc or set size for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # eyedata - eyetracking data
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # fix_all - mean fix for all ecc
    

    
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    # list of values of RT
    RT = data['RT'].values
    # radius of gabor in pixels
    r_gabor = ang2pix(size_gab/2,screenHeight,
                       screenDis,
                       vRes)
    # number of samples in 150ms (we'll not count with fixations prior to 150ms after stim display)
    sample_thresh = 1000*0.150 # 1000Hz * time in seconds
    
    if type_trial=='ecc':
        # list of values with target ecc
        target_type = data['target_ecc'].values
        val = ecc # values to loop over
    elif type_trial=='set':
        # list of values with target set size
        target_type = data['set_size'].values
        val = set_size # values to loop over
    
    fix_all = [] # RT for all ecc
    
    for _,j in enumerate(val):
        
        fix_per_group = []
        
        for i in range(len(data)): # for all actual trials 
            
            if key_or[i]==target_or[i] and int(target_type[i])==j: # if key press = target orientation and correct ecc/set size
                
                # save depending on density of trial
                if (density=='high' and density_arr[i]>=threshold) or (density=='low' and density_arr[i]<threshold):
                
                    # index for moment when display was shown
                    idx_display = np.where(np.array(eyedata[i]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                    # eye tracker sample time of display
                    smp_display = eyedata[i]['events']['msg'][idx_display][0]

                    # get target positions as strings in list
                    target_pos = data['target_position'][i].replace(']','').replace('[','').split(' ')
                    # convert to list of floats
                    target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])

                    num_fix = 0
                    for k,fix in enumerate(eyedata[i]['events']['Efix']):

                        # if fixations between 150ms after display and key press time
                        if (fix[0] > (smp_display+sample_thresh) and fix[0] < np.round(smp_display + RT[i]*1000)):

                            #if fixation not on target (not within target radius)
                            fix_x = fix[-2] - hRes/2
                            fix_y = fix[-1] - vRes/2; fix_y = - fix_y

                            if np.sqrt((fix_x-target_pos[0])**2+(fix_y-target_pos[1])**2) > r_gabor:
                                num_fix += 1

                    fix_per_group.append(num_fix) #append number of fixations for that trial value

        fix_all.append(np.mean(fix_per_group)) # append mean number of fixations per ecc
        #print('number of trials for group %d is %d'%(j,len(fix_per_group)))
    
    return fix_all   



def density_on_objectfix(data,eyedata,density_arr,type_trial='ecc',density='high',
                    threshold=0.03, ecc=[4,8,12],set_size=[5,15,30],
                    hRes=1680,vRes=1050,screenHeight=30,screenDis=57,radius=1.1):
    
    # function to check percentage of on object fixations as function of ecc or set size for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # eyedata - eyetracking data
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # fix_all - mean fix for all ecc
    
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    # list of values of RT
    RT = data['RT'].values
    # radius of gabor in pixels
    radius_pix = ang2pix(radius,screenHeight,
                       screenDis,
                       vRes)
        
    # number of samples in 150ms (we'll not count with fixations prior to 150ms after stim display)
    sample_thresh = 1000*0.150 # 1000Hz * time in seconds
    
    if type_trial=='ecc':
        # list of values with target ecc
        target_type = data['target_ecc'].values
        val = ecc # values to loop over
    elif type_trial=='set':
        # list of values with target set size
        target_type = data['set_size'].values
        val = set_size # values to loop over
    
    fix_all = [] # fix for all ecc
    
    for _,j in enumerate(val):
        
        fix_per_group = []
        
        for i in range(len(data)): # for all actual trials 
            
            if key_or[i]==target_or[i] and int(target_type[i])==j: # if key press = target orientation and correct ecc/set size
                
                # save depending on density of trial
                if (density=='high' and density_arr[i]>=threshold) or (density=='low' and density_arr[i]<threshold):
                
                    # index for moment when display was shown
                    idx_display = np.where(np.array(eyedata[i]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                    # eye tracker sample time of display
                    smp_display = eyedata[i]['events']['msg'][idx_display][0]

                    num_fix = 0
                    for k,fix in enumerate(eyedata[i]['events']['Efix']):

                        # if fixations between 150ms after display and key press time
                        if (fix[0] > (smp_display+sample_thresh) and fix[0] < np.round(smp_display + RT[i]*1000)):

                            #if fixation not on target (not within target radius)
                            fix_x = fix[-2] - hRes/2
                            fix_y = fix[-1] - vRes/2; fix_y = - fix_y

                            # get distractor positions as strings in list
                            distr_pos = data['distractor_position'][i].replace(']','').replace('[','').replace(',','').split(' ')
                            # convert to list of floats
                            distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])
                            # save distractor positions in pairs (x,y)
                            alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])

                            # if fixation within radius of any of the distractors
                            for n in range(len(alldistr_pos)): 
                                if np.sqrt((fix_x-alldistr_pos[n][0])**2+(fix_y-alldistr_pos[n][1])**2) < radius_pix:
                                    num_fix += 1 # save fixation

                    if len(eyedata[i]['events']['Efix'])==0:   # if empty, to avoid division by 0
                        on_obj_per = 0
                    else:
                        on_obj_per = num_fix/len(eyedata[i]['events']['Efix'])

                    fix_per_group.append(on_obj_per) #append percentage of on object fixations of trial

        fix_all.append(np.mean(fix_per_group)) # append mean percentage of on object fixations per ecc
        #print('number of trials for group %d is %d'%(j,len(fix_per_group)))
    
    return fix_all


def density_plot_correlation(arr_x_LOW,arr_x_HIGH,arr_y,label_x,label_y,plt_title,outfile,p_value=0.05):
    
    corr1, pval1 = spearmanr(arr_x_LOW,arr_y)

    print('correlation = %.6f, p-value = %.6f'%(corr1,pval1))
    if pval1<p_value:
        print('SIGNIFICANT CORRELATION for LOW density')
        
    corr2, pval2 = spearmanr(arr_x_HIGH,arr_y)

    print('correlation = %.6f, p-value = %.6f'%(corr2,pval2))
    if pval2<p_value:
        print('SIGNIFICANT CORRELATION for HIGH density')

    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    df4plot = pd.DataFrame({'xx_LOW': arr_x_LOW,
                            'xx_HIGH': arr_x_HIGH,
                            'yy': arr_y})


    sns.regplot(x=df4plot['xx_LOW'],y=df4plot['yy'],color='blue', marker='.',label='LOW')
    sns.regplot(x=df4plot['xx_HIGH'],y=df4plot['yy'],color='red', marker='+',label='HIGH')    

    ax = plt.gca()
    ax.set(ylabel = label_y, xlabel = label_x)
    ax.set_title(plt_title+' LOW - (rho=%.2f,pval=%.3f) HIGH - (rho=%.2f,pval=%.3f)'%(corr1,pval1,corr2,pval2))
    ax.legend()
    plt.savefig(outfile, dpi=100,bbox_inches = 'tight')


    
