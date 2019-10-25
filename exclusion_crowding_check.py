
# SCRIPT TO SEE IF SUBJECT SHOULD BE EXCLUDED FROM CROWDING
# ONLY BASED ON EYETRACKING DATA

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

# paths
base_dir =os.getcwd(); base_dir = os.path.join(base_dir,'Data2convert','Data7deg')
output_vs = os.path.join(base_dir,'output_VS')
output_crwd = os.path.join(base_dir,'output_crowding')


# define participant number and open json parameter file
if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')


else:
    # fill subject number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)


with open('/Users/verissimo/Documents/Projects/PSR_2019/Crowding/lab_parameters.json','r') as json_file:	
            analysis_params = json.load(json_file)	


# define dir to save plots
plot_dir = os.path.join(output_crwd,'plots')
if not os.path.exists(plot_dir):
     os.makedirs(plot_dir)

display_dir = os.path.join(plot_dir,'gaze_display',sj)
if not os.path.exists(display_dir):
     os.makedirs(display_dir)


vRes = analysis_params['vRes']
hRes = analysis_params['hRes']


# get eyetracking data
filename,edfdata = convert2asc(sj,'crowding',output_vs,output_crwd)

# first get duration of each trial from start of trial to end 
#(well check that by seeing the number of gaze points in a trial - ex:300 samples = 300 ms duration)
first_trial_indx = 72 # first 72 trials were training block, doesn't count
all_trial_dur = []
for trl in range(first_trial_indx,len(edfdata)): 
    all_trial_dur.append(len(edfdata[trl]['trackertime']))


# plot duration of all crowding trials 
fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
plt.plot(all_trial_dur)
plt.xlabel('Trial #',fontsize=18)
plt.ylabel('Time (ms)',fontsize=18)
plt.axhline(y=np.median(all_trial_dur), color='r', linestyle='--')
plt.title('Trial duration (median trial duration is %.2f ms)'%np.median(all_trial_dur))

fig.savefig(os.path.join(plot_dir,'all_trials_duration_sub-{sj}.png'.format(sj=str(sj).zfill(2))), dpi=100)


# I think trials with durations of less than 180 ms do not make sense - confirm with Chris
# or bigger than stim time (2 seconds)
# so let's get the index for the "good trials"
good_trl_indx = np.where((np.array(all_trial_dur)>180) & (np.array(all_trial_dur)<analysis_params['stim_time']*1000))[0]
good_trl_indx += first_trial_indx # then it corresponds to edf data index

exclusion_radius_deg = 2
exclusion_radius_pix = ang2pix(exclusion_radius_deg,analysis_params['screenHeight'],analysis_params['screenDis'],analysis_params['vRes'])



# Now do loop where in the end the trial indexes are saved for trials that pass the exclusion criteria
# (no median gaze position, within display window, outside fixation circle)

final_trl_indx = []
excluded_fix_radius = [] # just to plot hist of ecc of fixations made outside exclusion criteria 

for _,index in enumerate(good_trl_indx):
    
    # save gaze coordinates for a specific trial
    gaze = np.array([edfdata[index]['x'],edfdata[index]['y']])
    
    # find index corresponding to show display time (moment where display is on screen)
    display_gaze = gaze.copy()
    # we consider display show time to be tracker time stamp for VF message - because it's the closest to win.flip()
    # but in some cases the msg event also includes block number
    # because of this time stamp is not always in same position of array
    # messy coding leads to messys fixes... lesson for the future
    if edfdata[index]['events']['msg'][3][-1][4:6] == 'VF': 
        timemarker = edfdata[index]['events']['msg'][3][0]
        # save VF location for plots
        VF = 'right' if edfdata[index]['events']['msg'][3][-1][7:11] in 'right'else 'left'
        
    elif edfdata[index]['events']['msg'][3][-1][4:6] == 'RT': #trials that doen't correspond to beginning of block 
        timemarker = edfdata[index]['events']['msg'][2][0]
        # save VF location for plots
        VF = 'right' if edfdata[index]['events']['msg'][2][-1][7:11] in 'right'else 'left'
    
    # and ecc
    ECC = [x for x in analysis_params['ecc'] if str(x) in edfdata[index]['events']['msg'][-2][-1][-3:-1]]

    # now select the relevant gaze points   
    display_start_ind = np.where(edfdata[index]['trackertime']==timemarker)
    display_start_ind = display_start_ind[0][0] # make it integer

    # display end time will be the initial tracker time + 350 ms
    display_end_ind = np.where(edfdata[index]['trackertime']==timemarker+int(analysis_params['display_time']*1000))

    # save gaze x and y positions that where within display time
    if len(display_end_ind[0])>0: # not empty -> # trial duration was longer than display time
        display_end_ind = display_end_ind[0][0] # make it integer
        display_gaze = display_gaze[:,display_start_ind:display_end_ind]

    else: # empty -> trial ended before display disappeared from screen
        display_gaze = display_gaze[:,display_start_ind::] # until end of array


    # now filter trials, saving only the ones where display gaze within exclusion circle
    median_gaze = np.median(display_gaze,axis=-1) # median x,y position
    if ((median_gaze[0]-(hRes/2))**2 + (median_gaze[1] - (vRes/2))**2) <= exclusion_radius_pix**2:
        final_trl_indx.append(index) # SAVE TRIAL INDEX because they were fixating
    else:
        excluded_fix_radius.append(np.sqrt((median_gaze[0]-(hRes/2))**2 + (median_gaze[1] - (vRes/2))**2))
        #print('not fixating')
        # save plot of where gaze was for excluded trials
        # set figure and main axis
        fig, ax = plt.subplots()
        # change default range 
        ax.set_xlim((0, hRes))
        ax.set_ylim((0, vRes))

        # add target as red circle
        for ind in range(len(display_gaze)):

            circle = plt.Circle((display_gaze[0][ind], display_gaze[1][ind]), 10, color='b')
            ax.add_artist(circle)

        pos_deg = ECC[0] if VF=='right' else -ECC[0]

        target_ecc = ang2pix(pos_deg,analysis_params['screenHeight'],analysis_params['screenDis'],analysis_params['vRes'])
        circle_trgt = plt.Circle(((hRes/2)-target_ecc, vRes/2), 10, color='r')
        ax.add_artist(circle_trgt)

        plt.plot(hRes/2, vRes/2, marker='+', markersize=5, color="black")
        #plt.show()
        fig.savefig(os.path.join(display_dir,'display_exclgaze_trial-{trl}_sub-{sj}.png'.format(trl=str(index).zfill(3),sj=str(sj).zfill(2))), dpi=100)



# Calculate % of trials when they were not fixating properly
#
#total_trials = 576
#exclud_trial_per = len(excluded_fix_radius)/total_trials * 100
#print('%.2f %% of trials were excluded for subject %s'%(exclud_trial_per,sj))
#
## plot scatter of excluded fixations and include lines where 
## target eccentricities are and exclusion criteria too
#ecc_trgt_pix = [ang2pix(ecc,analysis_params['screenHeight'],analysis_params['screenDis'],analysis_params['vRes']) 
#                for _,ecc in enumerate(analysis_params['ecc'])]
#
#fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')
#
#plt.scatter(range(len(excluded_fix_radius)),excluded_fix_radius)
#plt.axhline(y=exclusion_radius_pix, color='black', linestyle='-')
#plt.axhline(y=ecc_trgt_pix[0], color='r', linestyle='--')
#plt.axhline(y=ecc_trgt_pix[1], color='m', linestyle='--')
#plt.axhline(y=ecc_trgt_pix[2], color='g', linestyle='--')
#
#plt.xlabel('excluded trials',fontsize=18)
#plt.ylabel('eccentricity (pixel)',fontsize=18)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.legend(['exclusion line','4deg','8deg','12deg'], fontsize=10)
##plt.show()
#
#fig.savefig(os.path.join(plot_dir,'excluded_fixations_sub-{sj}.png'.format(sj=str(sj).zfill(2))), dpi=100)




