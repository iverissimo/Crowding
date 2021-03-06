#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:54:18 2019

@author: inesverissimo

visual search display, as ellipse
"""


from psychopy import visual, core, event 
import numpy as np
import random as rnd
import math
import time
#import itertools
#import pickle
import pandas as pd
import json
import sys
import os

# append path to Psycholink folder, cloned from https://github.com/jonathanvanleeuwen/psychoLink.git
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'psychoLink','PsychoLink'))
import psychoLink as PL

from utils import *

## for labs will have to run with spyder/openSesame so just add input in console ########
        
print 'Specify setting json file: lab or mac'

filename = str(raw_input('File: '))

if filename == 'mac':
    with open('mac_parameters.json','r') as json_file:
        params = json.load(json_file)
elif filename =='lab':
    with open('lab_parameters.json','r') as json_file:
        params = json.load(json_file)
else:
    raise NameError('Machine not defined, no parametes will be given') 
      

print 'Please add subject number (ex:01)'

pp = str(raw_input('Subject number: ').zfill(2))

output_dir = os.getcwd()+'/output_VS/'

if not os.path.exists(output_dir): #check if path to save output exists
        os.makedirs(output_dir)       # if not create it
     
####################################################
    
# variables to save in settings json

ecc = np.array(range(params['min_ecc_vs'],params['max_ecc_vs'],params['step_ecc_vs'])) # all ecc presented on screen
ecc_pix = [ang2pix(j,params['screenHeight'],params['screenDis'],params['vRes']) for _,j in enumerate(ecc)] # in pixels
n_points = [(i+1)*4 for i, _ in enumerate(ecc)] # number of points per ecc

num_trl,tg_ecc,tg_set_size = vs_uniq_trials(params['ecc'],params['set_size'])
num_trl = num_trl*params['rep_vs'] #total number of trials per block

trls_idx = np.repeat(range(0,num_trl/params['rep_vs']),(params['rep_vs'])) #range of indexes for all trials 
ort_lbl = np.append(np.repeat(['right'],num_trl/2),np.repeat(['left'],num_trl/2)) #target orientation labels

# gabor info
siz_gab = ang2pix(params['siz_gab_deg'],params['screenHeight'],params['screenDis'],params['vRes']) #size in deg of visual angle
gab_sf = params['gab_sf_deg']/ang2pix(1,params['screenHeight'],params['screenDis'],params['vRes']) #sf cycles per pixel = sf_CyclesDeg / pixelsPerDegree
sd_gab = ang2pix(params['sd_gab_deg'],params['screenHeight'],params['screenDis'],params['vRes']) #standard deviation of gaussian

# fixation cross info
fixpos = (0,0) #center position
fixlineSize = ang2pix(params['fixlineSize_deg'],params['screenHeight'],params['screenDis'],params['vRes'])
linewidth = ang2pix(params['linewidth_deg'],params['screenHeight'],params['screenDis'],params['vRes']) 


# ellipse params
ax_major_deg = ecc[-1] #size of 1/2 of major axis - parallel to xx - in degrees
ax_major_pix = ang2pix(ax_major_deg,params['screenHeight'],params['screenDis'],params['vRes'])

ax_minor_deg = params['ecc'][-1]+2 #size of 1/2 minor axis - parallel to yy - in degrees
ax_minor_pix = ang2pix(ax_minor_deg,params['screenHeight'],params['screenDis'],params['vRes'])

# define initial circle grid for positions
circles = circle_points(ecc_pix,n_points)
# constrain them within ellipse
pos_list = []
for j in range(len(circles)):   
    pos_list.append(ellipse_inpoints(circles[j],ax_major_pix,ax_minor_pix))

# number of possible positions
num_pos = count_set_size(pos_list)

# array to save variables
RT_trl = np.array(np.zeros((params['blk_vs'],num_trl)),object); RT_trl[:]=np.nan #array for all RTs
key_trl = np.array(np.zeros((params['blk_vs'],num_trl)),object); key_trl[:]=np.nan #array for all key presses
trgt_ort_lbl = np.array(np.zeros((params['blk_vs'],num_trl)),object) #array for target orientations

trgt_pos_all = np.array(np.zeros((params['blk_vs'],num_trl)),object) #array for idx of all displays
distr_pos_all = np.array(np.zeros((params['blk_vs'],num_trl)),object) #array for all distance values
trgt_ecc_all = np.array(np.zeros((params['blk_vs'],num_trl)),object) #array for ecc of all targets

set_size_all = np.array(np.zeros((params['blk_vs'],num_trl)),object) #array for all set sizes

# create a window
win = visual.Window(size= (params['hRes'], params['vRes']),colorSpace='rgb255', color = params['backCol'], units='pix',fullscr  = True, screen = 0,allowStencil=True)   

## start tracker, define filename (saved in cwd)
#tracker = PL.eyeLink(win, fileName = 'eyedata_visualsearch_pp_'+pp+'.EDF', fileDest=output_dir)
#
## calibrate
#tracker.calibrate()

#pause
core.wait(2.0)

# Instructions
# Slide 1
text1 = 'In this task, you will see a screen with several gabors. Only one of the gabors will be tilted.\n\nYour task is to find this tilted gabor and indicate its orientation by pressing the left or right arrow keys.\n\nThis time, you can move your eyes around. Please return to the fixation cross at the end of each trial. '
BlockText = visual.TextStim(win, text=text1, colorSpace='rgb255', color = params['textCol'], pos = (0,120),height=30, wrapWidth = 700)
text2 = 'Press spacebar to continue'
PressText = visual.TextStim(win, text=text2, colorSpace='rgb255', color = params['textCol'], height=30, pos = (0,-160))
    
BlockText.draw()
PressText.draw()
win.flip()
event.waitKeys(keyList = 'space') 


    
for j in range(params['blk_vs']):
    
    np.random.shuffle(ort_lbl) #randomize target orientation
    np.random.shuffle(trls_idx) #randomize index for display
        
    text = 'Block %i' %(j+1)
    BlockText = visual.TextStim(win, text=text, color='white', height=50, pos = (0,140))
    text2 = 'Press spacebar to start'
    PressText = visual.TextStim(win, text=text2, color='white', height=30, pos = (0,-140))
    
    BlockText.draw()
    draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth,win) #draw fixation 
    PressText.draw()
    win.flip()
    event.waitKeys(keyList = 'space') 
    
    
    
    

    draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth,win) #draw fixation
    win.flip() # flip the screen
    core.wait(2.0) #pause
    
    for k in range(num_trl):

# =============================================================================
#         
#        # Start recording eye movements
#        tracker.startTrial()
#         
# =============================================================================
        
#        if k == 0: tracker.logVar('block_Nr', j) # save block start
#        tracker.logVar('trial_Nr', k) # save trial number
        
        if ort_lbl[k]=='right': #define target orientation for trial
            ort_trl = params['ort_trgt'][0]
            trgt_ort_lbl[j][k] = 'right' # save orientation in var
        else:
            ort_trl = params['ort_trgt'][1]
            trgt_ort_lbl[j][k] = 'left'
        
        tg_pos = rnd.choice(pos_list[np.where(ecc == tg_ecc[trls_idx[k]])[0][0]])  #randomly choose position within a certain ecc for target
        trgt_pos_all[j][k] = tg_pos #save target position
        trgt_ecc_all[j][k] = tg_ecc[trls_idx[k]] #save target ecc
        
#        tracker.logVar('target_pos', tg_pos) # save target position
#        tracker.logVar('target_ecc', tg_ecc[trls_idx[k]]) # save target ecc

        
        # all possible positions for distractors, except for the position already assigned to target
        all_pos = np.concatenate(pos_list) 
        all_pos = all_pos.tolist() # had to convert from np array to list to pop
        
        tg_idx = [idx for idx,lst in enumerate(all_pos) if lst==tg_pos.tolist()]        
        all_pos.pop(tg_idx[0]) 
        
    
        distr_pos = rnd.sample(all_pos,tg_set_size[trls_idx[k]]-1)  #randomly choose positions within a certain set size for distractors (-1 because one position of set already given to target)
        distr_pos_all[j][k] = distr_pos # save positions of distractors
        set_size_all[j][k] = tg_set_size[trls_idx[k]] #save set size
        
#        tracker.logVar('distractor_pos', distr_pos) 
#        tracker.logVar('distractor_pos', tg_set_size[trls_idx[k]]) 
        
        # draw display
        trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(tg_pos[0],tg_pos[1]),units=None)                        
        trgt.draw()
                
        for i in range(len(distr_pos)):
            distr = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=0,sf=gab_sf,size=siz_gab,pos=(distr_pos[i][0],distr_pos[i][1]),units=None)
            distr.draw()
                
        draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth,win) #draw fixation
        win.flip() # flip the screen
#        tracker.logVar('display', True)
        
        t0 = core.getTime() #get the time (seconds)
        key = [] # reset key to nothing 
           
        while(len(key)==0): # core.getTime() - t0 < params['stim_time']: # while current time < stimulus presentation time (seconds)
            
            key = event.getKeys(keyList = ['left','right','s'])
            
        if key[0] == 's': #stop key
#            tracker.stopTrial()
#            tracker.cleanUp()
            
            # save relevant variables in panda dataframe
            for d in range(params['blk_vs']): 
                dict_var = {'target_orientation':trgt_ort_lbl[d][:],'target_ecc':trgt_ecc_all[d][:],'key_pressed':key_trl[d][:],'RT':RT_trl[d][:],'target_position':trgt_pos_all[d][:],'distractor_position':distr_pos_all[d][:],'set_size':set_size_all[d][:]}
                if d==0:
                    df = pd.DataFrame(data=dict_var)
                else:
                    df=pd.concat([df, pd.DataFrame(data=dict_var)])
                    
            df.to_csv(output_dir+'data_visualsearch_pp_'+pp+'_block-'+str(j)+'_trial-'+str(k)+'.csv', sep='\t')
            
            win.close()
            core.quit()
        else:
            
            RT_trl[j][k] = core.getTime() - t0 # reaction time
            key_trl[j][k] = key[0] # pressed key
#            tracker.logVar('RT', RT_trl[j][k])
                   
                
        if key_trl[j][k] == ort_lbl[k]:
            response = 1
#            tracker.logVar('response', 'correct')
            print('correct')
        else:
            response = 0
#            tracker.logVar('response', 'incorrect')
            print('incorrect')

# =============================================================================
#        # stop tracking the trial
#        tracker.stopTrial()
#         
# =============================================================================

        fixation = draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth,win) #draw fixation
        win.flip()
       
        if 'tracker' in locals(): # if tracker object defined
                       
            waitForFixation = tracker.waitForFixation(fixation,maxDist=0, maxWait=4, nRings=3, fixTime=200)
            
        else:      # if no tracker
            #Pause for ITI
            core.wait(params['iti']) #pause
            

# save relevant variables in panda dataframe
for d in range(params['blk_vs']): 
    dict_var = {'target_orientation':trgt_ort_lbl[d][:],'target_ecc':trgt_ecc_all[d][:],'key_pressed':key_trl[d][:],'RT':RT_trl[d][:],'target_position':trgt_pos_all[d][:],'distractor_position':distr_pos_all[d][:],'set_size':set_size_all[d][:]}
    if d==0:
        df = pd.DataFrame(data=dict_var)
    else:
        df=pd.concat([df, pd.DataFrame(data=dict_var)])
        
df.to_csv(output_dir+'data_visualsearch_pp_'+pp+'.csv', sep='\t')
   

## cleanup
#tracker.cleanUp()
win.close() #close display
core.quit()











