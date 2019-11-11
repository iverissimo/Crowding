#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:32:51 2019

Author: Ines V.

Descrip:  crowding task to evaluate individual differences in critical spacing
between participants, at varying ecc

varying distance between target and flankers with staircase method

"""

from psychopy import visual, core, event 
import numpy as np
import random 
import math
import time
#import itertools
#import pickle
import pandas as pd
import json
import sys
import os

# append path to Psycholink folder, cloned from https://github.com/jonathanvanleeuwen/psychoLink.git
sys.path.append(os.getcwd()+'/psychoLink/PsychoLink')
import psychoLink as PL



############# functions #################

# draw fixation cross function
def draw_fixation(posit,lineSize,linecolor,linewidth): 
    
    t = lineSize/2.0
    fixation = visual.ShapeStim(win, 
        vertices=((posit[0], posit[1]-t), (posit[0], posit[1]+t), posit, (posit[0]-t,posit[1]), (posit[1]+t, posit[1])),
        lineWidth=linewidth,
        closeShape=False,
        lineColor=linecolor)
    
    fixation.draw()

# transform polar coordinates to cartesian
    
def pol2cart(hyp, theta):  
    x = hyp * np.cos(np.deg2rad(theta))
    y = hyp * np.sin(np.deg2rad(theta))
    return(x, y)

# Calculate the number of degrees that correspond to a single pixel
# and then calculate distance in pixels
# dist_in_deg - distance in deg of visual angle
#h - height of screen, d - distance from monitor, r - vertical resolution
def ang2pix(dist_in_deg,h,d,r): 
    deg_per_px = math.degrees(math.atan2(0.5*h,d))/(0.5*r)
    dist_in_px = dist_in_deg/deg_per_px
    return dist_in_px 

#define unique trials,with balanced visual field and #ecc
def uniq_trials(ecc):
    trials = 2*len(ecc)
    
    trgt_ecc = np.tile(ecc,2)
    trgt_vf = np.hstack((np.repeat(['right'],trials/2.0),np.repeat(['left'],trials/2.0)))
    
    return trials,trgt_ecc,trgt_vf

# staircase function
def staircase_1upDdown(D,response,step,max_val,min_val,curr_dist,counter):

    if response == 0: #if incorrect response
        if curr_dist < max_val: #and if the distance is not max value defined
            curr_dist = curr_dist + step # increase the flanker target separation by step
        counter = 1
        
    elif response == 1: #if correct response
        if counter == D: #if counted necessary number of responses (i.e. too easy)
            if curr_dist > min_val: #and if distance not minimal
                curr_dist = curr_dist - step #reduce distance between flanker and target
            counter = 0
        
        counter = counter + 1    
        
    return curr_dist,counter        
            
    
#######################################
    
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

output_dir = os.getcwd()+'/output_crowding/'

if not os.path.exists(output_dir): #check if path to save output exists
        os.makedirs(output_dir)       # if not create it
     

########## Initial parameters #########
        
# general info
num_uniq_trl, trgt_ecc,trgt_vf = uniq_trials(params['ecc'])

num_trl_fl = num_uniq_trl*params['rep_fl_crw'] #number of flanker present trials
num_trl_nofl = num_uniq_trl*params['rep_nofl_crw'] #number of no flanker trials
num_trl = num_trl_fl + num_trl_nofl #total number of trials

l_trl = r_trl = num_trl/2 #number of trials for left and right target locations

# fixation cross info
fixpos = (0,0) #center position
fixlineSize = ang2pix(params['fixlineSize_deg'],params['screenHeight'],params['screenDis'],params['vRes'])
linewidth = ang2pix(params['linewidth_deg'],params['screenHeight'],params['screenDis'],params['vRes']) 

# gabor info
siz_gab = ang2pix(params['siz_gab_deg'],params['screenHeight'],params['screenDis'],params['vRes']) #size in deg of visual angle
gab_sf = params['gab_sf_deg']/ang2pix(1,params['screenHeight'],params['screenDis'],params['vRes']) #sf cycles per pixel = sf_CyclesDeg / pixelsPerDegree
sd_gab = ang2pix(params['sd_gab_deg'],params['screenHeight'],params['screenDis'],params['vRes']) #standard deviation of gaussian
dist_fl = 360/params['num_fl'] #distance between stim (degrees)
max_jitter = ang2pix(params['max_jitter'],params['screenHeight'],params['screenDis'],params['vRes']) 

pos_fl = np.arange(params['initpos_fl'],params['initpos_fl']+360,dist_fl) #position of distractors (degrees), equally spaced
ort_fl = np.repeat(0,params['num_fl']) # all flankers have same orientation (0 equals vertical, goes clockwise until 360deg)


#trls_idx = np.repeat(range(0,num_trl/(params['rep_fl_crw']+params['rep_nofl_crw'])),(params['rep_fl_crw']+params['rep_nofl_crw'])) #range of indexes for all trials 
trls_idx_fl = np.repeat(range(0,num_trl_fl/params['rep_fl_crw']),(params['rep_fl_crw'])) #range of indexes for all trials 
trls_idx_nofl = np.repeat(range(0,num_trl_nofl/params['rep_nofl_crw']),(params['rep_nofl_crw'])) #range of indexes for all trials 


ort_lbl = np.append(np.repeat(['right'],num_trl/2),np.repeat(['left'],num_trl/2)) #target orientation labels
flank_lbl = np.append(np.repeat(['no flankers'],num_trl_nofl),np.repeat(['flankers'],num_trl_fl)) #flanker presence labels

# array to save variables
RT_trl = np.array(np.zeros((params['blk_crw'],num_trl)),object); RT_trl[:]=np.nan #array for all RTs
key_trl = np.array(np.zeros((params['blk_crw'],num_trl)),object); key_trl[:]=np.nan #array for all key presses
trgt_ort_lbl = np.array(np.zeros((params['blk_crw'],num_trl)),object) #array for target orientations
distances = np.array(np.zeros((params['blk_crw'],num_trl)),object) #array for all distance values
flank_trl = np.array(np.zeros((params['blk_crw'],num_trl)),object) #array for flanker presence

target_ecc_all = np.array(np.zeros((params['blk_crw'],num_trl)),object) #array to save all target ecc
target_vf_all = np.array(np.zeros((params['blk_crw'],num_trl)),object) #array to save all target vf

# create a window
#win = visual.Window(size=(hRes, vRes), color = backCol, units='pix',fullscr  = True, screen = 1,allowStencil=True)
win = visual.Window(size= (params['hRes'], params['vRes']),colorSpace='rgb255', color = params['backCol'], units='pix',fullscr  = True, screen = 0,allowStencil=True)   

## start tracker, define filename (saved in cwd)
tracker = PL.eyeLink(win, fileName = 'eyedata_crowding_pp_'+pp+'.EDF', fileDest=output_dir)
#
## calibrate
tracker.calibrate()
   
#pause
core.wait(2.0)

#Instructions
#Slide 1 - Intro
text1 = 'Welcome!\n\nIn this experiment, you will have to indicate the orientation of a shape that will appear in the periphery of your visual field.\n\nHowever, it is important that you keep your eyes fixated on the cross in the center of the screen during the whole task.\n\nOn the next slide, you will see an example of how the task display will look.'
Instruction1 = visual.TextStim(win, text=text1, colorSpace='rgb255', color = params['textCol'], pos = (0,120),height=30, wrapWidth = 700)
text2 = 'Press spacebar to continue with instructions'
PressText1 = visual.TextStim(win, text=text2, colorSpace='rgb255', color = params['textCol'], height=30, pos = (0,-160))
    
Instruction1.draw()
PressText1.draw()
win.flip()
event.waitKeys(keyList = 'space') 

#Slide2 - Example Screen
PressText1.draw()
xpos_trgt = ang2pix((8.0),params['screenHeight'],params['screenDis'],params['vRes'])
trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=params['ort_trgt'][0],sf=gab_sf,size=siz_gab,pos=(xpos_trgt,0),units=None) 
trgt.draw()
draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation 

xpos_fl,ypos_fl = pol2cart(ang2pix(8.0*float(params['max_dist']),params['screenHeight'],params['screenDis'],params['vRes']), pos_fl[0])
flank = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_fl[0],sf=gab_sf,size=siz_gab,pos=(xpos_fl+xpos_trgt,ypos_fl),units=None)
flank.draw()
xpos_fl,ypos_fl = pol2cart(ang2pix(8.0*float(params['max_dist']),params['screenHeight'],params['screenDis'],params['vRes']), pos_fl[1])
flank = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_fl[1],sf=gab_sf,size=siz_gab,pos=(xpos_fl+xpos_trgt,ypos_fl),units=None)
flank.draw()

win.flip() # flip the screen
event.waitKeys(keyList = 'space') 

#Slide3 - Response etc.
text3 = 'In the experiment, these stimuli will only be presented for a very short time. Please Indicate the orientation of the middle shape as best as you can, using the left or right arrow keys.\n\nSometimes, there will only be one shape presented, so do not let this confuse you.\n\nMake sure to just press once after each stimulus - only the first answer counts.\n\nThe experiment will start with a practice block.'
Instruction2 = visual.TextStim(win, text=text3, colorSpace='rgb255', color = params['textCol'], pos = (0,120),height=30, wrapWidth = 700)
text4 = 'Press spacebar to start'
PressText2 = visual.TextStim(win, text=text4, colorSpace='rgb255', color = params['textCol'], height=30, pos = (0,-160))
Instruction2.draw()
PressText2.draw()
win.flip()
event.waitKeys(keyList = 'space') 

counters = [1,1,1] #staircase counter per eccentricity
trgt_fl_dist = [params['max_dist'],params['max_dist'],params['max_dist']]#we start with max distance to make it easy
    
for j in range(params['blk_crw']):
    
    np.random.shuffle(ort_lbl) #randomize target orientation
    np.random.shuffle(trls_idx_fl) #randomize index for display
    np.random.shuffle(trls_idx_nofl) #randomize index for display
    np.random.shuffle(flank_lbl) #randomize flanker presence
    
    fl_counter = 0 # counters used to balance ecc of flankers and no flankers
    nofl_counter = 0
    
    #Text for training block
    if j == 0:
        text = 'Practice block' 
        BlockText = visual.TextStim(win, text=text, colorSpace='rgb255', color = params['textCol'], pos = (0,140),height=50)
        
        num_trl_blk = num_trl/2#define number of trials in this block (training block = half length)
        
    #Text for experimental blocks
    else:
        
        ## calibrate between blocks, gives participant time to have break
        tracker.calibrate()
        
        text = 'Block %i out of %i' %(j, params['blk_crw']-1)
        BlockText = visual.TextStim(win, text=text, colorSpace='rgb255', color = params['textCol'], pos = (0,140),height=50)
        
        num_trl_blk = num_trl#define number of trials in this block 
    
    if j == 1:
        counters = [1,1,1] #staircase counter per eccentricity
        trgt_fl_dist = [params['max_dist'],params['max_dist'],params['max_dist']]#we start with max distance to make it easy
    
    BlockText.draw()
    draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation 
    #trgt.draw()
    PressText2.draw()
    
    win.flip()
    event.waitKeys(keyList = 'space')     

    draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation 
    win.flip() # flip the screen
    core.wait(2.0) #pause
    
    for k in range(num_trl_blk):
        
# =============================================================================
#         
#        # Start recording eye movements
        tracker.startTrial()
#         
# =============================================================================
        
        if k == 0: 
		tracker.logVar('block_Nr', j) # save block start
        tracker.logVar('trial_Nr', k) # save trial number
        
        if flank_lbl[k]=='flankers': # doing this to guarantee balanced ecc for no flank and flank conditions
            trls_idx = trls_idx_fl[fl_counter]
            fl_counter += 1
        else : 
            trls_idx = trls_idx_fl[nofl_counter]
            nofl_counter += 1
  
        
        ort_trl = params['ort_trgt'][0] if ort_lbl[k]=='right' else params['ort_trgt'][1] #define target orientation for trial

        if trgt_vf[trls_idx] == 'left':
            xpos_trgt = -ang2pix(float(trgt_ecc[trls_idx]),params['screenHeight'],params['screenDis'],params['vRes'])
            trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(xpos_trgt,0),units=None)           
        else:
            xpos_trgt = ang2pix(float(trgt_ecc[trls_idx]),params['screenHeight'],params['screenDis'],params['vRes'])
            trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(xpos_trgt,0),units=None)            
             
        trgt.draw()
        
        ecc_index = params['ecc'].index(trgt_ecc[trls_idx])
        target_ecc_all[j][k] = trgt_ecc[trls_idx] # save all target ecc
        target_vf_all[j][k] = trgt_vf[trls_idx] # save all target vf 
        
        tracker.logVar('VF', trgt_vf[trls_idx]) # log VF 
        
        #Draw flankers, depending on eccentricity
        if flank_lbl[k] == 'flankers':
            for i in range(len(pos_fl)):
                xpos_fl,ypos_fl = pol2cart(ang2pix(float(trgt_ecc[trls_idx])*float(trgt_fl_dist[ecc_index]),params['screenHeight'],params['screenDis'],params['vRes']), pos_fl[i])
                flank = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_fl[i],sf=gab_sf,size=siz_gab,pos=(xpos_fl+xpos_trgt+random.uniform(max_jitter*(-1),max_jitter),ypos_fl),units=None)
                flank.draw()
                
                
        draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation
        win.flip() # flip the screen
        
        t0 = core.getTime() #get the time (seconds)
        key = [] # reset key to nothing 
           
        while core.getTime() - t0 < params['stim_time']: # while current time < stimulus presentation time (seconds)
            
            key = event.getKeys(keyList = ['left','right','s'])
            
            if len(key)>0:
                if key[0] == 's': #stop key
                    tracker.stopTrial()
                    tracker.cleanUp()
                    
                    # save relevant variables in panda dataframe
                    for d in range(params['blk_crw']):    
                        dict_var = {'target_orientation':trgt_ort_lbl[d][:], 'target_ecc':target_ecc_all[d][:], 'target_vf':target_vf_all[d][:], 'flanker_presence':flank_trl[d][:], 'target_flank_ratio':distances[d][:],'key_pressed':key_trl[d][:],'RT':RT_trl[d][:]}
                        if d==0:
                            df = pd.DataFrame(data=dict_var)
                        else:
                            df=pd.concat([df, pd.DataFrame(data=dict_var)])
                            
                    df.to_csv(output_dir+'data_crowding_pp_'+pp+'.csv', sep='\t')                                    
                    
                    win.close()
                    core.quit()
                    break 
                
                RT_trl[j][k] = core.getTime() - t0 
                key_trl[j][k] = key[0] 
                tracker.logVar('RT', RT_trl[j][k])
                break
            
            #if core.getTime() >= params['display_time']:#return to fixation display after 250ms
            if core.getTime() - t0 >= params['display_time']:
                print('time is',core.getTime()-t0)
                draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation
                win.flip()
        print'key is:',key_trl[j][k]  
        if key_trl[j][k] == ort_lbl[k]:
            response = 1
            tracker.logVar('response', 'correct')
        # needs to include condition for misses, doesn't work
        elif (key_trl[j][k] == 'left' and ort_lbl[k] == 'right') or (key_trl[j][k] == 'right' and ort_lbl[k] == 'left'):
            response = 0
        
        else:
            response = 2
            tracker.logVar('response', 'incorrect')
        
        if flank_lbl[k] == 'flankers':
            trgt_fl_dist[ecc_index],counters[ecc_index] = staircase_1upDdown(params['Down_factor'],response,params['step_stair'],params['max_dist'],params['min_dist'],curr_dist=trgt_fl_dist[ecc_index],counter=counters[ecc_index])
        
        print 'response is %d, distance is %.2f, ecc is %f, flank-condition is %s and index is %i' % (response, trgt_fl_dist[ecc_index],trgt_ecc[trls_idx],flank_lbl[k],trls_idx)

        tracker.logVar('distance', trgt_fl_dist[ecc_index])
        tracker.logVar('ecc', trgt_ecc[trls_idx])
        tracker.logVar('flank-condition', flank_lbl[k])

        distances[j][k] = trgt_fl_dist[ecc_index]
        
# =============================================================================
#        # stop tracking the trial
        tracker.stopTrial()
#         
# =============================================================================

        #Pause for ITI
        core.wait(params['iti']) #pause

    trgt_ort_lbl[j][:] = ort_lbl
    flank_trl[j][:] = flank_lbl



# save relevant variables in panda dataframe
for d in range(params['blk_crw']):    
    dict_var = {'target_orientation':trgt_ort_lbl[d][:], 'target_ecc':target_ecc_all[d][:], 'target_vf':target_vf_all[d][:], 'flanker_presence':flank_trl[d][:], 'target_flank_ratio':distances[d][:],'key_pressed':key_trl[d][:],'RT':RT_trl[d][:]}
    if d==0:
        df = pd.DataFrame(data=dict_var)
    else:
        df=pd.concat([df, pd.DataFrame(data=dict_var)])
        
df.to_csv(output_dir+'data_crowding_pp_'+pp+'.csv', sep='\t')

    
#cleanup
tracker.cleanUp()
win.close() #close display
core.quit()





