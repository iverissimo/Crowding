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
#import random 
import math
import time
#import itertools
#import pickle
import pandas as pd
import json
import sys
import os



############# functions #################

# draw fixation cross function
def draw_fixation(posit,lineSize,linecolor,linewidth): 
    
    t = lineSize/2.0
    vertical_line = visual.Line(win,start = (posit[0],posit[1]-t),end = (posit[0],posit[1]+t),lineColor = linecolor,lineWidth=linewidth)
    horizontal_line = visual.Line(win,start = (posit[0]-t,posit[1]),end = (posit[0]+t,posit[1]),lineColor = linecolor,lineWidth=linewidth)
    
    vertical_line.draw()
    horizontal_line.draw() 

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
        
    else: #if correct response
        if counter == D: #if counted necessary number of responses (i.e. too easy)
            if curr_dist > min_val: #and if distance not minimal
                curr_dist = curr_dist - step #reduce distance between flanker and target
            counter = 1
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


########## Initial parameters #########
        
# general info
num_blk = 2 #total number of blocks
num_rep = 20 #number of repetions of unique display per block

num_trl, trgt_ecc,trgt_vf = uniq_trials(params['ecc']) 
num_trl = num_trl*num_rep#total number of trials

l_trl = r_trl = num_trl/2 #number of trials for left and right target locations

# screen info   
# params['screenHeight'] - height of the screen in cm 
# params['screenDis'] - distance between screen and chinrest in cm 
# num_fl - number of distractors
# initpos_fl - initial pos (degree)
# ort_trgt - orientation of target (degrees)
# max_dist - ratio of ecc that is max possible flk-trgt distance 
# min_dist - ratio of ecc that is min possible flk-trgt distance
# step_stair - step size for the staircase 
# Down_factor - number of correct responses needed to increase difficulty
 

# fixation cross info
fixpos = (0,0) #center position
fixlineSize = ang2pix(params['fixlineSize_deg'],params['screenHeight'],params['screenDis'],params['vRes'])
linewidth = ang2pix(params['linewidth_deg'],params['screenHeight'],params['screenDis'],params['vRes']) 

# gabor info
siz_gab = ang2pix(params['siz_gab_deg'],params['screenHeight'],params['screenDis'],params['vRes']) #size in deg of visual angle
gab_sf = params['gab_sf_deg']/ang2pix(1,params['screenHeight'],params['screenDis'],params['vRes']) #sf cycles per pixel = sf_CyclesDeg / pixelsPerDegree
sd_gab = ang2pix(params['sd_gab_deg'],params['screenHeight'],params['screenDis'],params['vRes']) #standard deviation of gaussian
dist_fl = 360/params['num_fl'] #distance between stim (degrees)

pos_fl = np.arange(params['initpos_fl'],params['initpos_fl']+360,dist_fl) #position of distractors (degrees), equally spaced
ort_fl = np.repeat(0,params['num_fl']) # all flankers have same orientation (0 equals vertical, goes clockwise until 360deg)


trls_idx = np.repeat(range(0,num_trl/num_rep),num_rep) #range of indexes for all trials 
ort_lbl = np.append(np.repeat(['right'],num_trl/2),np.repeat(['left'],num_trl/2)) #taget orientation labels


# array to save variables
RT_trl = np.array(np.zeros((num_blk,num_trl)),object) #array for all RTs
key_trl = np.array(np.zeros((num_blk,num_trl)),object) #array for all key presses
display_idx = np.array(np.zeros((num_blk,num_trl)),object) #array for idx of all displays
trgt_ort_lbl = np.array(np.zeros((num_blk,num_trl)),object) #array for target orientations
distances = np.array(np.zeros((num_blk,num_trl)),object) #array for all distance values


# create a window
#win = visual.Window(size=(hRes, vRes), color = backCol, units='pix',fullscr  = True, screen = 1,allowStencil=True)
win = visual.Window(size= (params['hRes'], params['vRes']), color = params['backCol'], units='pix',fullscr  = True, screen = 0,allowStencil=True)   
   
#pause
core.wait(2.0)

text = 'Indicate the orientation of the middle gabor by pressing the left or right arrow keys.\nPlease keep your eyes fixated on the center.'
BlockText = visual.TextStim(win, text=text, alignVert='center',alignHoriz='center',color='white', pos = (0,140),height=30)
text2 = 'Press spacebar to start'
PressText = visual.TextStim(win, text=text2, color='white', height=30, pos = (0,-140))
    
BlockText.draw()
draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation 
PressText.draw()
win.flip()
event.waitKeys(keyList = 'space') 

counter1 = 1 #staircase counter per eccentricity
counter2 = 1
counter3 = 1
trgt_fl_dist1 = params['max_dist'] #we start with max distance to make it easy
trgt_fl_dist2 = params['max_dist']
trgt_fl_dist3 = params['max_dist']
    
for j in range(num_blk):
    
    np.random.shuffle(ort_lbl) #randomize target orientation
    np.random.shuffle(trls_idx) #randomize index for display

    #np.random.shuffle(trgt_ecc)
    
    text = 'Block %i' %(j+1)
    BlockText = visual.TextStim(win, text=text, color='white', height=50, pos = (0,140))
    #trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',ori=ort_blk,sf=gab_sf,size=siz_gab,pos=(0,0))
    text2 = 'Press spacebar to start'
    PressText = visual.TextStim(win, text=text2, color='white', height=30, pos = (0,-140))
    
    BlockText.draw()
    draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation 
    #trgt.draw()
    PressText.draw()
    win.flip()
    event.waitKeys(keyList = 'space') 

    draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation 
    win.flip() # flip the screen
    core.wait(2.0) #pause
    
    for k in range(num_trl):
        
        ort_trl = params['ort_trgt'][0] if ort_lbl[k]=='right' else params['ort_trgt'][1] #define target orientation for trial

        if trgt_vf[trls_idx[k]] == 'left':
            xpos_trgt = -ang2pix(float(trgt_ecc[trls_idx[k]]),params['screenHeight'],params['screenDis'],params['vRes'])
            trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(xpos_trgt,0),units=None)           
        else:
            xpos_trgt = ang2pix(float(trgt_ecc[trls_idx[k]]),params['screenHeight'],params['screenDis'],params['vRes'])
            trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(xpos_trgt,0),units=None)            
             
        trgt.draw()
        
        for i in range(len(pos_fl)):
            xpos_fl,ypos_fl = pol2cart(ang2pix(float(trgt_ecc[trls_idx[k]])*float(trgt_fl_dist),params['screenHeight'],params['screenDis'],params['vRes']), pos_fl[i])
            flank = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_fl[i],sf=gab_sf,size=siz_gab,pos=(xpos_fl+xpos_trgt,ypos_fl),units=None)
            flank.draw()

    
        draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation
        win.flip() # flip the screen
        
        t0 = core.getTime() #get the time (seconds)
        key = [] # reset key to nothing 
           
        while core.getTime() - t0 < params['stim_time']: # while current time < stimulus presentation time (seconds)
            
            key = event.getKeys(keyList = ['left','right','s'])
            
            if len(key)>0:
                if key[0] == 's': #stop key
                    win.close()
                    core.quit()
                    break 
                
                RT_trl[j][k] = core.getTime() - t0 
                key_trl[j][k] = key[0] 
                #time.sleep(params['stim_time']-(core.getTime() - t0)) 
                break
         
        if key_trl[j][k] == ort_lbl[k]:
            response = 1
        else:
            response = 0
        
        if trgt_ecc == ecc[0]:
            trgt_fl_dist1,counter1 = staircase_1upDdown(params['Down_factor'],response,params['step_stair'],params['max_dist'],params['min_dist'],trgt_fl_dist=trgt_fl_dist1,counter=counter1)
            print 'response is %d and distance is %.2f' % (response, trgt_fl_dist1)
            distances[j][k] = trgt_fl_dist1

        elif trgt_ecc == ecc[1]:
            trgt_fl_dist2,counter2 = staircase_1upDdown(params['Down_factor'],response,params['step_stair'],params['max_dist'],params['min_dist'],trgt_fl_dist=trgt_fl_dist2,counter=counter2)
            print 'response is %d and distance is %.2f' % (response, trgt_fl_dist2)
            distances[j][k] = trgt_fl_dist2
        
        else: #trgt_ecc == ecc[2]
            trgt_fl_dist3,counter3 = staircase_1upDdown(params['Down_factor'],response,params['step_stair'],params['max_dist'],params['min_dist'],trgt_fl_dist=trgt_fl_dist3,counter=counter3)
            print 'response is %d and distance is %.2f' % (response, trgt_fl_dist3)
            distances[j][k] = trgt_fl_dist3
    
        draw_fixation(fixpos,fixlineSize,params['fixcolor'],linewidth) #draw fixation 
        win.flip() # flip the screen
        core.wait(params['iti']) #pause

    display_idx[j][:] = trls_idx
    trgt_ort_lbl[j][:] = ort_lbl



# save relevant variables in panda dataframe
for d in range(num_blk):
    target_ecc = np.zeros((1,num_trl))
    for l in range(num_trl):
        target_ecc[0][l] = trgt_ecc[display_idx[0][l]]
    
    dict_var = {'target_orientation':trgt_ort_lbl[d][:], 'target_ecc':target_ecc[0][:], 'target_flank_ratio':distances[d][:],'key_pressed':key_trl[d][:],'RT':RT_trl[d][:]}
    if d==0:
        df = pd.DataFrame(data=dict_var)
    else:
        df=pd.concat([df, pd.DataFrame(data=dict_var)])
        
df.to_csv('data_crowding_pp_'+pp+'.csv', sep='\t')

    
#cleanup
win.close() #close display
core.quit()





