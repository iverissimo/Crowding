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
import pickle
import pandas as pd


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
    
#######################################

pp = '0' #raw_input("Participant number: ")

########## Initial parameters #########
# general info
num_blk = 2 #total number of blocks
num_rep = 5#20 #number of repetions of unique display per block
ecc_deg = [4,8,12]
dist_bin = np.arange(0.2,0.9,0.1) # target-flanker spacing ratio

num_trl, _, _ = uniq_trials(ecc_deg) 
num_trl = num_trl*num_rep#total number of trials

l_trl = r_trl = num_trl/2 #number of trials for left and right target locations

stim_time = 1.5 #stimulus presentation time (seconds)
iti = 0.5 #inter-trial-interval (seconds)

# screen info   
hRes = 1280#2560#1920 #900 
vRes = 800#1600#1080 #700 
screenHeight = 21.9#30 #height of the screen in cm 
screenDis = 71    #distance between screen and chinrest in cm     
backCol = 'black'

# fixation cross info
fixpos = (0,0) #center position
fixlineSize = ang2pix(0.25,screenHeight,screenDis,vRes)
fixcolor = 'white' 
linewidth = ang2pix(0.05,screenHeight,screenDis,vRes) 

# gabor info
siz_gab = ang2pix(1.5,screenHeight,screenDis,vRes) #size in deg of visual angle
gab_sf = 6/ang2pix(1,screenHeight,screenDis,vRes) #sf cycles per pixel = sf_CyclesDeg / pixelsPerDegree
sd_gab = ang2pix(0.06,screenHeight,screenDis,vRes) #standard deviation of gaussian
num_fl = 2 # number of distractors
dist_fl = 360/num_fl #distance between stim (degrees)
initpos_fl = 90 #initial pos (degree)

pos_fl = np.arange(initpos_fl,initpos_fl+360,dist_fl) #position of distractors (degrees), equally spaced
ort_fl = np.repeat(0,num_fl) # all flankers have same orientation (0 equals vertical, goes clockwise until 360deg)
ort_trgt = [5,355] #orientation of target (degrees)

trgt_fl_dist = [0.2]
#labels
_, trgt_ecc,trgt_vf = uniq_trials(ecc_deg) 


trls_idx = np.repeat(range(0,num_trl/num_rep),num_rep) #range of indexes for all trials 
ort_lbl = np.append(np.repeat(['right'],num_trl/2),np.repeat(['left'],num_trl/2)) #taget orientation labels


# array to save variables
RT_trl = np.array(np.zeros((num_blk,num_trl)),object) #array for all RTs
key_trl = np.array(np.zeros((num_blk,num_trl)),object) #array for all key presses
display_idx = np.array(np.zeros((num_blk,num_trl)),object) #array for idx of all displays
trgt_ort_lbl = np.array(np.zeros((num_blk,num_trl)),object) #array for target orientations

# create a window
#win = visual.Window(size=(hRes, vRes), color = backCol, units='pix',fullscr  = True, screen = 1,allowStencil=True)
win = visual.Window(size= (hRes, vRes), color = backCol, units='pix',fullscr  = True, screen = 0,allowStencil=True)   
   
#pause
core.wait(2.0)

text = 'Indicate the orientation of the middle gabor by pressing the left or right arrow keys.\nPlease keep your eyes fixated on the center.'
BlockText = visual.TextStim(win, text=text, alignVert='center',alignHoriz='center',color='white', pos = (0,140),height=30)
text2 = 'Press spacebar to start'
PressText = visual.TextStim(win, text=text2, color='white', height=30, pos = (0,-140))
    
BlockText.draw()
draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation 
PressText.draw()
win.flip()
event.waitKeys(keyList = 'space') 

core.wait(2.0)


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
    draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation 
    #trgt.draw()
    PressText.draw()
    win.flip()
    event.waitKeys(keyList = 'space') 

    draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation 
    win.flip() # flip the screen
    core.wait(2.0) #pause
    
    for k in range(num_trl):
        
        ort_trl = ort_trgt[0] if ort_lbl[k]=='right' else ort_trgt[1] #define target orientation for trial

        if trgt_vf[trls_idx[k]] == 'left':
            xpos_trgt = -ang2pix(float(trgt_ecc[trls_idx[k]]),screenHeight,screenDis,vRes)
            trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(xpos_trgt,0))           
        else:
            xpos_trgt = ang2pix(float(trgt_ecc[trls_idx[k]]),screenHeight,screenDis,vRes)
            trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(xpos_trgt,0))            
             
        trgt.draw()
        
        for i in range(len(pos_fl)):
            xpos_fl,ypos_fl = pol2cart(ang2pix(float(trgt_ecc[trls_idx[k]])*float(trgt_fl_dist[0]),screenHeight,screenDis,vRes), pos_fl[i])
            flank = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_fl[i],sf=gab_sf,size=siz_gab,pos=(xpos_fl+xpos_trgt,ypos_fl))
            flank.draw()

    
        draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation
        win.flip() # flip the screen
        
        t0 = core.getTime() #get the time (seconds)
        key = [] # reset key to nothing 
           
        while core.getTime() - t0 < stim_time:
            
            key = event.getKeys(keyList = ['left','right'])
            
            if len(key)>0:
                RT_trl[j][k] = core.getTime() - t0 
                key_trl[j][k] = key[0] 
                time.sleep(stim_time-(core.getTime() - t0)) 
                break
        
        if key_trl[j][k] == ort_lbl[k]:
            print 'correct' 
        else:
            print 'wrong'
        
    
        draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation 
        win.flip() # flip the screen
        core.wait(iti) #pause

    display_idx[j][:] = trls_idx
    trgt_ort_lbl[j][:] = ort_lbl




# save relevant variables in panda dataframe
for d in range(num_blk):
    target_ecc = np.zeros((1,num_trl))
    target_flank_dist= np.zeros((1,num_trl))
    for l in range(num_trl):
        target_ecc[0][l] = trgt_ecc[display_idx[0][l]];
        #target_flank_dist[0][l] = trgt_fl_dist[display_idx[0][l]];
    
    dict_var = {'target_orientation':trgt_ort_lbl[d][:], 'target_ecc':target_ecc[0][:], 'target_flank_ratio':target_flank_dist[0][:],'key_pressed':key_trl[d][:],'RT':RT_trl[d][:]}
    if d==0:
        df = pd.DataFrame(data=dict_var)
    else:
        df=pd.concat([df, pd.DataFrame(data=dict_var)])
        
df.to_csv('data_crowding_pp_'+pp+'.csv', sep='\t')

    
#cleanup
win.close() #close display
core.quit()





