#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:43:44 2019
Author: Ines V.

Descrip:  crowding task to evaluate individual differences in critical spacing
between participants, at varying ecc

varying distance between target and flankers factorial 

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


#######################################

pp = '0' #raw_input("Participant number: ")

########## Initial parameters #########
# general info
num_blk = 2 #total number of blocks
num_trl = 72 #total number of trials

l_trl = r_trl = num_trl/2 #number of trials for left and right target locations
stim_time = 2.5 #stimulus presentation time (seconds)
iti = 3.4 #inter-trial-interval (seconds)

# screen info   
hRes = 1920#2560#1920 #900 
vRes = 1080#1600#1080 #700 
screenHeight = 30 #height of the screen in cm 
screenDis = 71    #distance between screen and chinrest in cm     
backCol = 'black'

# fixation cross info
fixpos = (0,0) #center position
fixlineSize = ang2pix(0.25,screenHeight,screenDis,vRes)
fixcolor = 'white' 
linewidth = ang2pix(0.05,screenHeight,screenDis,vRes) 

# gabor info
siz_gab = ang2pix(1.5,screenHeight,screenDis,vRes) #size in deg of visual angle
gab_sf = 0.16 #degrees
num_fl = 6 # number of distractors
dist_fl = 360/num_fl #distance between stim (degrees)
init_dgr = 0 #initial pos (degree)
hyp = 100 

ort_fl = np.arange(init_dgr,init_dgr+360,dist_fl) #orientation of distractors (degrees)
ort_trgt = [60,120] #orientation of target (degrees)

# ecc
ecc_deg = [3,6,9,12]
#ecc_pix = [ang2pix(ecc_deg[i],screenHeight,screenDis,vRes) for i in range(len(ecc_deg))]

# target-flanker spacing
dist_bin = np.arange(0.1,0.9,0.1)

#np.vstack((trls,trgt_type))

#labels
trgt_fl_dist = np.tile(dist_bin,int(num_trl*(1/float(len(dist_bin)))))
trgt_ecc = np.tile(np.repeat(ecc_deg,int(num_trl*(1/float(len(dist_bin))))),2)
trgt_vf = np.hstack((np.repeat(['right'],len(trgt_ecc)),np.repeat(['left'],len(trgt_ecc))))


#trls = np.vstack((trgt_vf,np.tile(trgt_ecc,2)))
#trls = np.vstack((trls,trgt_fl_dist)) #info about trials (3xnum_trl with type,#set,trgt)

trls_idx = range(0,num_trl) #range of indexes for all trials 
ort_lbl = np.append(np.repeat(['right'],num_trl/2),np.repeat(['left'],num_trl/2)) #taget orientation labels


# array to save variables
RT_trl = np.array(np.zeros((num_blk,num_trl)),object) #array for all RTs
key_trl = np.array(np.zeros((num_blk,num_trl)),object) #array for all key presses
display_idx = np.array(np.zeros((num_blk,num_trl)),object) #array for idx of all displays
trgt_ort_lbl = np.array(np.zeros((num_blk,num_trl)),object) #array for target orientations

# create a window
win = visual.Window(size=(hRes, vRes), color = backCol, units='pix',fullscr  = False, screen = 1,allowStencil=True)
#win = visual.Window(size= (hRes, vRes), color = backCol, units='pix',fullscr  = True, screen = 1,allowStencil=True)   
   
#pause
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
            trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(xpos_trgt,0))           
        else:
            xpos_trgt = ang2pix(float(trgt_ecc[trls_idx[k]]),screenHeight,screenDis,vRes)
            trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(xpos_trgt,0))            
             
        trgt.draw()
        
        for i in range(len(ort_fl)):
            xpos_fl,ypos_fl = pol2cart(ang2pix(float(trgt_ecc[trls_idx[k]])*float(trgt_fl_dist[trls_idx[k]]),screenHeight,screenDis,vRes), ort_fl[i])
            flank = visual.GratingStim(win=win,tex='sin',mask='gauss',ori=ort_fl[i],sf=gab_sf,size=siz_gab,pos=(xpos_fl+xpos_trgt,ypos_fl))
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
            
    
        draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation 
        win.flip() # flip the screen
        core.wait(2.0) #pause

    display_idx[j][:] = trls_idx
    trgt_ort_lbl[j][:] = ort_lbl




# save relevant variables in panda dataframe
for d in range(num_blk):
    target_ecc = np.zeros((1,num_trl))
    target_flank_dist= np.zeros((1,num_trl))
    for l in range(num_trl):
        target_ecc[0][l] = trgt_ecc[display_idx[0][l]];
        target_flank_dist[0][l] = trgt_fl_dist[display_idx[0][l]];
    
    dict_var = {'target_orientation':trgt_ort_lbl[d][:], 'target_ecc':target_ecc[0][:], 'target_flank_ratio':target_flank_dist[0][:],'key_pressed':key_trl[d][:],'RT':RT_trl[d][:]}
    if d==0:
        df = pd.DataFrame(data=dict_var)
    else:
        df=pd.concat([df, pd.DataFrame(data=dict_var)])
        
df.to_csv('data_crowding_pp_'+pp+'.csv', sep='\t')

    
    #df = pd.DataFrame(np.array([trgt_ort_lbl[d][:],target_ecc[0][:], target_flank_dist[0][:],key_trl[d][:],RT_trl[d][:]]),columns=['target_orientation', 'target_ecc', 'target_flank_ratio','key_pressed','RT'])
    
#dict_var = {'target_orientation':trgt_ort_lbl,'display_index':display_idx,'key_press':key_trl,'RT':RT_trl}

#save data of interest
#with open('data_crowding_pp_' + pp + '.pickle', 'wb') as write_file:
#    pickle.dump(dict_var, write_file,protocol=pickle.HIGHEST_PROTOCOL)
    
#cleanup
win.close() #close display
core.quit()




