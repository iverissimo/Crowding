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

## append path to Psycholink folder, cloned from https://github.com/jonathanvanleeuwen/psychoLink.git
#sys.path.append(os.getcwd()+'/psychoLink/PsychoLink')
#import psychoLink as PL


############# functions #################

# draw fixation cross function
def draw_fixation(posit,lineSize,linecolor,linewidth): 
    
    t = lineSize/2.0
    vertical_line = visual.Line(win,start = (posit[0],posit[1]-t),end = (posit[0],posit[1]+t),lineColor = linecolor,lineWidth=linewidth)
    horizontal_line = visual.Line(win,start = (posit[0]-t,posit[1]),end = (posit[0]+t,posit[1]),lineColor = linecolor,lineWidth=linewidth)
    
    vertical_line.draw()
    horizontal_line.draw() 


# Calculate the number of degrees that correspond to a single pixel
# and then calculate distance in pixels
# dist_in_deg - distance in deg of visual angle
#h - height of screen, d - distance from monitor, r - vertical resolution
def ang2pix(dist_in_deg,h,d,r): 
    deg_per_px = math.degrees(math.atan2(0.5*h,d))/(0.5*r)
    dist_in_px = dist_in_deg/deg_per_px
    return dist_in_px 

#define unique trials,with balanced set size and #ecc
# return min number of balanced trials, with target eccs and set size    
def uniq_trials(ecc,set_size):
    trials = len(set_size)*len(ecc)
    
    tg_ecc = np.tile(ecc,len(set_size))
    
    tg_set_size = []
    for i in range(len(set_size)):  
        tg_set_size.append(np.repeat([set_size[i]],trials/float(len(set_size))))
    tg_set_size = np.ravel(tg_set_size)
    
    return trials,tg_ecc,tg_set_size

# define positions in circle given
# r - list of radius, n - list number of points per radius 

def circle_points(r, n):
    
    circles = []
    for r, n in zip(r, n):
        t = np.arange(0,2*np.pi,2*np.pi/float(n)) #np.linspace(0, 2 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles         

# define positions within ellipse
# pos - list of [x,y] positions
# a - 1/2 of major xx axis of ellipse, b - 1/2 of minor axis of ellipse

def ellipse_inpoints(pos,a,b):
    
    ellipse = []
    for i in range(len(pos)):
        
        if (pos[i][0]**2)/(a**2) + (pos[i][1]**2)/(b**2) <=  1: 
            ellipse.append(pos[i])
    
    return ellipse
    
# count total set size
# poslist - list of positions 
def count_set_size(poslist):
    a = []
    for i in range(len(pos_list)):
        a.append(len(pos_list[i]))
    num_pos = sum(a)

    return num_pos


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

#pp = str(raw_input('Subject number: ').zfill(2))

output_dir = os.getcwd()+'/output_VS/'

if not os.path.exists(output_dir): #check if path to save output exists
        os.makedirs(output_dir)       # if not create it
     
####################################################
    
# variables to save in settings json
   
h = 21.9
d = 50.0
r = 800.0    

params['ort_trgt'] = [5,355]

ecc = np.array(range(2,24,2))
ecc_pix = [ang2pix(j,h,d,r) for _,j in enumerate(ecc)] # in pixels
trgt_ecc = [4,8,12]
n_points = [(i+1)*6 for i, _ in enumerate(ecc)] # number of points per ecc

set_size = [5,15,30] #number of items to be displayed

num_blk = 5 #total number of blocks
num_rep = 20 #number of repetions of unique display per block 

num_trl,tg_ecc,tg_set_size = uniq_trials(trgt_ecc,set_size)
num_trl = num_trl*num_rep #total number of trials per block

trls_idx = np.repeat(range(0,num_trl/num_rep),(num_rep)) #range of indexes for all trials 
ort_lbl = np.append(np.repeat(['right'],num_trl/2),np.repeat(['left'],num_trl/2)) #target orientation labels



siz_gab = ang2pix(1.6,h,d,r) #size in deg of visual angle
gab_sf = 3/ang2pix(1,h,d,r) #sf cycles per pixel = sf_CyclesDeg / pixelsPerDegree
sd_gab = ang2pix(0.075,h,d,r) #standard deviation of gaussian

fixpos = (0,0)
fixlineSize = 15
fixcolor = 'white'
linewidth = 2

ax_major_deg = ecc[-1] #size of 1/2 of major axis - parallel to xx - in degrees
ax_major_pix = ang2pix(ax_major_deg,h,d,r)

ax_minor_deg = trgt_ecc[-1]+2 #size of 1/2 minor axis - parallel to yy - in degrees
ax_minor_pix = ang2pix(ax_minor_deg,h,d,r)

# define initial circle grid for positions
circles = circle_points(ecc_pix,n_points)
# constrain them within ellipse
pos_list = []
for j in range(len(circles)):   
    pos_list.append(ellipse_inpoints(circles[j],ax_major_pix,ax_minor_pix))

# number of possible positions
num_pos = count_set_size(pos_list)


# array to save variables
RT_trl = np.array(np.zeros((num_blk,num_trl)),object); RT_trl[:]=np.nan #array for all RTs
key_trl = np.array(np.zeros((num_blk,num_trl)),object); key_trl[:]=np.nan #array for all key presses
display_idx = np.array(np.zeros((num_blk,num_trl)),object) #array for idx of all displays
trgt_ort_lbl = np.array(np.zeros((num_blk,num_trl)),object) #array for target orientations
distances = np.array(np.zeros((num_blk,num_trl)),object) #array for all distance values


# create a window
#win = visual.Window(size=(hRes, vRes), color = backCol, units='pix',fullscr  = True, screen = 1,allowStencil=True)
#win = visual.Window(size= (params['hRes'], params['vRes']), color = params['backCol'], units='pix',fullscr  = True, screen = 0,allowStencil=True)   

# create a window
win = visual.Window(size= (1280, r), color = "grey", units='pix',fullscr  = True, screen = 1,allowStencil=True)   

#pause
core.wait(2.0)
    
for j in range(num_blk):
    
    np.random.shuffle(ort_lbl) #randomize target orientation
    np.random.shuffle(trls_idx) #randomize index for display
        
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
        
        
        ort_trl = params['ort_trgt'][0] if ort_lbl[k]=='right' else params['ort_trgt'][1] #define target orientation for trial

        tg_pos = rnd.choice(pos_list[np.where(ecc == tg_ecc[trls_idx[k]])[0][0]])  #randomly choose position within a certain ecc for target
        
        # all possible positions for distractors, except for the position already assigned to target
        all_pos = np.concatenate(pos_list).tolist() # had to convert from np array to list to pop
        all_pos.pop(np.where(all_pos == tg_pos)[0][0])
    
        distr_pos = rnd.sample(all_pos,tg_set_size[trls_idx[k]]-1)  #randomly choose positions within a certain set size for distractors (-1 because one pos of set if target)
        
        # draw display
        trgt = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=ort_trl,sf=gab_sf,size=siz_gab,pos=(tg_pos[0],tg_pos[1]),units=None)                        
        trgt.draw()
                
        for i in range(len(distr_pos)):
            distr = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=0,sf=gab_sf,size=siz_gab,pos=(distr_pos[i][0],distr_pos[i][1]),units=None)
            distr.draw()
                
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

win.close() #close display
core.quit()











