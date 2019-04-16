#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:47:51 2019

Author: Ines V.

Descrip:  visual search task, to correlate with crowding results

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

#define unique trials,with balanced set size and #ecc
def unique_trials(ecc,items):
    
    trials = len(items)*len(ecc)
       
    items_per_ecc = np.repeat(items,len(ecc))
    ecc_per_condition = np.tile(ecc,len(items))
    
    return trials,items_per_ecc,ecc_per_condition


#######################
    
# variables to save in settings json
    
ecc = [2,4,6,8,10,12]
trgt_ecc = [4,8,12]

siz_gab = ang2pix(1.6,21.9,50,800) #size in deg of visual angle
gab_sf = 3/ang2pix(1,21.9,50,800) #sf cycles per pixel = sf_CyclesDeg / pixelsPerDegree
sd_gab = ang2pix(0.075,21.9,50,800) #standard deviation of gaussian

fixpos = (0,0)
fixlineSize = 15
fixcolor = 'white'
linewidth = 2

set_siz = [4,8,16,32,64] #number of elements in set


# define all possible positions of display first

for j in range(len(ecc)):   
    
    posrow = np.arange(0,360,360/(6*(j+1))) # angles for each gabor in the row
    
    if j == 0:
        (xpos,ypos) = pol2cart(ang2pix(ecc[j],21.9,71,800), posrow) 
    else:
        (a,b) = pol2cart(ang2pix(ecc[j],21.9,71,800), posrow)
        xpos = np.append(xpos,a)
        ypos = np.append(ypos,b)


# create a window
win = visual.Window(size= (1280, 800), color = "grey", units='pix',fullscr  = True, screen = 1,allowStencil=True)   
   

draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation 
win.flip() # flip the screen
core.wait(2.0) #pause

for i in range(len(xpos)):
    gabor = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=0,sf=gab_sf,size=siz_gab,pos=(xpos[i],ypos[i]),units=None)
    gabor.draw()

draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation
win.flip() # flip the screen
event.waitKeys(keyList = 'space') 

draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation 
#text3 = 'Move your eyes back to fixation'
#moveEyesText = visual.TextStim(win, text=text3, color='white', height=30, pos = (0,100))
#moveEyesText.draw()
win.flip() # flip the screen
core.wait(2.0) #pause


#cleanup
win.close() #close display
core.quit()



