#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:54:18 2019

@author: inesverissimo

visual search display, as ellipse
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

#define unique trials,with balanced visual field and #ecc
def uniq_trials(ecc):
    trials = 2*len(ecc)
    
    trgt_ecc = np.tile(ecc,2)
    trgt_vf = np.hstack((np.repeat(['right'],trials/2.0),np.repeat(['left'],trials/2.0)))
    
    return trials,trgt_ecc,trgt_vf

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
    

#######################################
    
# variables to save in settings json
   
h = 21.9
d = 50.0
r = 800.0    

ecc = range(2,20,2)
ecc_pix = [ang2pix(j,h,d,r) for _,j in enumerate(ecc)] # in pixels
trgt_ecc = [4,8,12]
n_points = [(i+1)*6 for i, _ in enumerate(ecc)] # number of points per ecc


siz_gab = ang2pix(1.6,h,d,r) #size in deg of visual angle
gab_sf = 3/ang2pix(1,h,d,r) #sf cycles per pixel = sf_CyclesDeg / pixelsPerDegree
sd_gab = ang2pix(0.075,h,d,r) #standard deviation of gaussian

fixpos = (0,0)
fixlineSize = 15
fixcolor = 'white'
linewidth = 2

ax_major_deg = ecc[-1] #size of 1/2 of major axis - parallel to xx - in degrees
ax_major_pix = ang2pix(ax_major_deg,h,d,r)

ax_minor_deg = ax_major_deg/2.0 #size of 1/2 minor axis - parallel to yy - in degrees
ax_minor_pix = ang2pix(ax_minor_deg,h,d,r)

# define initial circle grid for positions
circles = circle_points(ecc_pix,n_points)
# constrain them within ellipse
pos_list = []
for j in range(len(circles)):   
    pos_list.append(ellipse_inpoints(circles[j],ax_major_pix,ax_minor_pix))


'''
a = ax_major_pix
b = ax_minor_pix
pos_list = []

all_ecc = [-x for x in ecc] #all x positions
all_ecc = all_ecc + ecc[1::]

for j in range(len(all_ecc)):

    xpos = ang2pix(all_ecc[j],h,d,r)
    ypos = b*np.sqrt(1-((xpos**2)/(a**2)))

    if ypos == 0:
        ypos = [ypos]
    else:
        ypos = [-ypos,ypos]

    for i in range(len(ypos)):
        pos_list.append([xpos,ypos[i]])
'''       

# create a window
win = visual.Window(size= (1280, r), color = "grey", units='pix',fullscr  = True, screen = 1,allowStencil=True)   
   

draw_fixation(fixpos,fixlineSize,fixcolor,linewidth) #draw fixation 
win.flip() # flip the screen
core.wait(2.0) #pause

#for i in range(len(pos_list)):
#    gabor = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=0,sf=gab_sf,size=siz_gab,pos=(pos_list[i][0],pos_list[i][1]),units=None)
#    gabor.draw()

for rad in range(len(pos_list)):
    for pts in range(len(pos_list[rad])):
        gabor = visual.GratingStim(win=win,tex='sin',mask='gauss',maskParams={'sd': sd_gab},ori=0,sf=gab_sf,size=siz_gab,pos=(pos_list[rad][pts][0],pos_list[rad][pts][1]),units=None)
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












