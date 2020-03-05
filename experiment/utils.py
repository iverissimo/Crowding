
#
# usefull functions for stimulus presentation
#

from psychopy import visual
import numpy as np
import math


########### functions #################

# draw fixation cross function
def draw_fixation(posit,lineSize,linecolor,linewidth,win): 
    
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
def crwd_uniq_trials(ecc):
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


#define unique trials,with balanced set size and #ecc
# return min number of balanced trials, with target eccs and set size    
def vs_uniq_trials(ecc,set_size):
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
    for i in range(len(poslist)):
        a.append(len(poslist[i]))
    num_pos = sum(a)

    return num_pos



