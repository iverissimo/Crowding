
# to save common functions

# then I don't need to reapeat all the time


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

# functions

# convert edf file to asc file
def convert2asc(sj,taskname,outVS,outCRWD):
    
    import os, glob, shutil
    from pygazeanalyser.edfreader import read_edf
    
    if type(sj) == int: #if sub num integer
        sj = str(sj).zfill(2) #turn to string
    
    #list of absolute paths to all edf files in that session for that subject
    #single hdf5 file that contains all eye data for the runs of that session
    if taskname=='vs':
        edf_file = glob.glob(os.path.join(outVS, 'eyedata_visualsearch_pp_%s.edf' %sj))[0]
        asc_dir = outVS+'/pp-{sj}/'.format(sj=sj)
    elif taskname=='crowding':
        edf_file = glob.glob(os.path.join(outCRWD, 'eyedata_crowding_pp_%s.edf' %sj))[0]
        asc_dir = outCRWD+'/pp-{sj}/'.format(sj=sj)
           
    if not os.path.exists(asc_dir): # check if path to save hdf5 files exists
        os.makedirs(asc_dir)     # if not create it
    
    os.system("edf2asc %s" % edf_file)
    
    asc_file = os.path.split(edf_file)[1].replace('.edf','.asc')
    shutil.move(os.path.join(os.path.split(edf_file)[0],asc_file),asc_dir+asc_file)
    
    asc_filename = asc_dir+asc_file
    edfdata = read_edf(asc_filename, 'start_trial', stop='stop_trial', debug=False)
    
    return asc_filename, edfdata #name of asccii, actual gaze data

# turn visual angle in degrees
def ang2pix(dist_in_deg,h,d,r): 
    import math 
    
    deg_per_px = math.degrees(math.atan2(0.5*h,d))/(0.5*r)
    dist_in_px = dist_in_deg/deg_per_px
    return dist_in_px 

# function to draw all gaze points in trial on top of display
def draw_rawdata_display(sj,trial,filename,target,distr,gazedata,hRes,vRes,radius):
        
    import matplotlib.pyplot as plt
    import os
    COLS ={"aluminium": ['#eeeeec',
                    '#d3d7cf',
                    '#babdb6',
                    '#888a85',
                    '#555753',
                    '#2e3436'],
            }

    # set figure and main axis
    fig, ax = plt.subplots()
    # change default range 
    ax.set_xlim((0, hRes))
    ax.set_ylim((0, vRes))
    
    # add target as red circle
    trg_circle = plt.Circle((target[0], target[1]), radius, color='r')
    ax.add_artist(trg_circle)

    # add distractors as blue circles
    for i in range(len(distr)):
        distr_circle = plt.Circle((distr[i][0], distr[i][1]), radius, color='b')
        ax.add_artist(distr_circle)

    # add new axis that can be inverted, for gaze data
    # invert y axis, as (0,0) is top left on a display
    ax1 = ax.twinx()
    ax1.set_xlim((0, hRes))
    ax1.set_ylim((0, vRes))
    ax1.invert_yaxis()
    ax1.get_yaxis().set_visible(False) #but don't show it
    ax1.get_xaxis().set_visible(False)

    # plot raw data points
    x = gaze[0]
    y = gaze[1] # did this to invert y axis, as (0,0) is top left on a display (#ax.invert_yaxis() will do whole thing, dont want that)

    ax1.plot(gazedata[0], gazedata[1], 'o', color=COLS['aluminium'][0], markeredgecolor=COLS['aluminium'][5])

    plt.show()
    
    absfile = os.path.join(os.path.split(filename)[0], "rawdata_pp-%s_trial-%s.png" % (sj,str(trial).zfill(3)))
    if not os.path.exists(absfile): # if file not in dir, save
        fig.savefig(absfile, dpi=1000)

    
# function to draw fixation locations and saccade path in trial on top of display   
def draw_scanpath_display(sj,trial,filename,target,distr,fixation,saccades,hRes,vRes,radius):
    
    import matplotlib.pyplot as plt
    import os
    COLS ={"aluminium": ['#eeeeec',
                    '#d3d7cf',
                    '#babdb6',
                    '#888a85',
                    '#555753',
                    '#2e3436'],
            "chameleon": ['#8ae234',
                '#73d216',
                '#4e9a06']
            }
    
    # set figure and main axis
    fig, ax = plt.subplots()
    # change default range 
    ax.set_xlim((0, hRes))
    ax.set_ylim((0, vRes))
    
    # add target as red circle
    trg_circle = plt.Circle((target[0], target[1]), radius, color='r')
    ax.add_artist(trg_circle)

    # add distractors as blue circles
    for i in range(len(distr)):
        distr_circle = plt.Circle((distr[i][0], distr[i][1]), radius, color='b')
        ax.add_artist(distr_circle)


    # ADD DISPLAY
    # add new axis with different orientation of gaze data
    ax1 = ax.twinx()
    ax1.set_xlim((0, hRes))
    ax1.set_ylim((0, vRes))
    ax1.get_yaxis().set_visible(False) #but don't show it
    ax1.get_xaxis().set_visible(False)
    # invert the y axis, as (0,0) is top left on a display
    ax1.invert_yaxis()

    alpha = 0.5 # alpha level for scanpath drawings
    
    # FIXATIONS
    # parse fixations
    from pygazeanalyser.gazeplotter import parse_fixations
    fix = parse_fixations(fixations)
    # draw fixations, size of dot depends on duration
    ax1.scatter(fix['x'],fix['y'], s=fix['dur'], c=COLS['chameleon'][2], marker='o', cmap='jet', alpha=alpha, edgecolors='none')
    # draw annotations (fixation numbers)
    for i in range(len(fixations)):
        ax1.annotate(str(i+1), (fix['x'][i],fix['y'][i]), color=COLS['aluminium'][5], alpha=1, horizontalalignment='center', verticalalignment='center', multialignment='center')

    # SACCADES
    if saccades:
        # loop through all saccades
        for st, et, dur, sx, sy, ex, ey in saccades:
            # draw an arrow between every saccade start and ending
            ax1.arrow(sx, sy, ex-sx, ey-sy, alpha=alpha, fc=COLS['aluminium'][0], ec=COLS['aluminium'][5], fill=True, shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)
    
    absfile = os.path.join(os.path.split(filename)[0], "scanpath_pp-%s_trial-%s.png" % (sj,str(trial).zfill(3)))
    if not os.path.exists(absfile): # if file not in dir, save
        fig.savefig(absfile, dpi=1000)