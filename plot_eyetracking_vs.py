
# plot that accepts as input subject number and trial number
# and plots scanpath and raw gaze for that trial in visual search task


import os, sys
from utils import *

# append Pygaz analyser folder, cloned from https://github.com/esdalmaijer/PyGazeAnalyser.git
sys.path.append(os.getcwd()+'/PyGazeAnalyser/')
import pygazeanalyser
import numpy as np

# paths
base_dir =os.getcwd(); base_dir = os.path.join(base_dir,'Data_50ms_7deg')
output_vs = os.path.join(base_dir,'output_VS')

# define dir to save plots
plot_dir = os.path.join(base_dir,'plots','scanpaths')
if not os.path.exists(plot_dir):
     os.makedirs(plot_dir)


# define participant number and open json parameter file
if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')

elif len(sys.argv) < 3:
    raise NameError('Please add trial number (ex:50) '
                    'as 2nd argument in the command line!')

else:
    # fill subject number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)
    trial = int(sys.argv[2])


# scanpath
draw_scanpath_display(sj,trial,output_vs,output_vs,plot_dir)

# raw gaze
draw_rawdata_display(sj,trial,output_vs,output_vs,plot_dir)
     