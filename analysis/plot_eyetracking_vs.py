
# plot that accepts as input subject number and trial number
# and plots scanpath and raw gaze for that trial in visual search task


import os, sys
from utils import *

# append Pygaz analyser folder, cloned from https://github.com/esdalmaijer/PyGazeAnalyser.git
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'PyGazeAnalyser'))
import pygazeanalyser

import numpy as np

# open params jason file
with open(os.path.join(os.getcwd(),'settings.json'),'r') as json_file:	
            params = json.load(json_file)	

# paths
output_vs = params['datadir_vs']

# define dir to save plots
plot_dir = os.path.join(os.path.join(os.path.split(output_vs)[0],'plots','scanpaths'))
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
    sj = str(sys.argv[1]).zfill(3)
    trial = int(sys.argv[2])


# scanpath
draw_scanpath_display(sj,trial,output_vs,plot_dir)

# raw gaze
draw_rawdata_display(sj,trial,output_vs,plot_dir)
     