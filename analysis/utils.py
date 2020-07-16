
# to save common functions

# then I don't need to reapeat all the time


import sys, os

# append Pygaz analyser folder, cloned from https://github.com/esdalmaijer/PyGazeAnalyser.git
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'PyGazeAnalyser'))
import pygazeanalyser

import numpy as np

import pandas as pd
import glob

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
#%matplotlib inline

import json
from pygazeanalyser.gazeplotter import parse_fixations
from scipy.stats import spearmanr
import seaborn as sns

from pygazeanalyser.edfreader import read_edf


def convert2asc(sj,taskname,data_dir):
    
    # convert edf file to asc file
    
    if type(sj) == int: #if sub num integer
        sj = str(sj).zfill(3) #turn to string
    
    # define edf file absolute name
    if taskname == 'search': # visual search task
        edf_file = os.path.join(data_dir,'eyedata_visualsearch_pp_%s.edf'%sj)
    elif taskname == 'crowding':
        edf_file = os.path.join(data_dir,'eyedata_crowding_pp_%s.edf'%sj)
    else:
        raise NameError('Task not recognized, please indicate if search or crowding')

    # define ascii file absolute name
    asc_file = edf_file.replace('.edf','.asc')

    # if ascii doesn't exist create it
    if not os.path.isfile(asc_file):
        print('converting edf to asccii')
        os.system("edf2asc %s" % edf_file)

    
    return asc_file #name of asccii


# turn visual angle in degrees
def ang2pix(dist_in_deg,h,d,r): 
    import math 
    
    deg_per_px = math.degrees(math.atan2(0.5*h,d))/(0.5*r)
    dist_in_px = dist_in_deg/deg_per_px
    return dist_in_px 

def draw_rawdata_display(sj,trial,data_dir,outdir,
                          rad_gab=1.1,fix_line=0.25,screenHeight=30,screenDis=57,
                         vRes = 1050,hRes=1680):
    
    # function to draw raw gaze for a specific trial
    # will draw display (target and distractor position)
    # and x-y gaze throughout time
    #
    #### INPUTS #####
    # sj - subject number (str)
    # trial - trial number (int)
    # df_vs - absolute path dataframe with behaviour data
    # eyedata_vs - absolute path eye data for visual search
    
    behvfile = [x for _,x in enumerate(os.listdir(data_dir)) if x.endswith('.csv') and sj in x]
    eyefile = [x for _,x in enumerate(os.listdir(data_dir)) if x.endswith('.EDF') and sj in x]
    # load csv for sub
    df_vs = pd.read_csv(os.path.join(data_dir,behvfile[0]), sep='\t')

    # load eye data
    asccii_name = convert2asc(sj,'search',data_dir)
    print('loading edf data')
    eyedata_vs = read_edf(asccii_name, 'start_trial', stop='stop_trial', debug=False)
    
    # NOTE - all positions in pixels
    
    # get target and distractor positions as strings in list
    target_pos = df_vs['target_position'][trial].replace(']','').replace('[','').split(' ')
    distr_pos = df_vs['distractor_position'][trial].replace(']','').replace('[','').replace(',','').split(' ')

    # convert to list of floats
    target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])
    distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])

    # save distractor positions in pairs (x,y)
    alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])
    
    f, s = plt.subplots(1, 1, figsize=(8,8))

    # set radius of circles
    r_gabor = ang2pix(rad_gab,screenHeight,
                       screenDis,vRes)
    r_fix = ang2pix(fix_line,screenHeight,
                       screenDis,vRes)
    
    # add target
    s.add_artist(plt.Circle((target_pos[0], target_pos[1]), radius=r_gabor, color='r', fill=True))
    s.set_xlim([-hRes/2,hRes/2])
    s.set_ylim([-vRes/2,vRes/2])
    s.set_aspect('equal') # to not make circles elipses
    
    # add disctractors
    for w in range(len(alldistr_pos)):
        s.add_artist(plt.Circle((alldistr_pos[w][0], alldistr_pos[w][1]), radius=r_gabor, color='b', fill=True))

    # add fixation cross
    s.add_artist(plt.Circle((0, 0), radius=r_fix, color='k', fill=True))

    # add circle of contingency 
    s.add_artist(plt.Circle((0, 0),radius=ang2pix(1,screenHeight,screenDis,vRes), color='k', fill=False))
    
    # plot raw data points
    x = eyedata_vs[trial]['x'] - hRes/2
    y = eyedata_vs[trial]['y'] - vRes/2 
    y = -y # have to invert y, because eyelink considers 0 to be upper corner

    s.plot(x, y, 'o', color='#eeeeec', markeredgecolor='#2e3436',alpha=0.5)

    f.savefig(os.path.join(outdir,'rawgaze_pp-%s_trial-%s.png' % (sj,str(trial).zfill(3))), dpi=1000)
    
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
    ax.plot(np.linspace(0,df_vs['RT'][trial],num=len(x)),x,color='k',label='x')
    ax.plot(np.linspace(0,df_vs['RT'][trial],num=len(y)),y,color='orange',label='y')
    
    plt.title('raw gaze trial %d sub %s'%(trial,sj))
    ax.set(ylabel='position [pixels]', xlabel='time [s]')
    plt.legend(loc='upper left')
    
    fig.savefig(os.path.join(outdir,'rawgaze_xy_pp-%s_trial-%s.png' % (sj,str(trial).zfill(3))), dpi=1000)
    

    
def draw_scanpath_display(sj,trial,data_dir,outdir,
                          rad_gab=1.1,fix_line=0.25,screenHeight=30,screenDis=57,
                         vRes = 1050,hRes=1680):
    # function to draw scanpath for a specific trial
    # will draw display (target and distractor position)
    # fixation positions and saccade path
    #
    #### INPUTS #####
    # sj - subject number (str)
    # trial - trial number (int)
    # df_vs - absolute path dataframe with behaviour data
    # eyedata_vs - absolute path eye data for visual search

    
    behvfile = [x for _,x in enumerate(os.listdir(data_dir)) if x.endswith('.csv') and sj in x]
    eyefile = [x for _,x in enumerate(os.listdir(data_dir)) if x.endswith('.EDF') and sj in x]
    # load csv for sub
    df_vs = pd.read_csv(os.path.join(data_dir,behvfile[0]), sep='\t')
    
    # load eye data
    asccii_name = convert2asc(sj,'search',data_dir)
    print('loading edf data')
    eyedata_vs = read_edf(asccii_name, 'start_trial', stop='stop_trial', debug=False)
    
    # NOTE - all positions in pixels
    
    # get target and distractor positions as strings in list
    target_pos = df_vs['target_position'][trial].replace(']','').replace('[','').split(' ')
    distr_pos = df_vs['distractor_position'][trial].replace(']','').replace('[','').replace(',','').split(' ')

    # convert to list of floats
    target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])
    distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])

    # save distractor positions in pairs (x,y)
    alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])
    
    f, s = plt.subplots(1, 1, figsize=(8,8))

    # set radius of circles
    r_gabor = ang2pix(rad_gab,screenHeight,
                       screenDis,vRes)

    r_fix = ang2pix(fix_line,screenHeight,
                       screenDis,vRes)
    
    # add target
    s.add_artist(plt.Circle((target_pos[0], target_pos[1]), radius=r_gabor, color='r', fill=True))
    s.set_xlim([-hRes/2,hRes/2])
    s.set_ylim([-vRes/2,vRes/2])
    s.set_aspect('equal') # to not make circles elipses
    
    # add disctractors
    for w in range(len(alldistr_pos)):
        s.add_artist(plt.Circle((alldistr_pos[w][0], alldistr_pos[w][1]), radius=r_gabor, color='b', fill=True))

    # add fixation cross
    s.add_artist(plt.Circle((0, 0), radius=r_fix, color='k', fill=True))

    # add circle of contingency 
    s.add_artist(plt.Circle((0, 0),radius=ang2pix(1,screenHeight,screenDis,vRes), color='k', fill=False))


    # FIXATIONS
    # parse fixations
    # fix is dictionary with list of x,y, and duration of fixations for trial
    fix = parse_fixations(eyedata_vs[trial]['events']['Efix'])

    # SACCADES
    if eyedata_vs[trial]['events']['Esac']:
        # loop through all saccades
        for st, et, dur, sx, sy, ex, ey in eyedata_vs[trial]['events']['Esac']:

            # make positions compatible with display
            sx = sx - hRes/2 # start x pos
            ex = ex - hRes/2 # end x pos
            sy = sy - vRes/2; sy = -sy #start y pos
            ey = ey - vRes/2; ey = -ey #end y pos

            # draw an arrow between every saccade start and ending
            s.arrow(sx, sy, ex-sx, ey-sy, alpha=0.5, fc='k', 
                    fill=True, shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)

    # make positions compatible with display
    fix['x'] = fix['x'] - hRes/2
    fix['y'] = fix['y'] - vRes/2; fix['y'] = -fix['y']

    # draw fixations, size of dot depends on duration
    s.scatter(fix['x'],fix['y'], s=fix['dur'], c='grey', marker='o', cmap='jet', alpha=0.5, edgecolors='none')

    # draw annotations (fixation numbers)
    for z in range(len(fix['x'])):
        s.annotate(str(z+1), (fix['x'][z],fix['y'][z]), 
                   color='w', alpha=1, horizontalalignment='center', verticalalignment='center', multialignment='center')
    
    f.savefig(os.path.join(outdir,'scanpath_pp-%s_trial-%s.png' % (sj,str(trial).zfill(3))), dpi=1000)





def check_nan_trials(data,exclusion_thresh=0.25):
    # function to check for missed trials (no response)
    ### inputs ###
    # data - behav dataframe
    # exclusion_thresh - percentage of trials on which to exclude subject
    ### outputs ###
    # missed_trials - percentage of missed trials
    # exclude - bool saying if we should exclude sub or not
    
    exclude = False
    miss = data['key_pressed'].isna().sum() #easier way to count NaNs in dataframe
    # check for missed trials (trials where they didn't respond)
    missed_trials = miss/len(data)#len(miss)/len(data_sub)
    
    if missed_trials>exclusion_thresh:
        print('Subject missed %.2f %% of trials (no response) based on behavioural data \n'\
          'EXCLUDE SUBJECT'%(missed_trials*100))
        exclude = True
    else:
        print('Subject missed %.2f %% of trials (no response) based on behavioural data \n'\
              'continuing analysis for crowding'%(missed_trials*100))

    return missed_trials*100, exclude


def accuracy_crowding(data,ecc,exclusion_thresh=0.6):
    # function to check accuracy for crowding
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # ecc - list with eccs used in task
    # exclusion_thresh - accuracy value on which to exclude subject
    #
    # OUTPUTS #
    # acc_ecc_flank - accuracy for all ecc, with flankers condition
    # acc_ecc_noflank - accuracy for all ecc, without flankers condition
    # overall_acc - overall accuracy
    # exclude - bool saying if we should exclude sub or not
    
    exclude = False
    
    # initialize counters
    flank_corr = np.zeros((len(ecc))) # counter for correct answers, with flanker trials
    flank_ecc = np.zeros((len(ecc))) # ecc counter for with flanker trials
    noflank_corr = np.zeros((len(ecc))) # counter for correct answers, no flanker trials
    noflank_ecc = np.zeros((len(ecc))) # ecc counter for no flanker trials
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of strings indicating flanker presence/absent
    flnk_pres = data['flanker_presence'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    
    for i in range(len(data)): # for all actual trials (576) 
        
        # ecc index for this trial
        ecc_idx = np.where(np.array(ecc)==int(target_ecc[i]))[0][0]
        
        if flnk_pres[i]=='flankers': # if flanker trial
            flank_ecc[ecc_idx] += 1 #increment overall counter for right ecc
            
            # if key pressed and target orientation the same  
            if key_or[i]==target_or[i]:
                flank_corr[ecc_idx] += 1 #increment correct answer counter for right ecc
            
        
        else: #no flanker trial
            noflank_ecc[ecc_idx] += 1 #increment overall counter for right ecc

            # if key pressed and target orientation the same  
            if key_or[i]==target_or[i]:
                noflank_corr[ecc_idx] += 1 #increment correct answer counter for right ecc
            
    
    acc_ecc_flank = flank_corr/flank_ecc # accuracy with flankers, per ecc
    acc_ecc_noflank = noflank_corr/noflank_ecc # accuracy withou flankers, per ecc
    overall_acc = (flank_corr.sum()+noflank_corr.sum())/(flank_ecc.sum()+noflank_ecc.sum()) # overall accuracy
    print('Accuracy with flankers is %s %% for ecc %s'%(str(acc_ecc_flank*100),str(ecc)))
    print('Accuracy without flankers is %s %% for ecc %s'%(str(acc_ecc_noflank*100),str(ecc)))
    
    if overall_acc<exclusion_thresh:
        print('Subject has overall accuracy of %.2f %% \n'\
              'EXCLUDE SUBJECT'%(overall_acc*100))
        exclude = True
    else:
        print('Subject has overall accuracy of %.2f %% \n' \
              'continuing analysis for crowding'%(overall_acc*100))

    return {'acc_ecc_flank': acc_ecc_flank, 'acc_ecc_noflank': acc_ecc_noflank ,
            'overall_acc': overall_acc,'exclude':exclude} 


def critical_spacing(data,ecc,num_trl=96):
    # function to calculate critical spacing
    
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # ecc - list with eccs used in task
    # num_trl - number of trials to use for CS
    #
    # OUTPUTS #
    # all_tfr - all target flanker rations, per ecc
    # all_crit_dis - all critical distances, per ecc
    # mean_CS - mean critical spacing
    # exclude - bool saying if we should exclude sub or not
    
    exclude = False
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of strings indicating flanker presence/absent
    flnk_pres = data['flanker_presence'].values
    # list of values with the target flanker ratio
    target_fl_ratio = data['target_flank_ratio'].values
    
    all_tfr = [] # list to append all ecc tfr
    all_crit_dis = [] # list of critical distances for all ecc
    
    for _,j in enumerate(ecc):
        
        tfr = []
    
        for i in range(len(data)): # for all actual trials (576) 
            if flnk_pres[i]=='flankers' and int(target_ecc[i])==j: # if flanker trial and coreect ecc
                
                tfr.append(target_fl_ratio[i]) # append target flanker ratio
                
        all_tfr.append(np.array(tfr))
        all_crit_dis.append(np.median(tfr[-num_trl:]))
        
        
    mean_CS = np.nanmean(np.array(all_crit_dis))

    #if np.around(mean_CS,decimals=1)==.2 or np.around(mean_CS,decimals=1)==.8:

    #boolean arrays to check bottom or ceiling effects
    bol_ceiling = [True for _,val in enumerate(np.array(all_crit_dis)) if np.round(val,decimals=2)==0.2]
    bol_bottom = [True for _,val in enumerate(np.array(all_crit_dis)) if np.round(val,decimals=2)==0.8]

    if (len(bol_ceiling) >= 2) or (len(bol_bottom) >= 2): # if they are at min or max for 2 out of the 3 ecc, we exclude
        print('Mean critical spacing is %.3f for subjectso bottom/ceiling effect\n EXCLUDE SUBJECT'%(mean_CS))
        exclude = True
    else:
        print('Mean critical spacing is %.3f for subject'%(mean_CS))
        
    
    return {'all_tfr': all_tfr, 'all_crit_dis': all_crit_dis ,
            'mean_CS': mean_CS,'exclude':exclude} 



def accuracy_search(data,ecc,exclusion_all_thresh=0.85,exclusion_ecc_thresh=0.75):
    # function to check accuracy for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # ecc - list with eccs used in task
    # exclusion_thresh - accuracy value on which to exclude subject
    #
    # OUTPUTS #
    # acc_ecc - accuracy for all ecc
    # overall_acc - overall accuracy
    # exclude - bool saying if we should exclude sub or not
    
    exclude = False
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    
    # initialize counters
    ans_corr = np.zeros((len(ecc))) # counter for correct answers, with flanker trials
    ans_ecc = np.zeros((len(ecc))) # ecc counter for with flanker trials
        
    for i in range(len(data)): # for all actual trials 
        
        # ecc index for this trial
        ecc_idx = np.where(np.array(ecc)==int(target_ecc[i]))[0][0]
        
        ans_ecc[ecc_idx] += 1 #increment overall counter for right ecc
                    
        if key_or[i]==target_or[i]: # if key press = target orientation
            ans_corr[ecc_idx] += 1 #increment correct answer counter for right ecc

    acc_ecc = ans_corr/ans_ecc # accuracy, per ecc
    overall_acc = ans_corr.sum()/ans_ecc.sum() # overall accuracy
    print('Accuracy is %s %% for ecc %s'%(str(acc_ecc*100),str(ecc)))
    
    if overall_acc<exclusion_all_thresh or min(acc_ecc)<exclusion_ecc_thresh:
        print('Accuracy for visual search is %s %% for subject \n'\
              'EXCLUDE SUBJECT'%(str(acc_ecc*100)))
        exclude = True
    else:
        print('Accuracy for visual search is %s %% for subject'%(str(acc_ecc*100)))

    return {'acc_ecc': acc_ecc,
            'overall_acc': overall_acc,'exclude':exclude} 
        
           
    

def plot_correlation(arr_x,arr_y,label_x,label_y,plt_title,outfile,p_value=0.05,
                     line_color='dimgrey',scatter_color = 'grey',
                     x_lim = [.5,2.5], y_lim = [.2,.8],decimals=1):
    
    corr, pval = spearmanr(arr_x,arr_y)

    print('correlation = %.6f, p-value = %.6f'%(corr,pval))
    if pval<p_value:
        print('SIGNIFICANT CORRELATION')

    fig = plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    df4plot = pd.DataFrame({'xx': arr_x,
                            'yy': arr_y})

    ax = sns.lmplot(x='xx', y='yy',data=df4plot,
                    line_kws={'color': line_color},scatter_kws={'color': scatter_color})
    #ax.set(ylabel = label_y, xlabel = label_x)
    #ax = plt.gca()
    #ax.set_title(plt_title+' (rho=%.2f,pval=%.3f)'%(corr,pval))
    
    plt.ylabel(label_y,fontsize=16,labelpad=10)
    plt.xlabel(label_x,fontsize=16,labelpad=10)
    plt.title(r'$\rho$'+' = %.2f, p = %.3f'%(corr,pval),fontsize=18,pad=10)
    
    plt.ylim(y_lim)
    plt.xlim(x_lim[0],None)
    
    plt.xticks(np.round(np.linspace(x_lim[0], np.round(max(arr_x),decimals=decimals), num=4),decimals=decimals),fontsize = 12)
    plt.yticks(fontsize = 12)
    
    
    sns.despine(offset=15)

    plt.savefig(outfile, dpi=100,bbox_inches = 'tight')

    return corr,pval


def surface_density(df_vs_dir,rad_roi,
                    rad_gab=1.1,screenHeight=30,screenDis=57,
                    vRes = 1050,hRes=1680):
    
    # function to calculate surface density values for all trials of visual search for subject
    #### INPUTS #####
    # df_vs - absolute path dataframe with behaviour data
    # rad_roi - radius around target, will be the ROI
    
    # returns
    # surf_density for all trials
    
    df_vs = pd.read_csv(df_vs_dir, sep='\t')
    
    # set radius around target to compute density (ROI)
    radius_roi = ang2pix(rad_roi,screenHeight,
                       screenDis,vRes)
    # radius of each gabor
    radius_gabor = ang2pix(rad_gab,screenHeight,
                       screenDis,vRes)

    # area of ROI to analyse
    roi_area = np.pi*(radius_roi)**2 - np.pi*(radius_gabor)**2 # subtract area of target 
    
    surf_density = []
    
    for trial in range(len(df_vs)): # for all trials
        
        # NOTE - all positions in pixels
    
        # get target and distractor positions as strings in list
        target_pos = df_vs['target_position'][trial].replace(']','').replace('[','').split(' ')
        distr_pos = df_vs['distractor_position'][trial].replace(']','').replace('[','').replace(',','').split(' ')

        # convert to list of floats
        target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])
        distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])

        # save distractor positions in pairs (x,y)
        alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])
        
        # convert positions to make everything positive and hence my life easier
        # (changing coordinate axis to have center position in bottom left of display 
        # instead of center position in center of screen)

        new_target_pos = target_pos + [hRes/2,vRes/2]
        new_alldistr_pos = alldistr_pos + [hRes/2,vRes/2]
        
        # compute total distractor area within ROI
        intersect_area = []

        for i in range(len(new_alldistr_pos)): # for all distractors
            # calculate distance between center of target and center of distractor
            distance = np.sqrt((new_target_pos[0]-new_alldistr_pos[i][0])**2+(new_target_pos[1]-new_alldistr_pos[i][1])**2)

            if distance >= radius_roi+radius_gabor: # circles not intersecting
                inters = 0

            elif distance + radius_gabor <= radius_roi: # distractor fully within ROI
                inters = np.pi*(radius_gabor)**2

            else: # distractor partially intersecting ROI
                # whole description of formula in https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6
                d1 = ((radius_roi)**2-(radius_gabor)**2+distance**2)/(2*distance)
                d2 = ((radius_gabor)**2-(radius_roi)**2+distance**2)/(2*distance)

                A1 = ((radius_roi)**2)*np.arccos(d1/radius_roi) - d1*np.sqrt((radius_roi)**2-d1**2)
                A2 = ((radius_gabor)**2)*np.arccos(d2/radius_gabor) - d2*np.sqrt((radius_gabor)**2-d2**2)

                inters = A1+A2

            intersect_area.append(inters)

        # sum of distractor areas
        dist_area = sum(intersect_area)

        if dist_area>roi_area: # if sum of distractor area bigger than ROI throw error, something is off
            raise ValueError('sum of disctractor area bigger than ROI for trial %d!'%trial)

        # surface density for trial    
        surf_density.append(dist_area/roi_area)

    return np.array(surf_density)


def density_mean_RT(data,density_arr,sub,density='high', ecc=[4,8,12],setsize=[5,15,30]):
    # function to check RT as function of ecc or set size for visual search
    # ends up being a 3x3 dataframe
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # density_arr - surface density array for all trials
    # density - select high or low density trials
    #
    # OUTPUTS #
    # RT_all - mean RT for all groups
        
        
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    # list RT
    RT = data['RT'].values
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of values with display set size
    target_set = data['set_size'].values
    
    # dataframe to output values
    df_out = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
    # dataframe to count the number of trials for each condition, sanity check
    df_trial_num = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub']) 
    
    for _,s in enumerate(setsize): # for all set sizes 
        for _,e in enumerate(ecc): # for all target ecc

            RT_ecc_set = []
            num_trial_ecc_set = 0

            for t in range(len(data)): # for all actual trials 
                
                # if key press = target orientation and specific ecc and specific set size
                if (key_or[t]==target_or[t]) and (int(target_ecc[t])==e) and (int(target_set[t])==s):
                    
                    # save depending on density of trial
                    if (density=='high' and density_arr[t]==True) or (density=='low' and density_arr[t]==False):
                        
                        if RT[t]> .250 and RT[t]<5 : # reasonable search times
                            RT_ecc_set.append(RT[t]) #append RT value
                            num_trial_ecc_set+=1 # increment trial counter

            if not RT_ecc_set: # if empty
                RT_ecc_set = float('Inf')

            # compute that mean RT and save in data frame           
            df_out = df_out.append({str(e)+'_ecc': np.nanmean(RT_ecc_set), 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
            # save number of trials
            df_trial_num = df_trial_num.append({str(e)+'_ecc': num_trial_ecc_set, 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
    # now reshape the dataframe (droping nans and making sure labels are ok)    
    df_out = df_out.apply(lambda x: pd.Series(x.dropna().values))
    df_out = df_out.dropna()
    
    for i in df_out.index:
        df_out.at[i, 'set_size'] = setsize[i]
      
    df_out = df_out.replace(np.inf,np.nan)  # replace any infinites with nan, makes averaging later easier 
        
    # same for trial number counter   
    df_trial_num = df_trial_num.apply(lambda x: pd.Series(x.dropna().values))
    df_trial_num = df_trial_num.dropna()
    
    for i in df_trial_num.index:
        df_trial_num.at[i, 'set_size'] = setsize[i]
    
    return df_out, df_trial_num

    
def density_meanfix(data,eyedata,density_arr,sub,density='high',
                    ecc=[4,8,12],setsize=[5,15,30],
                    hRes=1680,vRes=1050,screenHeight=30,screenDis=57,size_gab=2.2):
    
    # function to check mean number of fixations as function of ecc or set size for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # eyedata - eyetracking data
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # fix_all - mean fix for all ecc
    
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of values with display set size
    target_set = data['set_size'].values
    
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    # list of values of RT
    RT = data['RT'].values
    # radius of gabor in pixels
    r_gabor = ang2pix(size_gab/2,screenHeight,
                       screenDis,
                       vRes)
    # number of samples in 150ms (we'll not count with fixations prior to 150ms after stim display)
    sample_thresh = 1000*0.150 # 1000Hz * time in seconds
    
    # dataframe to output values
    df_out = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
    # dataframe to count the number of trials for each condition, sanity check
    df_trial_num = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub']) 
    
    for _,s in enumerate(setsize): # for all set sizes 
        for _,e in enumerate(ecc): # for all target ecc

            fix_ecc_set = []
            num_trial_ecc_set = 0

            for t in range(len(data)): # for all actual trials 
                
                # if key press = target orientation and specific ecc and specific set size
                if (key_or[t]==target_or[t]) and (int(target_ecc[t])==e) and (int(target_set[t])==s):
                    
                    # save depending on density of trial
                    if (density=='high' and density_arr[t]==True) or (density=='low' and density_arr[t]==False):
                    
                        # index for moment when display was shown
                        idx_display = np.where(np.array(eyedata[t]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                        # eye tracker sample time of display
                        smp_display = eyedata[t]['events']['msg'][idx_display][0]

                        # get target positions as strings in list
                        target_pos = data['target_position'][t].replace(']','').replace('[','').split(' ')
                        # convert to list of floats
                        target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])

                        num_fix = 0
                        for k,fix in enumerate(eyedata[t]['events']['Efix']):

                            # if fixations between 150ms after display and key press time
                            if (fix[0] > (smp_display+sample_thresh) and fix[0] < np.round(smp_display + RT[t]*1000)):

                                #if fixation not on target (not within target radius)
                                fix_x = fix[-2] - hRes/2
                                fix_y = fix[-1] - vRes/2; fix_y = - fix_y

                                if np.sqrt((fix_x-target_pos[0])**2+(fix_y-target_pos[1])**2) > r_gabor:
                                    num_fix += 1

                        fix_ecc_set.append(num_fix) #append number of fixations for that trial value                    
                        num_trial_ecc_set+=1 # increment trial counter

            if not fix_ecc_set: # if empty
                fix_ecc_set = float('Inf')

            # compute mean number of fixations and save in data frame           
            df_out = df_out.append({str(e)+'_ecc': np.nanmean(fix_ecc_set), 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
            # save number of trials
            df_trial_num = df_trial_num.append({str(e)+'_ecc': num_trial_ecc_set, 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
    # now reshape the dataframe (droping nans and making sure labels are ok)    
    df_out = df_out.apply(lambda x: pd.Series(x.dropna().values))
    df_out = df_out.dropna()
    
    for i in df_out.index:
        df_out.at[i, 'set_size'] = setsize[i]
        
    df_out = df_out.replace(np.inf,np.nan)  # replace any infinites with nan, makes averaging later easier 

    # same for trial number counter   
    df_trial_num = df_trial_num.apply(lambda x: pd.Series(x.dropna().values))
    df_trial_num = df_trial_num.dropna()
    
    for i in df_trial_num.index:
        df_trial_num.at[i, 'set_size'] = setsize[i]
    
    return df_out, df_trial_num


def density_plot_correlation(arr_x_LOW,arr_x_HIGH,arr_y,label_x,label_y,plt_title,outfile,p_value=0.05):
    
    corr1, pval1 = spearmanr(arr_x_LOW,arr_y)

    print('correlation = %.6f, p-value = %.6f'%(corr1,pval1))
    if pval1<p_value:
        print('SIGNIFICANT CORRELATION for LOW density')
        
    corr2, pval2 = spearmanr(arr_x_HIGH,arr_y)

    print('correlation = %.6f, p-value = %.6f'%(corr2,pval2))
    if pval2<p_value:
        print('SIGNIFICANT CORRELATION for HIGH density')

    fig= plt.figure(num=None, figsize=(15,7.5), dpi=100, facecolor='w', edgecolor='k')

    df4plot = pd.DataFrame({'xx_LOW': arr_x_LOW,
                            'xx_HIGH': arr_x_HIGH,
                            'yy': arr_y})


    sns.regplot(x=df4plot['xx_LOW'],y=df4plot['yy'],color='blue', marker='.',label='LOW')
    sns.regplot(x=df4plot['xx_HIGH'],y=df4plot['yy'],color='red', marker='+',label='HIGH')    

    ax = plt.gca()
    ax.set(ylabel = label_y, xlabel = label_x)
    ax.set_title(plt_title+' LOW - (rho=%.2f,pval=%.3f) HIGH - (rho=%.2f,pval=%.3f)'%(corr1,pval1,corr2,pval2))
    ax.legend()
    plt.savefig(outfile, dpi=100,bbox_inches = 'tight')



def check_fix(eyedata,exclusion_thresh=.10,rad=1,stim_time=0.05,
              hRes=1680,vRes=1050,screenHeight=30,screenDis=57):
    
    # check if sub if fixating in beginning of trial
    # or if needs to be excluded
    # made for crowding task
    
    exclude = False

    # radius around fixation to look at in pixels
    rad_fix = ang2pix(rad,screenHeight,
                       screenDis,
                       vRes)

    fix_center = [] # list to append if trial was fixating in center or not

    for t,_ in enumerate(eyedata): # for all trials

        # index for moment when display was shown
        idx_display = np.where(np.array(eyedata[t]['events']['msg'])[:,-1]=='var VF left\n')[0]
        if not idx_display:
            idx_display = np.where(np.array(eyedata[t]['events']['msg'])[:,-1]=='var VF right\n')[0]

        idx_display = idx_display[0]

        # eye tracker sample time of display
        smp_display = eyedata[t]['events']['msg'][idx_display][0]

        num_fix_out = 0 # fixation out of center counter
        for k,fix in enumerate(eyedata[t]['events']['Efix']):

            # if fixations within display time
            if (fix[0] > smp_display and fix[0] < np.round(smp_display + stim_time*1000)):
                #if fixation not on center
                fix_x = fix[-2] - hRes/2
                fix_y = fix[-1] - vRes/2; fix_y = - fix_y

                if np.sqrt((fix_x-hRes/2)**2+(fix_y-vRes/2)**2) > rad_fix:
                    num_fix_out += 1

        if num_fix_out>0: # if any fixation out of center
            fix_center.append(False)
        else:
            fix_center.append(True) # if fixating in center


    percent_fix = np.sum(fix_center)/len(eyedata)

    if percent_fix<(1-exclusion_thresh):
        print('Subject only fixates in center for %.2f %% of trials \n'\
              'EXCLUDE SUBJECT'%(percent_fix*100))
        exclude = True
    else:
        print('Subject fixates in center for %.2f %% of trials \n' \
              'continuing analysis for crowding'%(percent_fix*100))

    return {'percent_fix': percent_fix,'exclude':exclude} 




def exclude_subs(crwd_csv,crwd_edf,vs_csv,out_dir,trials_block=144,miss_trials=0.25,acc_cut_off_crwd=0.6,ecc=[4,8,12],
                                        num_cs_trials=96,cut_off_acc_vs=0.85,cut_off_acc_ecc_vs=0.75):
    # function to load all behavior and eyetracking data and 
    # check if subject should be excluded
    # giving back summary file and structure with relevant measure for
    # further analysis/plotting
    
    ### INPUTS ########
    # crwd_csv - list of absolute paths to crowding csv files
    # vs_csv - list of absolute paths to search csv files
    # out_dir - path to save outputs
    
    
    # loop over subjects
    
    all_subs = [] # list with all subject number
    missed_trials = [] # list with percentage of missed trials (crowding)
    acc_fl = [] # accuracy crowding flankers
    acc_nofl = [] # accuracy crowding no flankers
    acc_comb = [] # accuracy crowding combined

    all_tfr = [] # target-flanker ratios
    all_cs = [] # critical spacings
    mean_cs = [] # mean CS

    percent_fix_crwd = [] # percentage of trials were they were fixating in center

    acc_vs_ecc = [] # accuraccy visual search per ecc
    acc_vs_all = [] # accuracy visual search all

    rt_vs = [] # RT visual search 

    excluded_sub = [] # excluded subjects
    
    
    for ind,behfile in enumerate(crwd_csv):
    
        EXCLUDE = []
        
        all_subs.append(os.path.splitext(behfile)[0][-3::]) 
        print('analysing pp-%s'%all_subs[ind])
        
        # CROWDING

        # load crowding csv for sub
        data_sub = pd.read_csv(behfile, sep='\t')
        # choose only actual pratice block (first block (first 144) where used to save training, doesn't count)
        data_sub = data_sub.loc[int(trials_block)::]

        # check for missed trials
        missed_trials.append(check_nan_trials(data_sub,exclusion_thresh=miss_trials)[0])
        EXCLUDE.append(check_nan_trials(data_sub,exclusion_thresh=miss_trials)[1])

        # check for accuracy with and without flankers and combined
        # so actually see if crowding manipulation worked
        accur_crowding = accuracy_crowding(data_sub,ecc,exclusion_thresh=acc_cut_off_crwd)
        
        acc_fl.append(accur_crowding['acc_ecc_flank'])
        acc_nofl.append(accur_crowding['acc_ecc_noflank'])
        acc_comb.append(accur_crowding['overall_acc'])
        EXCLUDE.append(accur_crowding['exclude'])

        cs = critical_spacing(data_sub,ecc,num_trl=num_cs_trials)
        all_tfr.append(cs['all_tfr'])
        all_cs.append(cs['all_crit_dis'])
        mean_cs.append(cs['mean_CS'])
        EXCLUDE.append(cs['exclude'])

        # check if subject was fixating in the center during crowding
        # load eye data
        asccii_name = convert2asc(int(os.path.split(crwd_edf[ind])[-1][-7:-4]),'crowding',os.path.split(crwd_edf[ind])[0])
        print('loading edf for sub-%s data'%int(os.path.split(crwd_edf[ind])[-1][-7:-4]))
        eyedata = read_edf(asccii_name, 'start_trial', stop='stop_trial', debug=False)
        
        fix_center = check_fix(eyedata)
        percent_fix_crwd.append(fix_center['percent_fix'])

        EXCLUDE.append(fix_center['exclude'])
        
        
        # SEARCH
        
        # load csv for sub
        df_vs = pd.read_csv(vs_csv[ind], sep='\t')

        # check accuracy
        vs_acc = accuracy_search(df_vs,ecc,exclusion_all_thresh=cut_off_acc_vs,exclusion_ecc_thresh=cut_off_acc_ecc_vs)

        acc_vs_ecc.append(vs_acc['acc_ecc'])
        acc_vs_all.append(vs_acc['overall_acc'])

        EXCLUDE.append(vs_acc['exclude'])
        
        
        
        # check exclusion and save pp number
        if any(EXCLUDE)==True:
            print('need to exclude subject %s'%all_subs[ind])
            excluded_sub.append(all_subs[ind])
            ex_sub = 'True'
        else:
            ex_sub = 'False'

        # save relevant variables from above in descriptive tsv
        summary_df = pd.DataFrame({'sub':all_subs[ind],
                                    'nan_trials_pct': missed_trials[ind],
                                    'percent_fix_center':percent_fix_crwd[ind]*100,
                                    'accuracy_flankers_pct':acc_fl[ind]*100,
                                    'accuracy_noflankers_pct':acc_nofl[ind]*100,
                                    'crit_spacing_all':all_cs[ind],
                                    'crit_spacing_mean':mean_cs[ind],
                                    'accuracy_vs_all_pct':acc_vs_ecc[ind]*100,
                                    'accuracy_vs_mean_pct':acc_vs_all[ind]*100,
                                    'exclude':ex_sub
                                  })

        sum_dir = os.path.join(out_dir,'summary')
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)

        summary_df.to_csv(os.path.join(sum_dir,'summary_pp_'+all_subs[ind]+'.csv'), sep='\t')

    
    out_file = os.path.join(sum_dir,'sum_measures.npz')
    np.savez(out_file,
             all_subs = all_subs,
             excluded_sub = excluded_sub,
             all_tfr = all_tfr,
             mean_cs = mean_cs,
             acc_fl = acc_fl,
             acc_nofl = acc_nofl,
             all_cs = all_cs,
             acc_vs_ecc = acc_vs_ecc
             )
    
    return out_file
    

def crowded_trials(df_vs,rad_gab=1.1,screenHeight=30,screenDis=57, vRes = 1050,hRes=1680):
    
    # function to define if visual search trials are crowded or not
    # output binary array of trial size indicating if crowded trial or not
    
    #### INPUTS #####
    # df_vs - dataframe with behaviour data
    
    # output variable
    is_crowded = []
    
    # radius of each gabor
    radius_gabor = ang2pix(rad_gab,screenHeight,
                       screenDis,vRes)
    
    for trial in range(len(df_vs)): # for all trials
        
        # NOTE - all positions in pixels
        
        # set radius around target to compute density (ROI), which is 0.5 of the ecc of the target
        rad_roi = 0.5 * df_vs['target_ecc'].iloc[trial] 
        
        radius_roi = ang2pix(rad_roi,screenHeight,
                       screenDis,vRes)
        
        # get target and distractor positions as strings in list
        target_pos = df_vs['target_position'][trial].replace(']','').replace('[','').split(' ')
        distr_pos = df_vs['distractor_position'][trial].replace(']','').replace('[','').replace(',','').split(' ')

        # convert to list of floats
        target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])
        distr_pos = np.array([float(val) for i,val in enumerate(distr_pos) if len(val)>1])

        # save distractor positions in pairs (x,y)
        alldistr_pos = np.array([distr_pos[i*2:(i+1)*2] for i in range((len(distr_pos))//2)])
        
        # convert positions to make everything positive and hence my life easier
        # (changing coordinate axis to have center position in bottom left of display 
        # instead of center position in center of screen)

        new_target_pos = target_pos + [hRes/2,vRes/2]
        new_alldistr_pos = alldistr_pos + [hRes/2,vRes/2]
        
        append_trl = False
        
        for i in range(len(new_alldistr_pos)): # for all distractors
            # calculate distance between center of target and center of distractor
            distance = np.sqrt((new_target_pos[0]-new_alldistr_pos[i][0])**2+(new_target_pos[1]-new_alldistr_pos[i][1])**2)

            if distance < (radius_roi):#+radius_gabor): # circles intersecting (at least partially)
                append_trl = True # if at least one distractor is there, we label it as crowded
                
        is_crowded.append(append_trl)  
         
    return np.array(is_crowded)
       

# need new function that saves 
# mean RT per ecc for each set size 
# ends up being a 3x3 dataframe

def mean_RT_set_ecc_combo(data,sub,ecc=[4, 8, 12],setsize=[5,15,30]):
    # function to check RT as function of ecc for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # sub - sub number
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # RT_all - mean RT for all ecc
        
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of values with display set size
    target_set = data['set_size'].values
    
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    
    # list RT
    RT = data['RT'].values
        
    # dataframe to output values
    df_out = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
    # dataframe to count the number of trials for each condition, sanity check
    df_trial_num = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub']) 
    
    
    for _,s in enumerate(setsize): # for all set sizes 
        for _,e in enumerate(ecc): # for all target ecc

            RT_ecc_set = []
            num_trial_ecc_set = 0

            for t in range(len(data)): # for all actual trials 
                
                # if key press = target orientation and specific ecc and specific set size
                if (key_or[t]==target_or[t]) and (int(target_ecc[t])==e) and (int(target_set[t])==s):
 
                    if RT[t]> .250 and RT[t]<5 : # reasonable search times
                        RT_ecc_set.append(RT[t]) #append RT value
                        num_trial_ecc_set+=1 # increment trial counter

            if not RT_ecc_set: # if empty
                RT_ecc_set = float('Inf')

            # compute that mean RT and save in data frame           
            df_out = df_out.append({str(e)+'_ecc': np.nanmean(RT_ecc_set), 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
            # save number of trials
            df_trial_num = df_trial_num.append({str(e)+'_ecc': num_trial_ecc_set, 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
    # now reshape the dataframe (droping nans and making sure labels are ok)    
    df_out = df_out.apply(lambda x: pd.Series(x.dropna().values))
    df_out = df_out.dropna()
    
    for i in df_out.index:
        df_out.at[i, 'set_size'] = setsize[i]
        
    df_out = df_out.replace(np.inf,np.nan)  # replace any infinites with nan, makes averaging later easier 

    # same for trial number counter   
    df_trial_num = df_trial_num.apply(lambda x: pd.Series(x.dropna().values))
    df_trial_num = df_trial_num.dropna()
    
    for i in df_trial_num.index:
        df_trial_num.at[i, 'set_size'] = setsize[i]
    
    return df_out, df_trial_num



def mean_fix_set_ecc_combo(data,eyedata,sub,ecc=[4, 8, 12],
                           hRes=1680,vRes=1050,screenHeight=30,screenDis=57,
                           size_gab=2.2,setsize=[5,15,30]):
    # function to check mean number of fixations as function of ecc and set size for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # eyedata - eyetracking data
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # fix_all - mean fix for all ecc
        
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of values with display set size
    target_set = data['set_size'].values
    
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    
    # list RT
    RT = data['RT'].values
    
    # radius of gabor in pixels
    r_gabor = ang2pix(size_gab/2,screenHeight,
                       screenDis,
                       vRes)
    # number of samples in 150ms (we'll not count with fixations prior to 150ms after stim display)
    sample_thresh = 1000*0.150 # 1000Hz * time in seconds
            
    # dataframe to output values
    df_out = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
    # dataframe to count the number of trials for each condition, sanity check
    df_trial_num = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub']) 
    
    
    for _,s in enumerate(setsize): # for all set sizes 
        for _,e in enumerate(ecc): # for all target ecc

            fix_ecc_set = []
            num_trial_ecc_set = 0

            for t in range(len(data)): # for all actual trials 
                
                # if key press = target orientation and specific ecc and specific set size
                if (key_or[t]==target_or[t]) and (int(target_ecc[t])==e) and (int(target_set[t])==s): 
                    
                    # index for moment when display was shown
                    idx_display = np.where(np.array(eyedata[t]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                    # eye tracker sample time of display
                    smp_display = eyedata[t]['events']['msg'][idx_display][0]

                    # get target positions as strings in list
                    target_pos = data['target_position'][t].replace(']','').replace('[','').split(' ')
                    # convert to list of floats
                    target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])

                    num_fix = 0
                    for k,fix in enumerate(eyedata[t]['events']['Efix']):

                        # if fixations between 150ms after display and key press time
                        if (fix[0] > (smp_display+sample_thresh) and fix[0] < np.round(smp_display + RT[t]*1000)):

                            #if fixation not on target (not within target radius)
                            fix_x = fix[-2] - hRes/2
                            fix_y = fix[-1] - vRes/2; fix_y = - fix_y

                            if np.sqrt((fix_x-target_pos[0])**2+(fix_y-target_pos[1])**2) > r_gabor:
                                num_fix += 1
                                
                    fix_ecc_set.append(num_fix) #append number of fixations for that trial value                    
                    num_trial_ecc_set+=1 # increment trial counter
              
            if not fix_ecc_set: # if empty
                fix_ecc_set = float('Inf')

            # compute mean number of fixations and save in data frame           
            df_out = df_out.append({str(e)+'_ecc': np.nanmean(fix_ecc_set), 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
            # save number of trials
            df_trial_num = df_trial_num.append({str(e)+'_ecc': num_trial_ecc_set, 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
    # now reshape the dataframe (droping nans and making sure labels are ok)    
    df_out = df_out.apply(lambda x: pd.Series(x.dropna().values))
    df_out = df_out.dropna()
    
    for i in df_out.index:
        df_out.at[i, 'set_size'] = setsize[i]
        
    df_out = df_out.replace(np.inf,np.nan)  # replace any infinites with nan, makes averaging later easier 

    # same for trial number counter   
    df_trial_num = df_trial_num.apply(lambda x: pd.Series(x.dropna().values))
    df_trial_num = df_trial_num.dropna()
    
    for i in df_trial_num.index:
        df_trial_num.at[i, 'set_size'] = setsize[i]
    
    return df_out, df_trial_num


def df_all_trial_RT_fix(data,eyedata,sub,ecc=[4, 8, 12],
                           hRes=1680,vRes=1050,screenHeight=30,screenDis=57,
                           size_gab=2.2,setsize=[5,15,30]):
    # function to reunite RT and # fixations across trial for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # eyedata - eyetracking data
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # fix_all - mean fix for all ecc
    
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of values with display set size
    target_set = data['set_size'].values

    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values

    # list RT
    RT = data['RT'].values

    # radius of gabor in pixels
    r_gabor = ang2pix(size_gab/2,screenHeight,
                       screenDis,
                       vRes)
    # number of samples in 150ms (we'll not count with fixations prior to 150ms after stim display)
    sample_thresh = 1000*0.150 # 1000Hz * time in seconds

    # dataframe to output values
    df_out = pd.DataFrame(columns=['ecc','set_size','sub'])

    for _,s in enumerate(setsize): # for all set sizes 
        for _,e in enumerate(ecc): # for all target ecc

            fix_ecc_set = []
            RT_ecc_set = []
            num_trial_ecc_set = 0

            for t in range(len(data)): # for all actual trials 

                # if key press = target orientation and specific ecc and specific set size
                if (key_or[t]==target_or[t]) and (int(target_ecc[t])==e) and (int(target_set[t])==s): 

                    # index for moment when display was shown
                    idx_display = np.where(np.array(eyedata[t]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                    # eye tracker sample time of display
                    smp_display = eyedata[t]['events']['msg'][idx_display][0]

                    # get target positions as strings in list
                    target_pos = data['target_position'][t].replace(']','').replace('[','').split(' ')
                    # convert to list of floats
                    target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])

                    num_fix = 0
                    for k,fix in enumerate(eyedata[t]['events']['Efix']):

                        # if fixations between 150ms after display and key press time
                        if (fix[0] > (smp_display+sample_thresh) and fix[0] < np.round(smp_display + RT[t]*1000)):

                            #if fixation not on target (not within target radius)
                            fix_x = fix[-2] - hRes/2
                            fix_y = fix[-1] - vRes/2; fix_y = - fix_y

                            if np.sqrt((fix_x-target_pos[0])**2+(fix_y-target_pos[1])**2) > r_gabor:
                                num_fix += 1

                    fix_ecc_set.append(num_fix) #append number of fixations for that trial value  
                    RT_ecc_set.append(RT[t])
                    num_trial_ecc_set+=1 # increment trial counter

            if not fix_ecc_set: # if empty
                fix_ecc_set = float('Inf')

            # compute mean number of fixations and save in data frame           
            df_out = df_out.append({'fix': np.array(fix_ecc_set),
                                    'RT': np.array(RT_ecc_set),
                                    'ecc': e, 
                                    'set_size': s,
                                    'sub': sub},ignore_index=True)


    return df_out



def mean_ACC_set_ecc_combo(data,sub,ecc=[4, 8, 12],setsize=[5,15,30]):
    # function to check accuracy as function of ecc for visual search
    #
    # INPUTS #
    # data - df from behavioural csv, to get values for all trials
    # sub - sub number
    # ecc - list with eccs used in task
    #
    # OUTPUTS #
    # ACC_all - mean ACC for all ecc
        
    # list of values with target ecc
    target_ecc = data['target_ecc'].values
    # list of values with display set size
    target_set = data['set_size'].values
    
    # list of strings with the orientation of the target
    target_or = data['target_orientation'].values
    # list of strings with orientation indicated by key press
    key_or = data['key_pressed'].values
    
    # list RT
    RT = data['RT'].values
    
    # dataframe to output values
    df_out = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub'])
    # dataframe to count the number of trials for each condition, sanity check
    df_trial_num = pd.DataFrame(columns=[str(x)+'_ecc' for _,x in enumerate(ecc)]+['set_size','sub']) 
    
    
    for _,s in enumerate(setsize): # for all set sizes 
        for _,e in enumerate(ecc): # for all target ecc

            CORRECT_ecc_set = 0
            num_trial_ecc_set = 0

            for t in range(len(data)): # for all actual trials 
                
                # if specific ecc and specific set size
                if (int(target_ecc[t])==e) and (int(target_set[t])==s):
                    
                    num_trial_ecc_set+=1 # increment trial counter
                    
                    # if key press = target orientation 
                    if (key_or[t]==target_or[t]):

                        if RT[t]> .250 and RT[t]<5 : # reasonable search times
                            
                            CORRECT_ecc_set+=1 # count correct trials

            
            Accuracy = CORRECT_ecc_set/num_trial_ecc_set

            # compute that mean RT and save in data frame           
            df_out = df_out.append({str(e)+'_ecc': Accuracy, 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
            # save number of trials
            df_trial_num = df_trial_num.append({str(e)+'_ecc': num_trial_ecc_set, 
                                    'set_size': s, 
                                'sub': sub},ignore_index=True)
            
    # now reshape the dataframe (droping nans and making sure labels are ok)    
    df_out = df_out.apply(lambda x: pd.Series(x.dropna().values))
    df_out = df_out.dropna()
    
    for i in df_out.index:
        df_out.at[i, 'set_size'] = setsize[i]
        
    df_out = df_out.replace(np.inf,np.nan)  # replace any infinites with nan, makes averaging later easier 

    # same for trial number counter   
    df_trial_num = df_trial_num.apply(lambda x: pd.Series(x.dropna().values))
    df_trial_num = df_trial_num.dropna()
    
    for i in df_trial_num.index:
        df_trial_num.at[i, 'set_size'] = setsize[i]
    
    return df_out, df_trial_num


