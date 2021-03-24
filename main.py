import os
#import re
#from IPython.display import clear_output
import numpy as np
import pandas as pd
import scipy.io
import psytrack as psy

from construct_input import *
from plotting_functions import *
from helper import *


################### Modifiable variables ####################
subject_num=1
task='freeform'
num_variables='6'
sigma=5

#cross validation
do_cv=0
k=5

#behavioral
plot_behavior=1
window=50 #for dprime calc
step=100#for dotted black lines

# Set save path for all figures, decide whether to save permanently
SPATH = "C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Plots\\psytrack\\subject00"+str(subject_num)+"\\"+task+"\\"
datapath = 'C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Data\\AttentionalLearningData\\subject00'+str(subject_num)


#################### Non Modifiable ####################

#Load data
Data = scipy.io.loadmat(datapath + '\\sub_00'+str(subject_num)+'_'+task+'.mat')
if not os.path.exists(SPATH):
   	os.makedirs(SPATH)
	
sub_data=getSubjectData(Data,num_variables)

outData, weights = getData(sub_data,num_variables)
K = np.sum([weights[i] for i in weights.keys()])
print(outData, weights)

hyper_guess = {
 'sigma'   : [2**sigma]*K,
 'sigInit' : 2**5,
 'sigDay'  : None
  }
optList = ['sigma']

#main function to get the weights
hyp, evd, wMode, hess_info = psy.hyperOpt(outData, hyper_guess, weights, optList, showOpt=1)
W_std=hess_info['W_std']

xlim_val=(int(wMode.shape[1]/100)+1)*100
ylim_val=np.max(np.abs(wMode))
if ylim_val>5:
	ylim_val=30
else:
	ylim_val=5	

#plotting the weights
img_filename=task+'_'+num_variables+'_input_sigma_'+str(sigma)
img_title='Subject '+str(subject_num)+' '+task
standard_plot(wMode, W_std, weights,[0,xlim_val], [-ylim_val,ylim_val], SPATH+img_filename, img_title)
if num_variables=='4' or num_variables=='6':
	sep_plot(wMode, W_std, weights,[0,xlim_val], [-ylim_val,ylim_val], SPATH+img_filename+'_sep', img_title)


if do_cv==1:
	num=wMode.shape[1]-wMode.shape[1]%k
	perform_cross_validation(outData,num,hyper_guess,weights,optList,k,num_variables,SPATH+img_filename+'_trimmed', img_title+ ' ' +str(num)+' trials',xlim_val,ylim_val)

if plot_behavior==1:
	feedback_data = scipy.io.loadmat(datapath + '\\'+task+'\\feedback_vals_window_'+str(window)+'.mat')
	sorted_weights=sorted(weights)
	indices={
		'probe_L' : sorted_weights.index('probe_L'),
		'probe_R' : sorted_weights.index('probe_R'),
		'probe_Ch_L' : sorted_weights.index('probed_Ch_L'),
		'probe_Ch_R' : sorted_weights.index('probed_Ch_R')
	}	
	to_plot_behavior(feedback_data,indices,xlim_val,ylim_val,wMode,weights,step,SPATH,img_filename,window,img_title)
