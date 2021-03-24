import os
#import re
#from IPython.display import clear_output
import numpy as np
import pandas as pd
import scipy.io
import psytrack as psy
from plotting_functions import *
from construct_input import *


def perform_cross_validation(outData,length,hyper_guess,weights,optList,k,num_variables,img_filename,title,xlim_val,ylim_val):
	#trim the data if you're performing cross validation
	new_D = psy.trim(outData, END=length)
	hyp, evd, wMode, hess_info = psy.hyperOpt(new_D, hyper_guess, weights, optList)

	#cross validation 
	xval_logli, xval_pL = psy.crossValidate(new_D, hyper_guess, weights, optList, F=k, seed=41)

	W_std=hess_info['W_std']

	#plotting the weights
	if num_variables=='4' or num_variables=='6':
		sep_plot(wMode, W_std, weights,[0,xlim_val], [-ylim_val,ylim_val], img_filename+'.png',title)
	else:
		standard_plot(wMode, W_std, weights,[0,xlim_val], [-ylim_val,ylim_val], img_filename+'.png',title)
	fig_perf_xval = psy.plot_performance(new_D, xval_pL=xval_pL)
	fig_bias_xval = psy.plot_bias(new_D, xval_pL=xval_pL)

	fig_perf_xval.savefig(img_filename+'_'+str(k)+'fold_performance.png')
	fig_bias_xval.savefig(img_filename+'_'+str(k)+'fold_bias.png')
	return xval_logli

def compute_ic(evd,hyp,wMode,n):
	T=wMode[0].size     	#number of trials
	E=evd 					#log-evidence or log-likelihood
	K=len(hyp['sigma']) 	#number of hyperparameters
	print('T=%d, E=%f, K=%d' % (T,E,K))
	aic= (-2/T * E) + (2 * K/T)
	bic= (-2*E) + (np.log(T) * K)
	return round(aic,n), round(bic,n)


def modelling(subject_num,task,num_variables,sigma,do_plots,do_cv,k,window,step,SPATH,datapath):
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

	aic,bic=compute_ic(evd,hyp,wMode,2)
	evd=round(evd,2)
	xlim_val=(int(wMode.shape[1]/100)+1)*100
	ylim_val=np.max(np.abs(wMode))
	if ylim_val>5:
		ylim_val=30
	else:
		ylim_val=5	

	#plotting the weights
	img_filename=task+'_'+num_variables+'_input_sigma_'+str(sigma)
	img_title='Subject '+str(subject_num)+' '+task +'; Log Evidence='+str(evd)+' ; AIC='+str(aic)+' ; BIC='+str(bic)
	if do_plots==1:
		standard_plot(wMode, W_std, weights,[0,xlim_val], [-ylim_val,ylim_val], SPATH+img_filename, img_title)
	
		if num_variables=='4' or num_variables=='6':
			sep_plot(wMode, W_std, weights,[0,xlim_val], [-ylim_val,ylim_val], SPATH+img_filename+'_sep', img_title)


		sorted_weights=sorted(weights)
		
		if set(['probe_L','probe_R','probed_Ch_L','probed_Ch_R']).issubset(sorted_weights):
			print('plotting')
			feedback_data = scipy.io.loadmat(datapath + '\\'+task+'\\feedback_vals_window_'+str(window)+'.mat')
	
			indices={
				'probe_L' : sorted_weights.index('probe_L'),
				'probe_R' : sorted_weights.index('probe_R'),
				'probed_Ch_L' : sorted_weights.index('probed_Ch_L'),
				'probed_Ch_R' : sorted_weights.index('probed_Ch_R')
			}	
			to_plot_behavior(feedback_data,indices,xlim_val,ylim_val,wMode,weights,step,SPATH,img_filename,window,img_title)
	
	xval_logli=np.nan
	if do_cv==1:
		num=wMode.shape[1]-wMode.shape[1]%k
		xval_logli=perform_cross_validation(outData,num,hyper_guess,weights,optList,k,num_variables,SPATH+img_filename+'_trimmed', img_title+ ' ' +str(num)+' trials',xlim_val,ylim_val)
		print('Cross validated log likelihood='+str(xval_logli))
	
	values={
		'subject': subject_num,
		'task' : task,
		'input_type' : num_variables,
		'log-evidence' : evd,
		'AIC' : aic,
		'BIC' : bic,
		'Cross validated log-likelihood' : xval_logli
	}
	
	return values
	