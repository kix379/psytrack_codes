import os
#import re
#from IPython.display import clear_output
import numpy as np
import pandas as pd
import scipy.io
from statistics import NormalDist
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
	
	return values, outData, wMode, weights
	

def computeFeedbackPeriodic(resp_probe_side,change_info,response_key):

	left_probe_contingency =[[0,0],[0,0]];
	right_probe_contingency = [[0,0],[0,0]];

	for i in range(len(response_key)):
		if resp_probe_side[i]==-1: 		#probe left
			if change_info[i]==-1 or change_info[i]==2: 	#change left or both
				if response_key[i]==1: 		#response yes, left probe hit
					left_probe_contingency[0][0]=left_probe_contingency[0][0]+1;
				elif response_key[i]==0:		#response no, left probe miss
					left_probe_contingency[0][1]=left_probe_contingency[0][1]+1;
			
			if change_info[i]==1 or change_info[i]==0:		#change right or no change
				if response_key[i]==1: 		#response yes, left probe false alarm
					left_probe_contingency[1][0]=left_probe_contingency[1][0]+1;
				elif response_key[i]==0: 		#response no, left probe correct rejection
					left_probe_contingency[1][1]=left_probe_contingency[1][1]+1;

		elif resp_probe_side[i]==1: 	#probe right
			if change_info[i]==1 or change_info[i]==2: 	#change right or both
				if response_key[i]==1: 		#response yes, right probe hit
					right_probe_contingency[0][0]=right_probe_contingency[0][0]+1;
				elif response_key[i]==0: 	#response no, right probe miss
					right_probe_contingency[0][1]=right_probe_contingency[0][1]+1;
			if change_info[i]==-1 or change_info[i]==0:
				if response_key[i]==1:		#response yes, right probe false alarm
					right_probe_contingency[1][0]=right_probe_contingency[1][0]+1;
				elif response_key[i]==0: 	#response no, right probe correct rejection
					right_probe_contingency[1][1]=right_probe_contingency[1][1]+1;
	
	to_replace=0.5;
	for i in range(len(left_probe_contingency)):
		for j in range(len(left_probe_contingency[0])):
			if left_probe_contingency[i][j]==0:
				left_probe_contingency[i][j]=to_replace;
			if right_probe_contingency[i][j]==0:
				right_probe_contingency[i][j]=to_replace;

	print(left_probe_contingency)
	lp_hit_rate=left_probe_contingency[0][0]/(left_probe_contingency[0][0]+left_probe_contingency[0][1]);
	lp_false_alarm_rate=left_probe_contingency[1][0]/(left_probe_contingency[1][0]+left_probe_contingency[1][1]);
	rp_hit_rate=right_probe_contingency[0][0]/(right_probe_contingency[0][0]+right_probe_contingency[0][1]);
	rp_false_alarm_rate=right_probe_contingency[1][0]/(right_probe_contingency[1][0]+right_probe_contingency[1][1]);
	
	a=NormalDist().inv_cdf(lp_false_alarm_rate)
	b=NormalDist().inv_cdf(lp_hit_rate)

	lp_d = b-a;
	lp_c = -a;
	lp_bcc = lp_c - (lp_d/2);

	a=NormalDist().inv_cdf(rp_false_alarm_rate)
	b=NormalDist().inv_cdf(rp_hit_rate)

	rp_d = b-a;
	rp_c = -a;
	rp_bcc = lp_c - (lp_d/2);

	return lp_d,lp_c,lp_bcc,rp_d,rp_c,rp_bcc		