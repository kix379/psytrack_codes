import pandas as pd
import scipy.io
import numpy as np
from helper import *

#To compare between different models
def mult_models(subjects,tasks,variables):
	info=pd.DataFrame()

	for subject_num in subjects:
		for task in tasks:
			for num_variables in variables:
				# Set save path for all figures, decide whether to save permanently
				SPATH = "C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Plots\\psytrack\\subject00"+str(subject_num)+"\\"+task+"\\"
				datapath = 'C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Data\\AttentionalLearningData\\subject00'+str(subject_num)

				ic_values, dat, wMode, weights=modelling(subject_num,task,num_variables,sigma,do_plots,do_cv,k,window,step,SPATH,datapath)
				info=info.append(ic_values)

	
#for performing modelling once
def single_model(subject_num,task,num_variables):
	SPATH = "C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Plots\\psytrack\\subject00"+str(subject_num)+"\\"+task+"\\"
	datapath = 'C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Data\\AttentionalLearningData\\subject00'+str(subject_num)

	ic_values, dat, wMode, weights=modelling(subject_num,task,num_variables,sigma,do_plots,do_cv,k,window,step,SPATH,datapath)
	print(ic_values)
	return ic_values, dat, wMode, weights




#global variables
sigma=5
do_plots=0

#cross validation
do_cv=0
k=5

#behavioral
window=30 #for dprime calc
step=50 #for dotted black lines

subject_num=1
task='freeform'
num_variables='4'
ic_values, dat, wMode, weights=single_model(subject_num,task,num_variables)

from psytrack.helper.helperFunctions import read_input

g = read_input(dat, weights)

gw=[]
for t in range(wMode[0].size):
	gw.append(g[t] @ wMode[:, t])

pL=1 / (1 + np.exp(gw)) #P(y=0)
y_pred=np.round(1-pL) #P(y=1)



#construct contingency table
datapath = 'C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Data\\AttentionalLearningData\\subject00'+str(subject_num)	
Data = scipy.io.loadmat(datapath + '\\sub_00'+str(subject_num)+'_'+task+'.mat')
CData = np.array(Data['CData'])
y_act=CData[:,11];
indices = np.where(y_act != 5)[0].tolist()
y_act=y_act[indices]
response_probe_side=CData[indices,10];
change_info=CData[indices,1];


print(g[:5])
print(wMode[:,:5])
print(gw[:5])
print(y_pred[:5])
print(y_act[:5])

print(y_pred.sort()==y_act.sort())
feedback_vals_act=[];
trial_num=[];
step=5;

for i in range(len(y_act)):
	if i>=window-1 and (i-window+1)%step==0:
		idx=i-window+1
		feedback_vals_act.append(computeFeedbackPeriodic(response_probe_side[idx:i+1],change_info[idx:i+1],y_act[idx:i+1]))
		trial_num.append(i-window/2);

feedback_vals_act=np.array(feedback_vals_act)
left_dprime_a=feedback_vals_act[:,0];
left_c_a=feedback_vals_act[:,1];
left_bcc_a=feedback_vals_act[:,2];
right_dprime_a=feedback_vals_act[:,3];
right_c_a=feedback_vals_act[:,4];
right_bcc_a=feedback_vals_act[:,5];

print(len(y_act),len(y_pred))

feedback_vals_pred=[];
for i in range(len(y_pred)):
	if i>=window-1 and (i-window+1)%step==0:
		idx=i-window+1
		feedback_vals_pred.append(computeFeedbackPeriodic(response_probe_side[idx:i+1],change_info[idx:i+1],y_pred[idx:i+1]))
				
feedback_vals_pred=np.array(feedback_vals_pred)
left_dprime_p=feedback_vals_pred[:,0];
left_c_p=feedback_vals_pred[:,1];
left_bcc_p=feedback_vals_pred[:,2];
right_dprime_p=feedback_vals_pred[:,3];
right_c_p=feedback_vals_pred[:,4];
right_bcc_p=feedback_vals_pred[:,5];

print(left_dprime_p)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

xlim=[0,800]
ylim=[-1.5,4.5]
ax1.plot(trial_num,left_dprime_p, color='red', marker='o', markersize=3, label='model') #probed_Ch_L
ax1.plot(trial_num,left_dprime_a, color='blue',  marker='o', markersize=3, label='data') #probed_Ch_R

ax1.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
ax1.set(ylabel='left side d\'', xticks=50*np.arange(0,xlim[1]/50 + 1), yticks=np.arange(ylim[0],ylim[1],0.5), xlim=xlim)
ax1.legend(loc ="upper right")

ax2.plot(trial_num,right_dprime_p, color='red', marker='o', markersize=3, label='model') 
ax2.plot(trial_num,right_dprime_a, color='blue', marker='o', markersize=3, label='data')

ax2.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
ax2.set(xlabel='trial', ylabel='right side d\'', xticks=50*np.arange(0,xlim[1]/50 + 1), yticks=np.arange(ylim[0],ylim[1],0.5), xlim=xlim)
ax2.legend(loc ="upper right")

plt.show()
"""



