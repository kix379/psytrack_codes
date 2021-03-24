import pandas as pd
from helper import *

#subject_num=2
#task='freeform'
#num_variables='prevresp'
sigma=5
do_plots=0

#cross validation
do_cv=0
k=5

#behavioral
window=30 #for dprime calc
step=50 #for dotted black lines

info=pd.DataFrame()

for subject_num in [1,2]:
	for task in ['eqfb','freeform']:
		for num_variables in ['4','prevresp']:
			# Set save path for all figures, decide whether to save permanently
			SPATH = "C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Plots\\psytrack\\subject00"+str(subject_num)+"\\"+task+"\\"
			datapath = 'C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Data\\AttentionalLearningData\\subject00'+str(subject_num)

			ic_values=modelling(subject_num,task,num_variables,sigma,do_plots,do_cv,k,window,step,SPATH,datapath)
			info=info.append(ic_values)


