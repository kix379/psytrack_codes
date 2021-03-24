from matplotlib import pyplot as plt
import numpy as np
import scipy.io
from matplotlib.patches import ConnectionPatch

def standard_plot(wMode, W_std, weights, xlim, ylim, filename,title):
	plt.figure(figsize=(8,4))
	ax = plt.gca()
	for i in range(wMode.shape[0]):
		color = next(ax._get_lines.prop_cycler)['color']
		plt.plot(wMode[i], lw=1, linestyle='-', zorder=2*i+1, color=color)
		#plt.fill_between(np.arange(len(wMode[i])), wMode[i] - 2 * W_std[i], wMode[i] + 2 * W_std[i], facecolor=color, alpha=0.2, zorder=2*i)
	plt.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
	plt.xticks(50*np.arange(0,xlim[1]/50 + 1))
	plt.yticks(np.arange(-2,3,2))
	plt.xlim(xlim[0],xlim[1])
	plt.ylim(ylim[0],ylim[1])
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['top'].set_visible(False)

	plt.xlabel("trials")
	plt.ylabel("weights")
	legendNames=  sorted(weights)
	plt.legend(legendNames, loc ="lower right")
	plt.title(title) 
	plt.savefig(filename+'.png')
	#plt.show()

def sep_plot(wMode, W_std, weights, xlim, ylim, filename, title):
	
	legendNames=  sorted(weights)
	plt.figure(figsize=(8,12))
	ax = plt.gca()
	for j in range(1,int(wMode.shape[0]/2)+1):
		plt.subplot(int(wMode.shape[0]/2),1,j)
		first=2*(j-1)
		for i in [first,first+1]:
			color = next(ax._get_lines.prop_cycler)['color']
			plt.plot(wMode[i], c=color, lw=1, linestyle='-', alpha=0.85, zorder=2*i+1, label=legendNames[i])
			#plt.fill_between(np.arange(len(wMode[i])), wMode[i] - 2 * W_std[i], wMode[i] + 2 * W_std[i],
            #         facecolor=color, alpha=0.2, zorder=2*i)
			plt.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
			plt.xticks(50*np.arange(0,xlim[1]/50 + 1))
			plt.yticks(np.arange(-2,3,2))
			plt.xlim(xlim[0],xlim[1])
			plt.ylim(ylim[0],ylim[1])
			plt.gca().spines['right'].set_visible(False)
			plt.gca().spines['top'].set_visible(False)
			plt.xlabel("trials")
			plt.ylabel("weights")
			plt.legend(loc ="lower right") 

	plt.suptitle(title)
	plt.savefig(filename+'.png')
	#plt.show()

def to_plot_behavior(feedback_data,indices,xlim_val,ylim_val,wMode,weights,step,SPATH,img_filename,window,img_title):
	#plot with dprime, bcc, criterion
	feedback_vals = np.array(feedback_data['feedback_vals']);
	left_dprime=feedback_vals[:,0];
	left_c=feedback_vals[:,1];
	left_bcc=feedback_vals[:,2];
	right_dprime=feedback_vals[:,3];
	right_c=feedback_vals[:,4];
	right_bcc=feedback_vals[:,5];
	trial_num=np.array(feedback_data['trial_num']);


	params_1={
		'line_1_index':indices['probed_Ch_L'],
		'line_2_index':indices['probed_Ch_R'],
		'xlim': [0,xlim_val],
		'ylim': [-ylim_val,ylim_val]
	}

	params_2={
		'line_1':left_dprime,
		'line_2':right_dprime,
		'label_1':'dprime_left',
		'label_2':'dprime_right',
		'x-axis':trial_num,
		'ylim': [-1.5,4.5],
		'ylabel': 'd\''
	}

	if 'prev_resp' in weights.keys():
		plot_weight_behavior_prevresponse(wMode,weights,params_1,params_2,window,step,SPATH+img_filename+'_dprime_probed_ch_window_'+str(window)+'.png',img_title)
	else:
		plot_weights_behavior(wMode,weights,params_1,params_2,window,step,SPATH+img_filename+'_dprime_probed_ch_window_'+str(window)+'.png',img_title)
	
	params_1={
		'line_1_index':indices['probe_L'],
		'line_2_index':indices['probe_R'],
		'xlim': [0,xlim_val],
		'ylim': [-ylim_val,ylim_val]
	}
	"""
	params_2={
		'line_1':left_bcc,
		'line_2':right_bcc,
		'label_1':'bcc_left',
		'label_2':'bcc_right',
		'x-axis':trial_num,
		'ylim': [-2,2.5],
		'ylabel': 'bcc'
	}
	plot_weights_behavior(wMode,weights,params_1,params_2,window,step,SPATH+img_filename+'_bcc_probe_window_'+str(window)+'.png',img_title)
	"""
	#for criterion, take negative of the weights in order to be able to compare
	params_2={
		'line_1':left_c,
		'line_2':right_c,
		'label_1':'criterion_left',
		'label_2':'criterion_right',
		'x-axis':trial_num,
		'ylim': [-2,2.5],
		'ylabel': 'criterion'
	}
	wMode[params_1['line_1_index']]=np.negative(wMode[params_1['line_1_index']])
	wMode[params_1['line_2_index']]=np.negative(wMode[params_1['line_2_index']])

	if 'prev_resp' in weights.keys():
		plot_weight_behavior_prevresponse(wMode,weights,params_1,params_2,window,step,SPATH+img_filename+'_criterion_window_'+str(window)+'.png',img_title)
	else:
		plot_weights_behavior(wMode,weights,params_1,params_2,window,step,SPATH+img_filename+'_criterion_window_'+str(window)+'.png',img_title)


#plot dprime and probed_ch
def plot_weights_behavior(wMode,weights,params_1,params_2,window,step,img_filename,title):
	fig = plt.figure(figsize=(10,5))
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	legendNames=  sorted(weights)

	idx_1=params_1['line_1_index']
	idx_2=params_1['line_2_index']
	xlim=params_1['xlim']
	ylim=params_1['ylim']

	ax1.plot(wMode[idx_1], color='red', label=legendNames[idx_1]) #probed_Ch_L
	ax1.plot(wMode[idx_2], color='blue',  label=legendNames[idx_2]) #probed_Ch_R

	ax1.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)

	ax1.set(ylabel='weight', xticks=50*np.arange(0,xlim[1]/50 + 1), yticks=np.arange(-2,3,2), xlim=xlim,ylim=ylim)
	ax1.legend(loc ="upper right")

	
	ylim=params_2['ylim']

	ax2.plot(params_2['x-axis'], params_2['line_1'], color='red', marker='o', markersize=3, label=params_2['label_1']) 
	ax2.plot(params_2['x-axis'], params_2['line_2'], color='blue', marker='o', markersize=3, label=params_2['label_2'])

	ax2.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
	ax2.set(xlabel='trial', ylabel=params_2['ylabel'], xticks=50*np.arange(0,xlim[1]/50 + 1), yticks=np.arange(ylim[0],ylim[1],0.5), xlim=xlim)
	ax2.legend(loc ="upper right")

	for i in np.arange(step,xlim[1],step):
		con = ConnectionPatch(xyA=(i,params_2['ylim'][0]), xyB=(i,params_1['ylim'][1]), coordsA="data", coordsB="data",axesA=ax2, axesB=ax1, color="black", linestyle='--', alpha=0.5)
		ax2.add_artist(con)
	fig.suptitle(title+', Window='+str(window))
	fig.savefig(img_filename)    


def plot_weight_behavior_prevresponse(wMode,weights,params_1,params_2,window,step,img_filename,title):
	fig = plt.figure(figsize=(10,6))
	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	legendNames=  sorted(weights)
	
	idx_1=params_1['line_1_index']
	idx_2=params_1['line_2_index']
	xlim=params_1['xlim']
	ylim=params_1['ylim']

	ax1.plot(wMode[idx_1], color='red', label=legendNames[idx_1]) #probed_Ch_L
	ax1.plot(wMode[idx_2], color='blue',  label=legendNames[idx_2]) #probed_Ch_R

	ax1.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)

	ax1.set(ylabel='weight', xticks=50*np.arange(0,xlim[1]/50 + 1), yticks=np.arange(-2,3,2), xlim=xlim,ylim=ylim)
	ax1.legend(loc ="upper right")

	
	ylim=params_2['ylim']

	ax2.plot(params_2['x-axis'], params_2['line_1'], color='red', marker='o', markersize=3, label=params_2['label_1']) 
	ax2.plot(params_2['x-axis'], params_2['line_2'], color='blue', marker='o', markersize=3, label=params_2['label_2'])

	ax2.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
	ax2.set(ylabel=params_2['ylabel'], xticks=50*np.arange(0,xlim[1]/50 + 1), yticks=np.arange(ylim[0],ylim[1],0.5), xlim=xlim)
	ax2.legend(loc ="upper right")

	idx_3=legendNames.index('prev_resp')
	ax3.plot(wMode[idx_3], color='black', label=legendNames[idx_3])

	ax3.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
	ax3.set(xlabel='trial', ylabel='weight', xticks=50*np.arange(0,xlim[1]/50 + 1), yticks=np.arange(-2,3,2), xlim=params_1['xlim'], ylim=params_1['ylim'])
	ax3.legend(loc ="upper right")

	for i in np.arange(step,xlim[1],step):
		con = ConnectionPatch(xyA=(i,params_1['ylim'][0]), xyB=(i,params_1['ylim'][1]), coordsA="data", coordsB="data",axesA=ax3, axesB=ax1, color="black", linestyle='--', alpha=0.5)
		ax3.add_artist(con)
	fig.suptitle(title+', Window='+str(window))
	
	fig.savefig(img_filename)  


