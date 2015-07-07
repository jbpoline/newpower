# load libraries

import os
import numpy as np
import nibabel as nib
import pylab as pl
import math
from nipype.interfaces import fsl
from nipype.interfaces.fsl import L2Model
from nipype.interfaces.fsl import Threshold
from nipy.labs.utils.simul_multisubject_fmri_dataset import surrogate_3d_dataset
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.optimize
from palettable.colorbrewer.qualitative import Paired_9
import matplotlib 

# set folders
HOMEDIR = "/Users/Joke/Documents/Onderzoek/Studie_7_newpower/newpower/methods/"
execfile(os.path.join(HOMEDIR,'BUM.py'))
execfile(os.path.join(HOMEDIR,'neuropower.py'))
WORKDIR = "/Users/Joke/Documents/Onderzoek/Studie_7_newpower/WORKDIR/"
os.chdir(WORKDIR)

sims=50
powerovarr=np.empty((sims,45),dtype=np.object)
powerunarr=np.empty((sims,45),dtype=np.object)
powercorarr = np.empty((sims,45),dtype=np.object)
powerestarr = np.empty((sims,45),dtype=np.object)

sim=1
	# set temporary directory (and make)
	TEMPDIR = os.path.join(WORKDIR,"sim_"+str(sim)+"/")
	os.mkdir(TEMPDIR)
	os.chdir(TEMPDIR)
	
	# simulate dataset with 10 subjects
	positions = np.array([[40,40,40],
						 [40,70,40],
						 [70,70,70]])
	amplitude=np.array([4,5,2])
	mask = nib.load(os.path.join(WORKDIR,"mask.nii.gz"))
	smooth_FWHM = 3
	smooth_sd = smooth_FWHM/(2*math.sqrt(2*math.log(2)))
	data = surrogate_3d_dataset(n_subj=10,sk=smooth_sd,shape=mask.get_shape(),mask=mask,noise_level=1,pos=positions,ampli=amplitude,width=7)
	data = (data.transpose([1,2,3,0])*1000)
	data=data.astype(np.int16)
	img=nib.Nifti1Image(data,np.eye(4))
	img.to_filename("simulation1.nii.gz")
	nonoise = surrogate_3d_dataset(n_subj=1,sk=smooth_sd,shape=mask.get_shape(),mask=mask,noise_level=0,pos=positions,ampli=amplitude,width=7)
	nonoise = nonoise*1000
	nonoise=nonoise.astype(np.int16)
	img=nib.Nifti1Image(nonoise,np.eye(4))
	img.to_filename("activation.nii.gz")
	
	# perform group analysis
	model=L2Model(num_copes=10)
	model.run()
	
	flameo=fsl.FLAMEO(cope_file='simulation1.nii.gz', \
					 cov_split_file='design.grp', \
					 design_file='design.mat', \
					 t_con_file='design.con', \
					 mask_file=os.path.join(WORKDIR,'mask.nii.gz'), \
					 run_mode='ols')
	flameo.run()
	
	# extract peaks
	exc = 1.5
	cl=fsl.model.Cluster()
	cl.inputs.threshold=exc
	cl.inputs.in_file="stats/tstat1.nii.gz"
	cl.inputs.out_localmax_txt_file="locmax.txt"
	cl.inputs.num_maxima=2000
	cl.inputs.connectivity=26
	cl.run()
	
	# read in peak file
	peaks = pd.read_csv(os.path.join(TEMPDIR,"locmax.txt"),sep="\t").drop('Unnamed: 5',1)
	activation = nib.load("activation.nii.gz").get_data()
	actlist = []
	
	# search in activation map whether peak is in active area
	for p in range(0,len(peaks.Value)):
		actlist.append(activation[91-peaks.x[p],peaks.y[p],peaks.z[p]]>0)
	
	peaks['active']=actlist	
	Ptotal = sum(actlist)
	
	# compute power over all peaks, or power only above threshold
	thresrange = np.arange(1.5,6,0.1)
	powerover = []
	powerunder=[]
	powercorr=[]
	powerest=[]
	for exc in thresrange:
		newpeaks = peaks[peaks['Value']>exc]
		newpeaks['p-values'] = 1-nulcumdens(exc,newpeaks['Value'])
		Bumres = bumOptim(newpeaks['p-values'],starts=100)
		NPres = npowerOptim(newpeaks['Value'],Bumres.pi1,exc)
		x = np.arange(exc,15,0.00001)
		y = 1-nulcumdens(exc,x)
		cutoff = min(x[y<0.001])
		powest = (1-altcumdens(NPres.mu,NPres.sigma,exc,cutoff))
		powerest.append(powest)
		
		newpeaks['significant'] = newpeaks['p-values']<0.001		
		sig = np.asarray(newpeaks.significant)
		ac = np.asarray(newpeaks.active)
		TP = sum([all(tuple) for tuple in zip(sig,ac)])
		P = sum(newpeaks.active.tolist())
		S = sum(newpeaks.significant.tolist())
		powerover.append(float(TP)/float(P))
		powerunder.append(float(TP)/float(Ptotal))
		
		allpvals = 1-nulcumdens(1.5,peaks['Value'])
		Bumresall = bumOptim(allpvals,starts=100)
		totalactive=Bumresall.pi1*len(allpvals)
		foundactive=Bumres.pi1*len(newpeaks.Value)
		powercorr.append(powest*foundactive/totalactive)
		#Pcorr = Bumres.pi1*len(miss['p-values'])
		#powercorr.append(float(TP)/(float(P)+Pcorr))
	
	powerovarr[sim,:]=powerover
	powerunarr[sim,:]=powerunder
	powercorarr[sim,:]=powercorr
	powerestarr[sim,:]=powerest


np.savetxt("poweroverarr.txt",powerovarr)
np.savetxt("powerunarr.txt",powerunarr)
np.savetxt("powercorarr.txt",powercorarr)
np.savetxt("powerestarr.txt",powerestarr)


powerover = np.mean(powerovarr,0)
powerunder = np.mean(powerunarr,0)
powercorr = np.mean(powercorarr,0)
powerest = np.mean(powerestarr,0)

cols = Paired_9.mpl_colors

plt.figure(figsize=(14,12))
ax=plt.subplot(111)
ax.spines["top"].set_visible(False)  
ax.spines["bottom"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.spines["left"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()
plt.ylim(0, 1)  
plt.xlim(1.5, 6)  

for y in np.arange(0, 1.1,0.1):  
    plt.plot(range(2, 7), [y]*len(range(2,7)), "--", lw=0.5, color="black", alpha=0.3) 

plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                labelbottom="on", left="off", right="off", labelleft="on")  

plt.xlabel("Clusterforming threshold")
plt.ylabel("Power")
plt.title("Influence of clusterforming threshold on power \n (average over 50 simulations)")
plt.plot(thresrange,powerover,color=cols[1],linewidth=2,label="power>u")
plt.plot(thresrange,powerunder,color=cols[3],linewidth=2,label="power")
#plt.plot(thresrange,powerest,color=cols[0],linewidth=2,label="hat(power>u)")
#plt.plot(thresrange,powercorr,color=cols[2],linewidth=2,label="hat(power)")

plt.legend(loc="lower left",frameon=False,numpoints=1)
plt.show()

plt.savefig(os.path.join(WORKDIR,"power.png"))


#### plot example fit model

bins,edges=np.histogram(peaks['Value'],density=True,bins=50)
plt.hist(peaks.Value,50,normed=True)
x=np.arange(exc,40,0.001)
y0=nulprobdens(exc,x)
yA=altprobdens(NPres.mu,NPres.sigma,exc,x)
yT=mixprobdens(NPres.mu,NPres.sigma,Bumres.pi1,exc,x)
plt.plot(x,(1-Bumres.pi1)*y0)
plt.plot(x,Bumres.pi1*yA)
plt.plot(x,yT)
plt.xlim((exc,16))
plt.ylim((0,0.8))
plt.show()












