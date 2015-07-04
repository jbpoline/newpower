# load libraries

import os
from nipy.labs.utils.simul_multisubject_fmri_dataset import surrogate_3d_dataset
import numpy as np
import nibabel as nib
import pylab as pl
import math
from nipype.interfaces import fsl
from nipype.interfaces.fsl import L2Model
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.optimize

TEMPDIR = "/Users/Joke/Documents/Onderzoek/Studie_7_newpower/newpower/simulations/"
os.chdir(TEMPDIR)

positions = np.array([[40,40,40],
					 [40,70,40],
					 [70,70,70]])
amplitude=np.array([4,5,2])
mask = nib.load("mask.nii.gz")
smooth_FWHM = 3
smooth_sd = smooth_FWHM/(2*math.sqrt(2*math.log(2)))
data = surrogate_3d_dataset(n_subj=10,sk=smooth_sd,shape=mask.get_shape(),mask=mask,noise_level=1,pos=positions,ampli=amplitude,width=7)
data = (data.transpose([1,2,3,0])*1000)
data=data.astype(np.int16)
img=nib.Nifti1Image(data,np.eye(4))
img.to_filename("simulation1.nii.gz")

model=L2Model(num_copes=10)
model.run()

flameo=fsl.FLAMEO(cope_file='simulation1.nii.gz', \
				 cov_split_file='design.grp', \
				 design_file='design.mat', \
				 t_con_file='design.con', \
				 mask_file='mask.nii.gz', \
				 run_mode='ols')
flameo.run()


cl=fsl.model.Cluster()
cl.inputs.threshold=2.3
cl.inputs.in_file="stats/tstat1.nii.gz"
cl.inputs.out_localmax_txt_file="locmax.txt"
cl.run()
