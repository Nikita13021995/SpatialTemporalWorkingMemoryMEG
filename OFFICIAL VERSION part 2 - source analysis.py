# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:29:37 2022

@author: Nikita O
Project: Spatial and Temporal WM
"""
# %% LIBRARIES

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import sklearn
import h5io
import h5py
import conpy
import nilearn

import nibabel
import vtk
import pyvista
import mayavi
import joblib
import dipy
import mne_connectivity

from   mne.stats import (spatio_temporal_cluster_1samp_test,
                         summarize_clusters_stc, 
                         spatio_temporal_cluster_test,
                         permutation_cluster_test)
from   mne.minimum_norm import apply_inverse, read_inverse_operator
from   mne import read_source_estimate
from   mne.minimum_norm import read_inverse_operator
from   mne.cov import compute_covariance
from   mne.beamformer import (make_dics, apply_dics_csd, make_lcmv,
                            apply_lcmv_cov)
from   mne.minimum_norm import (make_inverse_operator, apply_inverse_cov)
from   mne.transforms import apply_trans

from   scipy import linalg
from   scipy import stats as st
from   scipy import stats as stats

from   mne.io.constants import FIFF
from   mne.io import read_info
from   mne.datasets import fetch_fsaverage

%matplotlib qt

# %% PREPARATION FOR CALCULATION

### SUPPLEMENTARY ARRAYS
index_array            = [3,4,6,7,9,10, 11,14, 18, 19, 20, 21,23,25,26,27,28] 
data_path              = 'L:/SATURN/ABRAR WM/fif for coreg _ forward modeling/subjects'
trans_path             = 'L:/'
raw_list               = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,134,15,16,17,
                          17,8,56,234,234,346,4576,56,123,12,       12] 
sub_list               = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,134,15,16,17,
                          17,8,56,234,234,346,4576,56,123,12,       12]
index_array_2          = [3,6,7,9,10,11,14,18,19,20,21,23,25,26,27,28] 
index_array            = [3,6,7,9,10,11,14,18,19,20,21,23,25,26,27,28] 
SDF                    = 'L:/SATURN/ABRAR WM/JAN' #Firstly we put in a DIRECT way a directory with the file
data_path              = 'L:/SATURN/ABRAR WM/fif for coreg _ forward modeling/subjects'
f_avg_path             = 'L:/SATURN/ABRAR WM/fif for coreg _ forward modeling/subjects'
trans_path             = 'L:/SATURN/ABRAR WM/'

### VISUALIZATION OPTIONS
#Number of multi-samples. Should be 1 for MESA for volumetric rendering to work properly.
mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)   #This is the piece of code, which makes the MNE plotting working. 
                                                                                #Otherwise: do not call mne, use directly pyvista
                                                                                #https://github.com/mne-tools/mne-python/issues/10890

### LIST OF SUBJECT   
index                  = 3
for index in index_array[:]:    
    Subject            = 'Sub{}'.format(index)
    sub_list[index]    = Subject
    
    
# %% FREESURFER OUTPUT

### MRI CHECKER
index                  = 10
watch_path             = 'L:/SATURN/ABRAR WM/fif for coreg _ forward modeling/subjects/Sub{}/mri'.format(index)
t1_fname               = os.path.join(watch_path, 'T1.mgz')
t1                     = nibabel.load(t1_fname)
t1.orthoview()

### RAW FILE UPLOADER
index                  = 10
for index in index_array[:]:    
    SDRF               = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index))
    raw_index          = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False,
                                             on_split_missing='raise', verbose=None)
    raw_list[index]    = raw_index                

### CHECKING THE ALIGHMENT AFTER FREESURFER
index                  = 10
for index in index_array_2[:1]:  
    os.chdir('C:\Users\Nikita O\Desktop') 
    trans              = op.join(trans_path, 'sdfsd-trans.fif'.format(index)) #either p - last version, a bit belowed, or without letter
    info               = raw_list[index].info
    mne.viz.plot_alignment(info, trans, subject=sub_list[index], dig=True,
                           meg=['helmet', 'sensors'], subjects_dir=data_path,
                           surfaces='head-dense')
    
mne.gui.coregistration(subject='Sub10', subjects_dir=data_path)

# %% CREATING SOURCE SPACE OBJECTS

#### SURFACE SOURCE SPACE
index            = 3
for index in index_array_2[:]:  
    i            = index
    src          = mne.setup_source_space(sub_list[index], spacing='ico4', 
                                          subjects_dir = data_path,  n_jobs=6) #add_dist='patch', 
    print(src)
    plot_bem_kwargs = dict(subject=sub_list[index], subjects_dir=data_path, 
                           brain_surfaces='white', orientation='coronal',
                           slices=[50, 100, 150, 200])
    mne.viz.plot_bem(src=src, **plot_bem_kwargs)
    os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec') 
    mne.write_source_spaces('Sub{}b-ico4-src.fif'.format(index), src, overwrite = True)
  
#### VOLUME SOURCE SPACE
labels_vol      = [     
                   "Left-Hippocampus",                         
                   "Left-Amygdala",                           
                   "Right-Hippocampus",                      
                   "Right-Amygdala"]    

index           = 3
for index in index_array_2[:]:  
    i           = index
    subject     = sub_list[index]
    # surface     = data_path +  '/' + subject + '/' + 'bem' + '/' +  'inner_skull.surf' 
    # mri         = 'L:/SATURN/ABRAR WM/fif for coreg _ forward modeling/subjects/Sub{}/mri/T1.mgz'.format(index)
    fname_aseg  = os.path.join(f_avg_path, subject, 'mri', 'aseg.mgz')
    surface     = os.path.join(f_avg_path, subject, 'bem', 'inner_skull.surf')
    fname_vsrc  = os.path.join(f_avg_path, subject, 'files', 'thal_vol_source_space.fif')
    vol_src     = mne.setup_volume_source_space(subject, pos = 6.2,  mri = fname_aseg,
                                       subjects_dir = f_avg_path,  volume_label=labels_vol,# surface=surface, 
                                       single_volume=True, add_interpolator=True, verbose=True) 
    print(vol_src)
    plot_bem_kwargs = dict(subject=sub_list[index], subjects_dir=f_avg_path, 
                           brain_surfaces='white', orientation='coronal',
                           slices=[50, 100, 150, 200])
    mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)
    os.chdir('L:/SATURN/ABRAR WM/FEB/VOLUME SOURCE SPACE _ dec') 
    mne.write_source_spaces('Sub{}-ico4-vol-hypoamyg_pos5-src.fif'.format(index), vol_src, overwrite = True) 


#### MIXED SOURCE SPACE
index = 3
for index in index_array_2[:]:  
    os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec') 
    src_surf = mne.read_source_spaces('Sub{}-oct6-src.fif'.format(index))
    os.chdir('L:/SATURN/ABRAR WM/FEB/VOLUME SOURCE SPACE _ dec') 
    src_vol  = mne.read_source_spaces('Sub{}-vol-hypoamyg_pos5-src.fif'.format(index))
    src_mixed = src_surf
    src_mixed += src_vol 
    os.chdir('L:/SATURN/ABRAR WM/FEB/MIXED SOURCE SPACE _ dec') 
    mne.write_source_spaces('Sub{}b-ico4-mixed-hypoamyg-src.fif'.format(index), src_mixed, overwrite = True)
        
#### FSAVERAGE SOURCE SPACE
f_avg_path              = 'L:/SATURN/ABRAR WM/fif for coreg _ forward modeling/subjects'
src_avg                 = mne.setup_source_space('fsaverage', spacing = 'ico4',  subjects_dir = f_avg_path)
src_avg_vol             = mne.setup_volume_source_space('fsaverage', pos = 6.2,    subjects_dir = f_avg_path,  
                                                        volume_label=labels_vol, surface=surface, 
                                                        single_volume=True,      add_interpolator=True, verbose=True)

# os.chdir('L:/SATURN/ABRAR WM/FEB/VOLUME SOURCE SPACE _ dec') 
# src_surf                = mne.read_source_spaces('Sub_avg-vol-hypoamyg_pos5-src.fif'.format(index))
# os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec') 
# src_vol  = mne.read_source_spaces('Sub_avg-oct6-src.fif'.format(index))
src_avg_mix             = src_surf
src_avg_mix             +=src_avg_vol
os.chdir('L:/SATURN/ABRAR WM/FEB') 
mne.write_source_spaces('Sub_avg-ico4--src.fif', src_avg, overwrite = True)
mne.write_source_spaces('Sub_avg-ico4-vol-hypoamyg_pos5-src.fif', src_avg_vol, overwrite = True)
mne.write_source_spaces('Sub_avg-ico4-mix-hypoamyg_pos5-src.fif', src_avg_mix, overwrite = True)

#### VISUALIZATION
# index                 = 10
# os.chdir('L:/SATURN/ABRAR WM/FEB/VOLUME SOURCE SPACE _ dec') 
# os.chdir('L:/SATURN/ABRAR WM/FEB/MIXED SOURCE SPACE _ dec') 
# os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec') 
# #Sub{}-oct6-src.fif, Sub{}-oct6-mixed-hypoamyg-src.fif, Sub{}-vol-hypoamyg_pos5-src.fif
# src                     = mne.read_source_spaces('Sub{}-oct6-mixed-hypoamyg-src.fif'.format(index))
# plot_bem_kwargs         = dict(subject=sub_list[index], subjects_dir=data_path,
#                                 brain_surfaces='white', orientation='coronal',
#                                 slices=[50, 100, 150, 200]) 
# mne.viz.plot_bem(src=src, **plot_bem_kwargs)

# %% CREATING FORWARD MODEL
 
#### SUBJECTS
index = 3
for index in index_array[:]: 
    ### UPLOADER
    i = index
    os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec') 
    src_surf            = mne.read_source_spaces('Sub{}b-ico4-src.fif'.format(index))
    os.chdir('L:/SATURN/ABRAR WM/FEB/MIXED SOURCE SPACE _ dec') 
    src_mix             = mne.read_source_spaces('Sub{}b-ico4-mixed-hypoamyg-src.fif'.format(index))
   
    trans               = op.join(trans_path, 'Subject{}-trans.fif'.format(index))
    os.chdir('L:/SATURN/ABRAR WM/FEB') 
    SDRF                = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index)) 
    raw_index           = mne.io.read_raw_fif(SDRF, allow_maxshield=False, 
                                              preload=False, 
                                              on_split_missing='raise', 
                                              verbose=None) 
    info                = raw_index.info

    ### BEM
    # os.chdir('L:/SATURN/ABRAR WM/FEB') 
    # conductivity = (0.3,)  # for single layer
    # model = mne.make_bem_model(subject='Sub{}'.format(index), ico=5,
    #                             conductivity=conductivity,
    #                             subjects_dir=data_path)
    # bem = mne.make_bem_solution(model)
    # mne.write_bem_solution(    'Sub{}-dec-ind-bem-sol.fif'.format(index), bem, 
    #                             overwrite=True, verbose=None)
    bem = mne.read_bem_solution('Sub{}-dec-ind-bem-sol.fif'.format(index))
       
    #FORWARD SOLUTION - SURF
    fwd                 = mne.make_forward_solution(info, trans=trans,
                                                    src=src_surf, bem=bem,
                                                    meg=True, eeg=False, mindist=5.0,
                                                    verbose=True, n_jobs=4)
    #SAVER
    os.chdir('L:/SATURN/ABRAR WM') 
    mne.write_forward_solution('Sub{}b-ico4-dec-surf-fwd.fif'.format(index), fwd, 
                               overwrite = True) 
    #FORWARD SOLUTION - MIX
    fwd_m               = mne.make_forward_solution(info, trans=trans, 
                                                    src=src_mix, bem=bem,
                                                    meg=True, eeg=False, mindist=5.0,
                                                    verbose=True, n_jobs=4)
    #SAVER
    os.chdir('L:/SATURN/ABRAR WM') 
    mne.write_forward_solution('Sub{}b-ico4-dec-mix-fwd.fif'.format(index), fwd_m, 
                                overwrite = True) 
 
#### FSAVERAGE FORWARD MODEL
os.chdir('L:/SATURN/ABRAR WM/FEB') 
SDRF                    = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index)) 
raw_index               = mne.io.read_raw_fif(SDRF, allow_maxshield=False, 
                                              preload=False, 
                                              on_split_missing='raise', 
                                              verbose=None) 

os.chdir('L:/SATURN/ABRAR WM/FEB') 
src_avg                 = mne.read_source_spaces('Sub_avg-ico4--src.fif')
os.chdir('L:/SATURN/ABRAR WM/FEB') 
src_avg_mix             = mne.read_source_spaces('Sub_avg-ico4-mix-hypoamyg_pos5-src.fif')
conductivity            = (0.3,)
model                   = mne.make_bem_model(subject='fsaverage', ico=5,  #ICO 5 → 10240 downsampling
                          conductivity=conductivity, 
                          subjects_dir=f_avg_path)
bem                     = mne.make_bem_solution(model)
trans                   = op.join(trans_path, 'fsaverage-trans.fif')

## SURFACE SRC
fwd                     = mne.make_forward_solution(raw_index.info, trans=trans, 
                                                    src=src_avg, bem=bem,  
                                                    meg=True, eeg=False, mindist=5.0,
                                                    verbose=True, n_jobs = 4)

## MIXED SRC
fwd_m                   = mne.make_forward_solution(raw_index.info, trans=trans,
                                                    src=src_avg_mix, bem=bem, 
                                                    meg=True, eeg=False, mindist=5.0,
                                                    verbose=True, n_jobs = 4)

mne.write_bem_solution(    'Sub_ave--ico4-bem-sol.fif',           bem,   overwrite = True)
mne.write_forward_solution('Sub_ave--ico4-dec-surf-fwd.fif', fwd,   overwrite = True) 
mne.write_forward_solution('Sub_ave--ico4-dec-mix-fwd.fif',  fwd_m, overwrite = True) 


# %% VISUAL CHECK OF THE SRC AND FWD

#### UPLOADER
fwd_ind                =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
fwd_mix                =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
trans_list             =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
src_surf               =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
src_mix                =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
raw_list               =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
sub_list               =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]


index = 3
for index in index_array_2[:]:  
    i = index
    SDRF               = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index))
    raw_index          = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False,
                                             on_split_missing='raise', verbose=None)
    raw_list[index]    = raw_index             
    Subject            = 'Sub{}'.format(index)
    sub_list[index]    = Subject
    trans_list[index]  = op.join(trans_path, 'Subject{}-trans.fif'.format(index))
    
    ### SURFACE SPACE
    os.chdir('L:/SATURN/ABRAR WM/FEB')  
    fwd_ind[ index]    = mne.read_forward_solution('Sub{}-oct6-dec-surf-fwd.fif'.format(index)) 
    os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec')  
    src_surf[index]    = mne.read_source_spaces(            'Sub{}-oct6-src.fif'.format(index))
    
    ### MIXED SPACE
    # os.chdir('L:/SATURN/ABRAR WM/FEB')  
    # fwd_mix[index]     = mne.read_forward_solution('Sub{}-oct6-dec-mix-fwd.fif'.format(index)) 
    # os.chdir('L:/SATURN/ABRAR WM/FEB/MIXED SOURCE SPACE _ dec')  
    # src_mix[index]     = mne.read_source_spaces('Sub{}-oct6-mixed-hypoamyg-src.fif'.format(index))

#### SRC CHECK
index = 10
for index in index_array_2: 
    visualize          = src_surf[index] 
    plot_bem_kwargs    = dict(subject=sub_list[index], subjects_dir=data_path,
                                brain_surfaces='white', orientation='coronal',
                                slices=[50, 100, 150, 200]) 
    mne.viz.plot_bem(src=visualize, **plot_bem_kwargs)
    

### SOURCE CHECK
index = 3
for index in index_array_2[:]: 
    visualize          = src_surf[index] 
    fig                = mne.viz.plot_alignment(subject=sub_list[index], 
                                                   subjects_dir=data_path,
                                                   surfaces='white', coord_frame='mri',
                                                   src=visualize)
    mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                        distance=0.40, focalpoint=(-0.03, -0.01, 0.03))
    mne.viz.set_3d_title(fig, title = 'Sub{}'.format(index))

### FDW CHECK
index = 3
for index in index_array_2: 
    fwd                = fwd_ind[index]
    leadfield          = fwd['sol']['data']
    print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
    print(f'Before: {src_surf[index]}')
    print(f'After:  {fwd["src"]}')
    mag_map = mne.sensitivity_map(fwd)
    kwargs = dict(clim=dict(kind='percent', lims=[0, 50, 99]),
              # no smoothing, let's see the dipoles on the cortex.
              smoothing_steps=1, hemi='rh', views=['lat'])
    
    brain_subject = mag_map.plot(  # plot forward in subject source space (morphed)
    time_label='Original Sub{}'.format(index), subjects_dir=data_path, **kwargs)


# %% CREATING SOURCE ESTIMATE OBJECTS

#### UPLOADER
fwd_ind                =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
fwd_mix                =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
trans_list             =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
src_surf               =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
src_mix                =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
raw_list               =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
sub_list               =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
csd_t_list             =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
csd_s_list             =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
csd_Ab_list            =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

index = 3
for index in index_array_2:  
    i                  = index
    SDRF               = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index))
    raw_index          = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False,
                                             on_split_missing='raise',    verbose=None)
    raw_list[  index]  = raw_index             
    Subject            = 'Sub{}'.format(index)
    sub_list[  index]  = Subject
    trans_list[index]  = op.join(trans_path, 'Subject{}-trans.fif'.format(index))
    
    ### SURFACE SPACE
    os.chdir('L:/SATURN/ABRAR WM/FEB')  
    fwd_ind[index]    = mne.read_forward_solution('Sub{}-oct6-dec-surf-fwd.fif'.format(index)) 
    os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec')  
    src_surf[index]   = mne.read_source_spaces('Sub{}-oct6-src.fif'.format(index))
    
    # ### MIXED SPACE
    # os.chdir('L:/SATURN/ABRAR WM/FEB')  
    # fwd_mix[index]    = mne.read_forward_solution('Sub{}b-oct6-dec-mix-fwd.fif'.format(index)) 
    # os.chdir('L:/SATURN/ABRAR WM/FEB/MIXED SOURCE SPACE _ dec')  
    # src_mix[index]    = mne.read_source_spaces('Sub{}-oct6-mixed-hypoamyg-src.fif'.format(index))
       
    os.chdir('L:/SATURN/ABRAR WM/JAN')
    csd_s             = mne.time_frequency.read_csd('S{}_nov_full_S_csd.h5'.format(index))       
    csd_s_list[index] = csd_s 
    csd_t             = mne.time_frequency.read_csd('S{}_nov_full_T_csd.h5'.format(index))      
    csd_t_list[index] = csd_t 
    
    #uploading average baseline csd for each pax
    os.chdir('L:/SATURN/ABRAR WM/JAN')
    csd_Ab            = mne.time_frequency.read_csd('S{}_nov_average_base_csd.h5'.format(index))        
    csd_Ab_list[index]= csd_Ab 
    
    
  
#### SOURCE ESTIMATE SURFACE ##########################################################
a                     = 31                                                      # !!! Min frequency 
b                     = 80                                                      # !!! Max frequency  
orientation           = 'fix'                                                  # !!! Orientation: free, fixed, tang

stc_s_list            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_t_list            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_A_list            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

index = 3 
for index in index_array_2[:]:    
    #PREPARATION
    ### UNIQUE CSD
    i = index 
    csd_s             = csd_s_list[index]
    csd_t             = csd_t_list[index] 
    csd_to_use        = csd_s.copy()
    csd_to_use._data += csd_t._data
    csd_to_use._data /=2 
    csd_dics          = csd_to_use.mean(fmin=a, fmax=b)  
    
    ### CONDITIONAL CSD
    csd_s             = csd_s_list[index ].mean(fmin=a, fmax=b)    
    csd_t             = csd_t_list[index ].mean(fmin=a, fmax=b)    
    csd_Ab            = csd_Ab_list[index].mean(fmin=a, fmax=b)
 
    fwd               = fwd_ind[index]                                     
    fwd               = mne.convert_forward_solution(fwd, surf_ori=True,        #SOURCE ORIENATION
                                                     force_fixed=True, copy=True, use_cps=True, verbose=None)
    info              = raw_list[index].info
    
    ### FILTER CREATION
    dics_filter     = mne.beamformer.make_dics(info,fwd, csd_dics, reg=0.05, 
                                               #pick_ori='normal',           #can be normal ~ tangential but doesn't work: Normal orientation can only be picked when a forward operator oriented in surface coordinates is used.
                                               inversion='single',              #To get a fixed-orientation forward solution, use mne.convert_forward_solution() to convert the free-orientation solution to (surface-oriented) fixed orientation.
                                               weight_norm=None, 
                                               real_filter=True, depth = 1.) 
    
    print(dics_filter)
    type( dics_filter)
    os.chdir('L:/SATURN/ABRAR WM/JAN') 
    dics_filter.save('Sub{}_dec_from_{}_to_{}_{}_unique-dics.h5'.format( index,a,b,orientation), overwrite = True)
    
    #SOURCE ESTIMATE - FILTER APPLICATION
    stc_s,  freq_s    = mne.beamformer.apply_dics_csd(csd_s,  dics_filter)
    stc_t,  freq_t    = mne.beamformer.apply_dics_csd(csd_t,  dics_filter)
    stc_Ab, freq_Ab   = mne.beamformer.apply_dics_csd(csd_Ab, dics_filter)
    print(stc_s)
    type( stc_s)
    print('sub{}'.format(index), ': ', 'freq_s =',freq_s, ';', 'freq_t =',freq_t, ';', 'freq_A =',freq_Ab, '.') #??? FOR WHAT YOU HAVE IT HERE??? 
    
    #CREATING AN ARRAY
    stc_s_list[index] = stc_s
    stc_t_list[index] = stc_t 
    stc_A_list[index] = stc_Ab
    
    #SAVER
    os.chdir('L:/SATURN/ABRAR WM/JAN') 
    stc_s.save( 'Sub{}_dec_from_{}_to_{}_{}_unique_surf_spatial_stc'.format( index,a,b,orientation),overwrite=True) 
    stc_t.save( 'Sub{}_dec_from_{}_to_{}_{}_unique_surf_temporal_stc'.format(index,a,b,orientation),overwrite=True) 
    stc_Ab.save('Sub{}_dec_from_{}_to_{}_{}_unique_surf_AvBase_stc'.format(  index,a,b,orientation),overwrite=True)


#### SOURCE ESTIMATE MIXED ##########################################################
# a                     = 31                                                      # !!! Min frequency 
# b                     = 80                                                      # !!! Max frequency  
# orientation           = 'free'                                                  # !!! Orientation: free, fixed

# stc_s_list            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
# stc_t_list            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
# stc_A_list            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]


# index = 3 
# for index in index_array_2[:1]:    
#     #PREPARATION
#     i = index 
#     ### UNIQUE CSD
#     csd_s             = csd_s_list[index]
#     csd_t             = csd_t_list[index] 
#     csd_to_use        = csd_s.copy()
#     csd_to_use._data += csd_t._data
#     csd_to_use._data /=2 
#     csd_dics          = csd_to_use.mean(fmin=a, fmax=b)  
    
#     ### CONDITIONAL CSD
#     csd_s             = csd_s_list[index ].mean(fmin=a, fmax=b)    
#     csd_t             = csd_t_list[index ].mean(fmin=a, fmax=b)    
#     csd_Ab            = csd_Ab_list[index].mean(fmin=a, fmax=b)
  
#     fwd               = fwd_mix[ index]                                       
#     info              = raw_list[index].info
    
#     #DICS CREATION
#     dics_filter       = mne.beamformer.make_dics(info, fwd, csd_dics, reg=0.05, 
#                                                  pick_ori='max-power',
#                                                  inversion='single', 
#                                                  weight_norm=None, 
#                                                  real_filter=True) #depth = 1.) 

#     print(dics_filter_t)
#     type(dics_filter_t)
#     os.chdir('L:/SATURN/ABRAR WM/JAN') 
#     dics_filter.save('Sub{}_dec_from_{}_to_{}_{}_mix_unique-dics.h5'.format( index,a,b,orientation), overwrite = True)
    
#     #SOURCE ESTIMATE - FILTER APPLICATION
#     stc_s,  freq_s    = mne.beamformer.apply_dics_csd(csd_s,  dics_filter)
#     stc_t,  freq_t    = mne.beamformer.apply_dics_csd(csd_t,  dics_filter)
#     stc_Ab, freq_Ab   = mne.beamformer.apply_dics_csd(csd_Ab, dics_filter)
#     print(stc_s)
#     type( stc_s)
#     print('sub{}'.format(index), ': ', 'freq_s =',freq_s, ';', 'freq_t =',freq_t, ';', 'freq_A =',freq_Ab, '.')
    
#     #CREATING AN ARRAY
#     stc_s_list[index] = stc_s
#     stc_t_list[index] = stc_t 
#     stc_A_list[index] = stc_Ab
    
#     #SAVER
#     os.chdir('L:/SATURN/ABRAR WM/JAN') 
#     stc_s.save( 'Sub{}_dec_from_{}_to_{}_{}_unique_mix_spatial_stc'.format( index,a,b,orientation),overwrite=True) 
#     stc_t.save( 'Sub{}_dec_from_{}_to_{}_{}_unique_mix_temporal_stc'.format(index,a,b,orientation),overwrite=True) 
#     stc_Ab.save('Sub{}_dec_from_{}_to_{}_{}_unique_mix_AvBase_stc'.format(  index,a,b,orientation),overwrite=True)

########################################### TEMPORAL CHECKER - WORK #################################
# i        = 3 
# stc_plot = stc_s_list[i]
# src      = src_mix[  i] #src_mix #src_surf
# sub      = sub_list[  i]    

# stc_plot.plot(subject=sub, surface='pial',
#                          hemi='split', colormap='auto', 
#                          time_label='auto', smoothing_steps=10, transparent=True, 
#                          alpha=.8, time_viewer='auto', 
#                          subjects_dir=f_avg_path, figure=None, 
#                          views=['dorsal', 'lateral', 'medial','ventral'], 
#                          colorbar=True, 
#                          clim='auto', cortex='classic', size=800, 
#                          background='black', foreground=None, 
#                          initial_time=None, time_unit='s', backend='auto', 
#                          spacing='oct6', title=None, show_traces='auto', src=src, 
#                          volume_options=1.0, view_layout='vertical',
#                          add_data_kwargs=None, brain_kwargs=None, verbose=None)

# %% VISUAL CHECKER

#### UPLOADER - SURFACE
a                    = 12
b                    = 31
orientation          = 'free'                                                   # !!! Orientation: free, fixed
 
src_surf             = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
src_mix              = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
sub_list             = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_s_surf           = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_t_surf           = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_A_surf           = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_s_mix            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_t_mix            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_A_mix            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_fs_s_list        = stc_s_surf
stc_fs_A_list        = stc_fs_t_list = stc_fs_s_list

i                    = 3
for i in index_array_2[:]:    
    index            = i
    os.chdir('L:/SATURN/ABRAR WM/JAN') 
    #SURFACE STC
    stc_s_surf[i]    = mne.read_source_estimate('Sub{}_dec_from_{}_to_{}_{}_unique_surf_spatial_stc'.format( i,a,b, orientation))
    stc_t_surf[i]    = mne.read_source_estimate('Sub{}_dec_from_{}_to_{}_{}_unique_surf_temporal_stc'.format(i,a,b, orientation))
    stc_A_surf[i]    = mne.read_source_estimate('Sub{}_dec_from_{}_to_{}_{}_unique_surf_AvBase_stc'.format(  i,a,b, orientation))
    #MIXED STC
    # stc_s_mix[i]     = mne.read_source_estimate('Sub{}_dec_from_{}_to_{}_{}_mix_spatial_stc'.format( i,a,b, orientation))
    # stc_t_mix[i]     = mne.read_source_estimate('Sub{}_dec_from_{}_to_{}_{}_mix_temporal_stc'.format(i,a,b, orientation))
    # stc_A_mix[i]     = mne.read_source_estimate('Sub{}_dec_from_{}_to_{}_{}_mix_AvBase_stc'.format(  i,a,b, orientation))
    
    Subject          = 'Sub{}'.format(i)
    sub_list[i]      = Subject
    
    os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec')  
    src_surf[i]      = mne.read_source_spaces('Sub{}-oct6-src.fif'.format(i))
    # os.chdir('L:/SATURN/ABRAR WM/FEB/MIXED SOURCE SPACE _ dec')  
    # src_mix[ i]      = mne.read_source_spaces('Sub{}-oct6-mixed-hypoamyg-src.fif'.format(i))

# os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec')      
# src_surf_fs          = mne.read_source_spaces('Sub_avg-oct6-src.fif')
# os.chdir('L:/SATURN/ABRAR WM/FEB/MIXED SOURCE SPACE _ dec')  
# src_mixed_fs         = mne.read_source_spaces('Sub_avg-mix-hypoamyg_pos5-src.fif')

#### PLOTTING
i                    = 3
for i in index_array_2[:]:  
    stc_to_plot      = stc_t_surf[i]
    src              = src_surf[  i]    
    brain_before     = stc_to_plot.plot(subject=sub_list[i], surface='inflated',
                            hemi='split', colormap='auto', 
                            time_label='auto', smoothing_steps=10, transparent=True, 
                            alpha=1., time_viewer='auto', 
                            subjects_dir=f_avg_path, figure=None, 
                            views=['dorsal', 'lateral', 'medial','ventral'], 
                            colorbar=True, 
                            clim='auto', cortex='classic', size=800, 
                            background='black', foreground=None, 
                            initial_time=None, time_unit='s', backend='auto', 
                            spacing='oct6', title=None, show_traces='auto', src=src, 
                            volume_options=1.0, view_layout='vertical',
                            add_data_kwargs=None, brain_kwargs=None, verbose=None)
    
    os.chdir('L:/SATURN/ABRAR WM') 
    brain_before.save_image('Sub{}_from{}_to_{}_T.png'.format(i,a,b))
    
#     stc_before       = (stc_s_surf[i] - stc_t_surf[i]) / stc_A_surf[i]
#     brain_before     = stc_before.plot(subject=sub_list[i], surface='inflated', 
#                                       hemi='both',         colormap='auto', 
#                                       time_label='auto',   smoothing_steps=10, 
#                                       transparent=True,    alpha=1.0, time_viewer='auto', 
#                                       subjects_dir=f_avg_path,    figure=None, 
#                                       views=['dorsal', 'lateral', 'medial','ventral'], 
#                                       colorbar=True, clim='auto', cortex='classic', 
#                                       size=800, background='black', 
#                                       foreground=None, initial_time=None, time_unit='s', backend='auto', 
#                                       spacing='oct6', title=None, show_traces='auto', src=src, 
#                                       volume_options=1.0, view_layout='vertical', add_data_kwargs=None, 
#                                       brain_kwargs=None, verbose=None)
    

# %% MORPHING THE RESULTS TO AVERAGE SUBJECT

#### UPLOADER
a                    = 31
b                    = 80
orientation          = 'tang'                                                 # Orientation: free, fixed
   
src_surf             = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
src_mix              = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
sub_list             = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_s_surf           = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_t_surf           = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_A_surf           = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_s_mix            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_t_mix            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_A_mix            = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_fs_s_list        = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_fs_t_list        = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
stc_fs_A_list        = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

i                    = 3
for i in index_array_2[:]:    
    index            = i
    os.chdir('L:/SATURN/ABRAR WM/JAN') 
    #SURFACE STC          
    stc_s_surf[i]    = mne.read_source_estimate('Sub{}_ico4_dec_from_{}_to_{}_{}_unique_surf_spatial_stc'.format( i,a,b, orientation)) #Sub{}_ico4_dec_from_{}_to_{}_{}_unique_surf_spatial_stc
    stc_t_surf[i]    = mne.read_source_estimate('Sub{}_ico4_dec_from_{}_to_{}_{}_unique_surf_temporal_stc'.format(i,a,b, orientation)) #Sub{}_dec_from_{}_to_{}_{}_unique_surf_spatial_stc'
    stc_A_surf[i]    = mne.read_source_estimate('Sub{}_ico4_dec_from_{}_to_{}_{}_unique_surf_AvBase_stc'.format(  i,a,b, orientation))
    #MIXED STC
    # stc_s_mix[i]     = mne.read_source_estimate('Sub{}_dec_from_{}_to_{}_{}_mix_spatial_stc'.format( i,a,b,orientation))
    # stc_t_mix[i]     = mne.read_source_estimate('Sub{}_dec_from_{}_to_{}_{}_mix_temporal_stc'.format(i,a,b,orientation))
    # stc_A_mix[i]     = mne.read_source_estimate('Sub{}_dec_from_{}_to_{}_{}_mix_AvBase_stc'.format(  i,a,b,orientation))
    
    Subject          = 'Sub{}'.format(i)
    sub_list[i]      = Subject
    
    os.chdir('L:/SATURN/ABRAR WM/FEB/SURFACE SOURCE SPACE _ dec')  
    src_surf[i]      = mne.read_source_spaces('Sub{}b-ico4-src.fif'.format(i))
    # os.chdir('L:/SATURN/ABRAR WM/FEB/MIXED SOURCE SPACE _ dec')  
    # src_mix[ i]      = mne.read_source_spaces('Sub{}-oct6-mixed-hypoamyg-src.fif'.format(i))

os.chdir('L:/SATURN/ABRAR WM/FEB')      
src_surf_fs          = mne.read_source_spaces('Sub_avg-ico4--src.fif')
# os.chdir('L:/SATURN/ABRAR WM/FEB/MIXED SOURCE SPACE _ dec')  
# src_mixed_fs         = mne.read_source_spaces('Sub_avg-mix-hypoamyg_pos5-src.fif')

#### MORPHING THE SURFACE STC
index_array_2 =  [3, 6, 7, 9, 11, 14, 18, 19, 20, 21, 23, 25, 26, 27, 28]

i                    = 3
for i in index_array_2[:]: 
    index            = i
    src              = src_surf[i]
    src_fs           = src_surf_fs

    stc_s_to_morph   = stc_s_surf[i]      
    stc_t_to_morph   = stc_t_surf[i]
    stc_A_to_morph   = stc_A_surf[i]

    #PLOTTING WHAT WAS BEFORE THE MORPING - REQUIRE STC.EXPAND!!!
    stc_before       = (stc_s_to_morph - stc_t_to_morph)/stc_A_to_morph
    brain_before     = stc_before.plot(subject=sub_list[i], surface='inflated', 
                                        hemi='both',         colormap='auto', 
                                        time_label='auto',   smoothing_steps=10, 
                                        transparent=True,    alpha=1.0, time_viewer='auto', 
                                        subjects_dir=f_avg_path,    figure=None, 
                                        views=['dorsal', 'lateral', 'medial','ventral'], 
                                        colorbar=True, clim='auto', cortex='classic', 
                                        size=800, background='black', 
                                        foreground=None, initial_time=None, time_unit='s', backend='auto', 
                                        spacing='oct6', title=None, show_traces='auto', src=src, 
                                        volume_options=1.0, view_layout='vertical', add_data_kwargs=None, 
                                        brain_kwargs=None, verbose=None)
    os.chdir('C:/Users/Nikita O/Desktop/DICS')
    brain_before.save_image('Sub{}_ico4_from{}_to_{}_unique_cont_before.png'.format(i,a,b))
    
    morph_surf_s     = mne.compute_source_morph(stc_s_to_morph, 
                                     subject_from=sub_list[i], 
                                     subject_to = 'fsaverage', 
                                     src_to=src_fs, #src_to=src_fs, 
                                     subjects_dir=f_avg_path, smooth = 20,
                                     # spacing = fsave_vertices, #verbose
                                     verbose=True)
    morph_surf_t     = mne.compute_source_morph(stc_t_to_morph, 
                                     subject_from=sub_list[i], 
                                     subject_to = 'fsaverage', 
                                     src_to=src_fs, #src_to=src_fs, 
                                     subjects_dir=f_avg_path, smooth = 20,
                                     # spacing = fsave_vertices, #verbose
                                     verbose=True)
    morph_surf_A     = mne.compute_source_morph(stc_A_to_morph, 
                                     subject_from=sub_list[i], 
                                     subject_to = 'fsaverage', 
                                     src_to=src_fs, #src_to=src_fs, 
                                     subjects_dir=f_avg_path, smooth = 20,
                                     # spacing = fsave_vertices, #verbose
                                     verbose=True)
    
    stc_fs_s        = morph_surf_s.apply(stc_s_to_morph) #.morph_mat - to create a matrix to use instead of SourceMorph
    stc_fs_t        = morph_surf_t.apply(stc_t_to_morph)
    stc_fs_A        = morph_surf_A.apply(stc_A_to_morph)    
    stc_fs_s_list[i]= stc_fs_s
    stc_fs_t_list[i]= stc_fs_t
    stc_fs_A_list[i]= stc_fs_A
    os.chdir('L:/SATURN/ABRAR WM/') 
    stc_fs_s.save('Sub{}_ico4_from_{}_to_{}_{}_unique_morph_surf_spatial_stc'.format(   i,a,b,orientation),overwrite=True)
    stc_fs_t.save('Sub{}_ico4_from_{}_to_{}_{}_unique_morph_surf_temporal_stc'.format(  i,a,b,orientation),overwrite=True) 
    stc_fs_A.save('Sub{}_ico4_from_{}_to_{}_{}_unique_morph_surf_AvBaseline_stc'.format(i,a,b,orientation),overwrite=True)
    morph_surf_s.save('Sub{}_ico4_from_{}_to_{}_{}_morph_surf_s.h5'.format(      i,a,b,orientation),overwrite=True)
    morph_surf_t.save('Sub{}_ico4_from_{}_to_{}_{}_morph_surf_t.h5'.format(      i,a,b,orientation),overwrite=True)
    morph_surf_A.save('Sub{}_ico4_from_{}_to_{}_{}_morph_surf_A.h5'.format(      i,a,b,orientation),overwrite=True)
    
    #CONTRASTING PLOT
    stc_after       = (stc_fs_s - stc_fs_t) / stc_fs_A
    brain_after     = stc_after.plot(subject='fsaverage', surface='inflated', hemi='both', 
                      colormap='auto', time_label='auto', smoothing_steps=10,
                      transparent=True, alpha=1.0, time_viewer='auto', subjects_dir=f_avg_path, 
                      figure=None, views=['dorsal', 'lateral', 'medial','ventral'], colorbar=True, clim='auto', cortex='classic',
                      size=800, background='black', foreground=None, initial_time=0, time_unit='s',
                      backend='auto', spacing='oct6', title=None, show_traces='auto', src=src_fs,
                      volume_options=1.0, view_layout='vertical', add_data_kwargs=None,
                      brain_kwargs=None, verbose=None)
    os.chdir('C:/Users/Nikita O/Desktop/DICS')
    brain_after.save_image('Sub{}_ico4_from{}_to_{}_unique_cont_{}_after.png'.format(i,a,b,orientation))


#### MORPHING THE MIXED SOURCE ESTIMATE

# i                    = 3
# for i in index_array_2: 
    # stc_s_to_morph   = stc_s_mix[i]    
    # stc_t_to_morph   = stc_t_mix[i]
    # stc_A_to_morph   = stc_A_mix[i]
    # src              = src_mix[  i]
    # src_fs           = src_mixed_fs
      
    # #MORPHING
    # morph_vol = mne.compute_source_morph(src, 
    #                                 subject_from=sub_list[i], 
    #                                 subject_to = 'fsaverage', 
    #                                 src_to=src_fs, #src_to=src_fs, 
    #                                 subjects_dir=f_avg_path,
    #                                 verbose=True)
            
    # stc_fs_s          = morph_vol.apply(stc_s_to_morph)
    # # img_s           = morph.apply(stc_s_to_morph, mri_resolution=2, output='nifti1')
    # stc_fs_t          = morph_vol.apply(stc_t_to_morph)
    # # img_t           = morph.apply(stc_t_to_morph, mri_resolution=2, output='nifti1')
    # stc_fs_A          = morph_vol.apply(stc_A_to_morph) 
    # # img_A           = morph.apply(stc_A_to_morph, mri_resolution=2, output='nifti1')
    
    # stc_fs_s_list[i]  = stc_fs_s
    # stc_fs_t_list[i]  = stc_fs_t
    # stc_fs_A_list[i]  = stc_fs_A
    # os.chdir('L:/SATURN/ABRAR WM/FEB') 
    # stc_fs_s.save('Sub{}_dec_from_{}_to_{}_vol_morph_spatial_stc'.format(   i,a,b),overwrite=True)
    # stc_fs_t.save('Sub{}_dec_from_{}_to_{}_vol_morph_temporal_stc'.format(  i,a,b),overwrite=True) 
    # stc_fs_A.save('Sub{}_dec_from_{}_to_{}_vol_morph_AvBaseline_stc'.format(i,a,b),overwrite=True)
    # morph_vol.save('Sub{}_dec_morph_vol.h5'.format(i), overwrite=True)

    # display = (stc_fs_s - stc_fs_t) / stc_fs_A
    # display = plot_glass_brain(t1_fsaverage,
    #                        title='subject results to fsaverage',
    #                        draw_cross=False,
    #                        annotate=True)




