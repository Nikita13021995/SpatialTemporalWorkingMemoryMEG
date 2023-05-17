# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:48:51 2023

@author: Nikita O
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:51:05 2022

@author: Nikita O
"""
# %% LIBRARIES

import os
import os.path as op
import tracemalloc

import numpy as np
from numpy.random import randn
from scipy import stats as stats

import matplotlib.pyplot as plt

import mne
from h5io import write_hdf5, read_hdf5
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
from mayavi import mlab
import mne
from operator import add
from functools import reduce
from mne.epochs import equalize_epoch_counts
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.datasets import sample
from mne import read_source_estimate
from mne.minimum_norm import read_inverse_operator
import scipy
%matplotlib qt

import conpy
import sklearn
import h5io
import h5py
import nilearn

folder_with_files         = 'T:/to github/'                                     ###!!!CHANGABLE!!!
SDF                       = folder_with_files
data_path                 = folder_with_files + '/freesurfer'
bem_dir                   = folder_with_files + '/freesurfer' + '/fsaverage/bem'
trans_path                = folder_with_files

##### FOR CYCLES
index_array_2             = [3,6,7,9,11, 14, 18,19,20,21,23,25,26,27,28] 
spacing = 'oct6'

# tracemalloc.start()

# %% Average Source Space
import shutil
fsaverage_src_dir = op.join(data_path, 'subjects', 'fsaverage')
fsaverage_dst_dir = op.join(data_path,'fsaverage', 'inner_skull.surf')

# if not op.isdir(fsaverage_dst_dir,'inner_skull.surf'):
#     # Remove symlink if present
#     os.unlink(fsaverage_dst_dir,'inner_skull.surf')

# if not op.exists('inner_skull.surf'):
#     shutil.copytree(fsaverage_src_dir,  'inner_skull.surf')

if not op.isdir(bem_dir):
    os.mkdir(bem_dir)

# Create source space on the fsaverage brain
fsaverage = mne.setup_source_space('fsaverage', spacing=spacing,
                                   subjects_dir=data_path,
                                   n_jobs=4, add_dist=False)
mne.write_source_spaces('Sub_avg-oct6-src.fif', fsaverage, overwrite = True)

# os.chdir(folder_with_files)      
# src_avg                 = mne.setup_source_space('fsaverage', spacing = 'oct6',  subjects_dir = data_path) 
# mne.write_source_spaces('Sub_avg-oct6-src.fif', src_avg, overwrite = True)

# %% MORPHING

#####  AVERAGE SUBJECT
os.chdir(folder_with_files)     
src_surf_fs               = mne.read_source_spaces('Sub_avg-oct6-src.fif')

index = 3
for index in index_array_2:  
   # SDRF                   = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index))
   # raw_index              = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False,
   #                                       on_split_missing='raise', verbose=None)
   os.chdir(folder_with_files)
   # subject_src = mne.morph_source_spaces(src_surf_fs, 'Sub{}'.format(index), data_path)
   subject_src = mne.morph_source_spaces(src_surf_fs, 
                                         'Sub{}'.format(index), 
                                         subjects_dir  = data_path, 
                                         )
   os.chdir(folder_with_files)      
   mne.write_source_spaces('Sub{}-mor-src.fif'.format(index), subject_src, overwrite = True)

# %% FORWARD MODELLING for morphed subject

index = 3
for index in index_array_2:
    i = index
    SDRF                   = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index))
    raw_index              = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False,
                                          on_split_missing='raise', verbose=None)
    os.chdir(folder_with_files) 
    src = mne.read_source_spaces('Sub{}-mor-src.fif'.format(index))
    
    ### TRANS FILE
    trans                  = os.path.join(trans_path, 'Subject{}-trans.fif'.format(index))
    
    verts                  = conpy.select_vertices_in_sensor_range(src, 
                                                                   dist=0.07, 
                                                                   info=raw_index.info, 
                                                                   trans = trans) 
    src_sub                = conpy.restrict_src_to_vertices(       src, verts)

    os.chdir(folder_with_files) 
    
    bem_model = mne.make_bem_model('Sub{}'.format(index), ico=5, subjects_dir=data_path,
                               conductivity=(0.3,))
    bem = mne.make_bem_solution(bem_model)
    # bem = mne.read_bem_solution('Sub{}-dec-ind-bem-sol.fif'.format(index))
    
    fwd   = mne.make_forward_solution(
                                    raw_index.info,
                                    trans=trans,
                                    src=src_sub,
                                    bem=bem,
                                    meg=True,
                                    eeg=False,
                                    mindist= 0,
                                    n_jobs=6)                                     ###!!!CHANGABLE!!!
  
    os.chdir(folder_with_files) 
    mne.write_forward_solution('Sub{}-MOR-fwd.fif'.format(index), fwd, overwrite=True)
    
# #### AVERAGE SUBJECT
# os.chdir(folder_with_files)     
# src_surf_fs            = mne.read_source_spaces('Sub_avg-oct6-src.fif')

# trans                  = os.path.join(trans_path, 'fsaverage-trans.fif')
# verts                  = conpy.select_vertices_in_sensor_range(src_surf_fs, 
#                                                                dist=0.07, 
#                                                                info=raw_index.info, 
#                                                                trans = trans) 
# src_sub                = conpy.restrict_src_to_vertices( src_surf_fs, verts)

# os.chdir(folder_with_files)   
# model                  = mne.make_bem_model(subject='fsaverage', ico=5,  #ICO 5 → 10240 downsampling
#                           conductivity=(0.3,), 
#                           subjects_dir=data_path)
# bem                    = mne.make_bem_solution(model)
# fwd                    = mne.make_forward_solution(
#                                 raw_index.info,
#                                 trans=trans,
#                                 src=src_sub,
#                                 bem=bem,
#                                 meg=True,
#                                 eeg=False,
#                                 mindist= 0,
#                                 n_jobs=8)

# os.chdir(folder_with_files)   
# mne.write_forward_solution('Sub_ave-fwd.fif'.format(index), fwd, overwrite=True)


# %% PREPARATION TO CONNECTIVITY ALL-TO-ALL

#### UPLOADER
fwd_ind                =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
src_surf               =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
sub_list               =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
os.chdir(folder_with_files)   
src_surf_fs            = mne.read_source_spaces('Sub_avg-oct6-src.fif')
# fwd_ave                = mne.read_forward_solution('Sub_ave-fwd.fif') 
fwd_ind_2              =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

index = 3
for index in index_array_2[:]:  
    os.chdir(folder_with_files)   
    fwd_ind[index]     = mne.read_forward_solution('Sub{}-MOR-fwd.fif'.format(index)) 
    fwd_ind[index]     = conpy.forward_to_tangential( fwd_ind[index] )
    os.chdir(folder_with_files)   
    src_surf[index]    = mne.read_source_spaces('Sub{}-mor-src.fif'.format(index))

### LIST OF SUBJECT   
index                  = 3
for index in index_array_2[:]:    
    Subject            = 'Sub{}'.format(index)
    sub_list[index]    = Subject
    

### FINDING THE SHARED VERTICES
fwd_ind                = np.delete(fwd_ind, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
src_surf               = np.delete(src_surf,[0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])

max_sensor_dist        = 0.07
fwd_ind[0]             = conpy.restrict_forward_to_sensor_range(fwd_ind[0], max_sensor_dist)

vert_inds              = conpy.select_shared_vertices(fwd_ind, src_surf_fs, data_path)


for fwd, vert_ind, index in zip(fwd_ind, vert_inds, index_array_2):
    fwd_r = conpy.restrict_forward_to_vertices(fwd, vert_ind)
    os.chdir(folder_with_files)   
    fwd_ind_2[index] = fwd_r
    mne.write_forward_solution('Sub{}-oct6-commonvertices-surf-fwd.fif'.format(index), fwd_r,
                               overwrite=True)
    # if index == index_array_2[0]:
    #     fwd_ind[0]  = fwd_r
    
fwd_ind_2                = np.delete(fwd_ind_2, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])      
fwd1 = fwd_ind_2[0]
#### IDENTIFING PAIRS
print('Computing connectivity pairs for all subjects...')
pairs = conpy.all_to_all_connectivity_pairs(fwd1, min_dist=0.04)

subj1_to_fsaverage = conpy.utils.get_morph_src_mapping(
    src_surf_fs,  fwd1['src'], indices=True, subjects_dir=data_path)[1]
pairs = [[subj1_to_fsaverage[v] for v in pairs[0]],
         [subj1_to_fsaverage[v] for v in pairs[1]]]
np.save('Average-pairs', pairs)
  
# %% CONNECTIVITY ALL-TO-ALL
os.chdir(folder_with_files) 
pairs                = np.load('Average-pairs.npy')


# Compute connectivity for one frequency band across all conditions
csd_t_list             =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
csd_s_list             =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
csd_Ab_list            =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#cons                   =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

reg                    = 0.05
a                      = 31
b                      = 80 
index                  = 3

for index in index_array_2:  
    os.chdir(folder_with_files)   
    fwd_ind   = mne.read_forward_solution('Sub{}-oct6-commonvertices-surf-fwd.fif'.format(index)) 
  
    fwd_tan              = conpy.forward_to_tangential(fwd_ind)

    fsaverage            = mne.read_source_spaces('Sub_avg-oct6-src.fif')

    fsaverage_to_subj    = conpy.utils.get_morph_src_mapping(fsaverage, fwd_tan['src'], 
                                                        indices=True, 
                                                        subjects_dir=data_path)[0]
    pairs                = [[fsaverage_to_subj[v] for v in pairs[0]],  [fsaverage_to_subj[v] for v in pairs[1]]]


    os.chdir('L:/SATURN/ABRAR WM/JAN')
    csd_s              = mne.time_frequency.read_csd('S{}_nov_full_S_csd.h5'.format(index))       
    # csd_s_list[index]  = csd_s 
    csd_t              = mne.time_frequency.read_csd('S{}_nov_full_T_csd.h5'.format(index))      
    # csd_t_list[index]  = csd_t 
    # #uploading average baseline csd for each pax
    # os.chdir('L:/SATURN/ABRAR WM/JAN')
    csd_Ab             = mne.time_frequency.read_csd('S{}_nov_average_base_csd.h5'.format(index))        
    # csd_Ab_list[index] = csd_Ab 
    # csd_s              = csd_s_list[index]
    # csd_t              = csd_t_list[index] 
    # csd_to_use         = csd_s.copy()
    # csd_to_use._data  += csd_t._data
    # csd_to_use._data  /=2 
    # csd_dics           = csd_to_use.mean(fmin=a, fmax=b)  
    
    # csd_s              = csd_s_list[index ].mean(fmin=a, fmax=b)    
    # csd_t              = csd_t_list[index ].mean(fmin=a, fmax=b)    
    # csd_Ab            =  csd_Ab_list[index].mean(fmin=a, fmax=b)
    csd_s              = csd_s.mean(fmin=a, fmax=b)    
    csd_t              = csd_t.mean(fmin=a, fmax=b)    
    csd_Ab            =  csd_Ab.mean(fmin=a, fmax=b)
    
    # Compute connectivity for all frequency bands
    con_s = conpy.dics_connectivity(
        vertex_pairs=pairs,
        fwd=fwd_tan,
        data_csd=csd_s,
        reg=reg,
        n_jobs=7)
    con_t = conpy.dics_connectivity(
        vertex_pairs=pairs,
        fwd=fwd_tan,
        data_csd=csd_t,
        reg=reg,
        n_jobs=7)
    # con_av = conpy.dics_connectivity(
    #     vertex_pairs=pairs,
    #     fwd=fwd_tan,
    #     data_csd=csd_Ab,
    #     reg=reg,
    #     n_jobs=7)
    
    os.chdir(folder_with_files)   
    con_s.save('Sub{}-connectivity for band from {} to {}_spatial'.format(index,a,b))
    con_t.save('Sub{}-connectivity for band from {} to {}_temporal'.format(index,a,b))
    con_av.save('Sub{}-connectivity for band from {} to {}_avbase'.format(index,a,b))
    ### PLOTTING THE adjacency MATRIX
    # adj = (cons[conditions[0]] - cons[conditions[1]]).get_adjacency()
    # fig = plt.figure()
    # plt.imshow(adj.toarray(), interpolation='nearest')
    # report.add_figs_to_section(fig, ['Adjacency matrix'],
    #                         section='Source-level', replace=True)

# %% VISUALIZATION ALL-TO-ALL - separate

# ADJECENCY
adj = (con_s - con_t).get_adjacency()
fig = plt.figure()
plt.imshow(adj.toarray(), interpolation='nearest')


# CIRCLE
index = 7
labels = mne.read_labels_from_annot('Sub{}'.format(index), 'aparc',
                                    subjects_dir=data_path)
del labels [-1]

p = con_s.parcellate(labels, 'degree', weight_by_degree=True)                   #summary='degree', #weight_by_degree=True


p.plot(n_lines=1000, vmin=0, vmax=1)
plt.title('Strongest parcel-to-parcel connection', color='white')
fig.savefig('../paper/figures/squircle.pdf', bbox_inches='tight')

# BRAIN
con_contrast = con_s - con_t
all_to_all = con_contrast.make_stc('absmax')                                    #'degree', weight_by_degree=True
brain = all_to_all.plot('Sub{}'.format(index), subjects_dir=data_path, hemi='both',
                        figure=6, size=400)
# mlab.title('All-to-all coherence (contrast)', height=0.9)
brain.add_annotation('aparc', borders=2)


# %% CONNECTIVITY STAT

#### UPLOADER
con_surf_theta_s = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
con_surf_theta_t = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
con_surf_theta_a = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
con_surf_alpha_s = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
con_surf_alpha_t = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
con_surf_alpha_a = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
con_surf_beta_s  = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
con_surf_beta_t  = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
con_surf_beta_a  = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] 
con_surf_gamma_s = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
con_surf_gamma_t = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
con_surf_gamma_a = [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

os.chdir(folder_with_files)   
fsaverage        = mne.read_source_spaces('Sub_avg-oct6-src.fif')

 
i = 3
for i in index_array_2:  
    os.chdir(folder_with_files)
    # con_t_s =  conpy.read_connectivity('Sub{}-connectivity for band from {} to {}_spatial.h5'.format(i, 4,8  )) 
    con_t_s =  conpy.read_connectivity('Sub{}-av-connectivity for band from {} to {}_spatial.h5'.format(i, 4,8  )) 
    # con_fsaverage_t_s = con_t_s.to_original_src(fsaverage, subjects_dir=data_path)
    # con_fsaverage_t_s.save('Sub{}-av-connectivity for band from {} to {}_spatial'.format(i,4,8))
    # con_a_s =  conpy.read_connectivity('Sub{}-connectivity for band from {} to {}_spatial.h5'.format(i, 8,12 )) 
    con_a_s =  conpy.read_connectivity('Sub{}-av-connectivity for band from {} to {}_spatial.h5'.format(i, 8,12 )) 
    # con_fsaverage_a_s = con_a_s.to_original_src(fsaverage, subjects_dir=data_path)
    # con_fsaverage_a_s.save('Sub{}-av-connectivity for band from {} to {}_spatial'.format(i,8,12))
    # con_b_s =  conpy.read_connectivity('Sub{}-connectivity for band from {} to {}_spatial.h5'.format(i, 12,31))
    con_b_s =  conpy.read_connectivity('Sub{}-av-connectivity for band from {} to {}_spatial.h5'.format(i, 12,31))
    # con_fsaverage_b_s = con_b_s.to_original_src(fsaverage, subjects_dir=data_path)
    # con_fsaverage_b_s.save('Sub{}-av-connectivity for band from {} to {}_spatial'.format(i,12,31))
    # con_g_s =  conpy.read_connectivity('Sub{}-connectivity for band from {} to {}_spatial.h5'.format(i, 31,80))
    con_g_s =  conpy.read_connectivity('Sub{}-av-connectivity for band from {} to {}_spatial.h5'.format(i, 31,80))
    # con_fsaverage_g_s = con_g_s.to_original_src(fsaverage, subjects_dir=data_path)
    # con_fsaverage_g_s.save('Sub{}-av-connectivity for band from {} to {}_spatial'.format(i,31,80))
                              
    # con_surf_theta_s[i] = con_fsaverage_t_s
    con_surf_theta_s[i] = con_t_s
    # con_surf_alpha_s[i] = con_fsaverage_a_s
    con_surf_alpha_s[i] = con_a_s
    # con_surf_beta_s[ i] = con_fsaverage_b_s
    con_surf_beta_s[ i] = con_b_s
    # con_surf_gamma_s[i] = con_fsaverage_g_s
    con_surf_gamma_s[i] = con_g_s
   
    # con_t_t =  conpy.read_connectivity('Sub{}-connectivity for band from {} to {}_temporal.h5'.format(i, 4,8  ))
    con_t_t =  conpy.read_connectivity('Sub{}-av-connectivity for band from {} to {}_temporal.h5'.format(i, 4,8  ))
    # con_fsaverage_t_t = con_t_t.to_original_src(fsaverage, subjects_dir=data_path)
    # con_fsaverage_t_t.save('Sub{}-av-connectivity for band from {} to {}_temporal'.format(i,4,8))
    # con_a_t =  conpy.read_connectivity('Sub{}-connectivity for band from {} to {}_temporal.h5'.format(i, 8,12 ))
    con_a_t =  conpy.read_connectivity('Sub{}-av-connectivity for band from {} to {}_temporal.h5'.format(i, 8,12 ))
    # con_fsaverage_a_t = con_a_t.to_original_src(fsaverage, subjects_dir=data_path)
    # con_fsaverage_a_t.save('Sub{}-av-connectivity for band from {} to {}_temporal'.format(i,8,12))
    # con_b_t =  conpy.read_connectivity('Sub{}-connectivity for band from {} to {}_temporal.h5'.format(i, 12,31))
    con_b_t =  conpy.read_connectivity('Sub{}-av-connectivity for band from {} to {}_temporal.h5'.format(i, 12,31))
    # con_fsaverage_b_t = con_b_t.to_original_src(fsaverage, subjects_dir=data_path)
    # con_fsaverage_b_t.save('Sub{}-av-connectivity for band from {} to {}_temporal'.format(i,12,31))
    # con_g_t =  conpy.read_connectivity('Sub{}-connectivity for band from {} to {}_temporal.h5'.format(i, 31,80))
    con_g_t =  conpy.read_connectivity('Sub{}-av-connectivity for band from {} to {}_temporal.h5'.format(i, 31,80))
    # con_fsaverage_g_t = con_g_t.to_original_src(fsaverage, subjects_dir=data_path)
    # con_fsaverage_g_t.save('Sub{}-av-connectivity for band from {} to {}_temporal'.format(i,31,80))
    
    # con_surf_theta_t[i] = con_fsaverage_t_t
    con_surf_theta_t[i] = con_t_t
    # con_surf_alpha_t[i] = con_fsaverage_a_t
    con_surf_alpha_t[i] = con_a_t
    # con_surf_beta_t[ i] = con_fsaverage_b_t
    con_surf_beta_t[ i] = con_b_t
    # con_surf_gamma_t[i] = con_fsaverage_g_t
    con_surf_gamma_t[i] = con_g_t
    

    
#### DELETER
#THETA
con_surf_theta_a = np.delete(con_surf_theta_a, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_theta_a)
con_surf_theta_s = np.delete(con_surf_theta_s, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_theta_s)
con_surf_theta_t = np.delete(con_surf_theta_t, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_theta_t)

#ALPHA
con_surf_alpha_a = np.delete(con_surf_alpha_a, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_alpha_a)
con_surf_alpha_s = np.delete(con_surf_alpha_s, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_alpha_s)
con_surf_alpha_t = np.delete(con_surf_alpha_t, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_alpha_t)

#BETA
con_surf_beta_a = np.delete(con_surf_beta_a,   [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_beta_a)
con_surf_beta_s = np.delete(con_surf_beta_s,   [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_beta_s)
con_surf_beta_t = np.delete(con_surf_beta_t,   [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_beta_t)

#GAMMA
con_surf_gamma_a = np.delete(con_surf_gamma_a, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_gamma_a)
con_surf_gamma_s = np.delete(con_surf_gamma_s, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_gamma_s)
con_surf_gamma_t = np.delete(con_surf_gamma_t, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
np.shape(con_surf_gamma_t)

##########
what_to_stat_s = con_surf_alpha_s
what_to_stat_t = con_surf_alpha_t
 
# what_to_stat_t = con_surf_theta_t
# what_to_stat_s = con_surf_theta_s + con_surf_theta_t
# what_to_stat_s2= con_surf_theta_s
# what_to_stat_s2.append(con_surf_theta_t)
# what_to_stat_a = con_surf_theta_a
# what_to_stat_a = con_surf_theta_a

# ##### prior parcellation - doesn't give
# i = 0
# stat = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# base = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# for i in range(len(what_to_stat_s)): 
#     stat[i] = what_to_stat_s[i].parcellate(selected_label, summary='absmax',
#                                 weight_by_degree=False)
#     base[i] = what_to_stat_a[i].parcellate(selected_label, summary='absmax',
#                                 weight_by_degree=False)

############################################################################# CHECKER - COMPARISON OF MAINTANANCE and BASELINE

# i=1
# what_to_stat_s[i].is_compatible(what_to_stat_t[i])
# con_surf_theta_s[i].is_compatible(con_surf_theta_a[i])
# con_surf_theta_t[i].is_compatible(con_surf_theta_a[i])

# asd_s = what_to_stat_s[i].make_stc(summary='sum', weight_by_degree=True)
# asd_t = what_to_stat_t[i].make_stc(summary='sum', weight_by_degree=True)
# asd_a = what_to_stat_a[i].make_stc(summary='sum', weight_by_degree=True)

# rar = asd_s + asd_t - asd_a

# mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)                                                                             
# brain = rar.plot('fsaverage', subjects_dir=data_path, hemi='both',
#                         figure=6, size=400, views=['dorsal', 'lateral','ventral'])
# brain = asd_s.plot('fsaverage', subjects_dir=data_path, hemi='both',
#                         figure=6, size=400, views=['dorsal', 'lateral','ventral'])
# brain = asd_t.plot('fsaverage', subjects_dir=data_path, hemi='both',
#                         figure=6, size=400, views=['dorsal', 'lateral','ventral'])
# brain = asd_a.plot('fsaverage', subjects_dir=data_path, hemi='both',
#                         figure=6, size=400, views=['dorsal', 'lateral','ventral'])

# pravda = (what_to_stat_s[i] +  what_to_stat_t[i]) - what_to_stat_a[i]
# avbas = what_to_stat_a[i]
# spat = what_to_stat_s[i] 
# temp = what_to_stat_t[i]

# what='pravda'
# con_parc = pravda .parcellate(selected_label, summary='absmax',
#                                 weight_by_degree=False)

# con_parc.plot(n_lines=100, vmin=0, vmax=1, node_colors=label_colors, fontsize_names=10, fontsize_colorbar = 10)
# plt.title('{}'.format(what), color='white')




############################################################################# YOU CAN ALSO TRY ".data"
ga_con = dict()

con = what_to_stat_s[0].copy() #let's consider 0 to be a subject , THEN # con = con_fsaverage_t_a.copy() #Equivalent 
for other_con in what_to_stat_s[1:]:
    con += other_con
con /= len(what_to_stat_s)
ga_con[1] = con

con = what_to_stat_t[0].copy()
for other_con in what_to_stat_t[1:]:
    con += other_con
con /= len(what_to_stat_t)
ga_con[2] = con

contrast =  ga_con[1] - ga_con[2]

#### CLEANING MEMORY

# ######### HEMISPHERE PARCELLATION
# hemi_to_use = 'lh' #'rh', 
del con_surf_beta_s, con_surf_beta_t, con_surf_beta_a
del con_surf_gamma_s, con_surf_gamma_s
# os.chdir(folder_with_files)   
# labels = mne.read_labels_from_annot('fsaverage', 'aparc',  subjects_dir=data_path)
# selected_label = mne.read_labels_from_annot(
#     'fsaverage', regexp=hemi_to_use, subjects_dir=data_path)

# for i in range(len(what_to_stat_s)):
#     what_to_stat_s[i] = what_to_stat_s[i].parcellate(selected_label, summary='absmax',
#                                 weight_by_degree=False)
#     what_to_stat_t[i] = what_to_stat_t[i].parcellate(selected_label, summary='absmax',
#                                 weight_by_degree=False)

# contrast = contrast.parcellate(selected_label, summary='absmax',
#                             weight_by_degree=False)

# ### CHECKER
# i = 10

# label_colors = [label.color for label in selected_label]
# del labels[-1]  # drop 'unknown-lh' label
# what_to_stat_s[i].plot(n_lines=10, vmin=0, vmax=1, node_colors=label_colors, fontsize_names=10, fontsize_colorbar = 10)
# plt.title('Strongest parcel-to-parcel connection', color='white')

### #Compute contrast between faces and scrambled pictures
### # contrast = what_to_stat_s - what_to_stat_t
### # contrast.save(fname.ga_con(condition='contrast'))
### sdf = contrast.data
### Perform a permutation test to only retain connections that are part of a
### significant bundle.
cluster_threshold = 6
n_perm            = 1000
nj                = 1
ms                = 0.05
tl                = 1

stats = conpy.cluster_permutation_test(
    what_to_stat_s, what_to_stat_t,
    cluster_threshold=cluster_threshold, src=fsaverage, n_permutations=n_perm, 
    verbose=True, alpha=0.05, tail = tl, 
    n_jobs=nj,
    seed=10, 
    return_details=True, max_spread=ms)

connection_indices, bundles, bundle_ts, bundle_ps, H0 = stats

##### H5 file creation
freqs = 'alpha_{}-TAIL'.format(tl)
con_clust = contrast[connection_indices]
con_clust.save('Con_stat_cl-{}_maxspread-{}_{}_prunned_to_contr'.format(cluster_threshold, ms, freqs))

write_hdf5('Con_stat_cl-{}_maxspread-{}_{}'.format(cluster_threshold, ms, freqs), dict(
    connection_indices=connection_indices,
    bundles=bundles,
    bundle_ts=bundle_ts,
    bundle_ps=bundle_ps,
    H0=H0), overwrite=True)
#overwrite=False, compression=4,title='h5io', slash='error', use_json=False):
    

##### PARCEL
os.chdir(folder_with_files)   
labels = mne.read_labels_from_annot('fsaverage', 'aparc',  subjects_dir=data_path)
label_colors = [label.color for label in labels]
del labels[-1]  # drop 'unknown-lh' label
con_parc = con_clust.parcellate(labels, summary='degree',
                                weight_by_degree=False)
con_parc.save('Con_stat_cl-{}_maxspread-{}_{}_prunned_to_contr-parcelled-1sign'.format(cluster_threshold, ms, freqs))

##### VISUAL
# # CIRCLE
# con_parc.plot(n_lines=50, vmin=0, vmax=1, node_colors=label_colors)
# plt.title('Strongest parcel-to-parcel connection', color='white')
# plt.savefig('Con_{}-50_0to100-.05_parcell_to_parcel-1sign', bbox_inches='tight')


# # BRAIN
# mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)   
                                                                                
# all_to_all = con_clust.make_stc('absmax')                                    #'degree', weight_by_degree=True
# brain = all_to_all.plot('fsaverage', subjects_dir=data_path, hemi='split',
#                         figure=6, size=400, views=['dorsal', 'lateral', 'medial','ventral'])
# # mlab.title('All-to-all coherence (contrast)', height=0.9)
# brain.add_annotation('aparc', borders=1)


# %% H5 READER 

what_to_stat_s = con_surf_beta_s
what_to_stat_t = con_surf_beta_t
#what_to_stat_s = np.concatenate([con_surf_gamma_s,con_surf_gamma_t])
#what_to_stat_t = np.concatenate([con_surf_gamma_a,con_surf_gamma_a])


############################################################################# YOU CAN ALSO TRY ".data"
ga_con = dict()

con = what_to_stat_s[0].copy() #let's consider 0 to be a subject , THEN # con = con_fsaverage_t_a.copy() #Equivalent 
for other_con in what_to_stat_s[1:]:
    con += other_con
con /= len(what_to_stat_s)
ga_con[1] = con

con = what_to_stat_t[0].copy()
for other_con in what_to_stat_t[1:]:
    con += other_con
con /= len(what_to_stat_t)
ga_con[2] = con

contrast =  ga_con[2] - ga_con[1]


# os.chdir(folder_with_files)
# cluster_threshold = 5
# ms    = 0.05
# freq = 'Con_stat_cl-5_maxspread-0.05_theta_-1-TAIL' #spatVSbase

# freq = 'Theta_TWOTAIL_spatVSbase' #spatVSbase
astra = read_hdf5('Con_stat_cl-5_maxspread-0.05_beta_-1-TAIL') #.format(cluster_threshold, ms,freq))

connection_indices = astra.get('connection_indices')
con_clust = contrast[connection_indices]


##### UNILATERALITY
os.chdir(folder_with_files)   
labels = mne.read_labels_from_annot('fsaverage', 'aparc',  subjects_dir=data_path)

selected_label = mne.read_labels_from_annot(
    'fsaverage', 
    regexp='lh', 
    subjects_dir=data_path)

label_colors = [label.color for label in selected_label]

del labels[-1]  # drop 'unknown-lh' label
con_parc = con_clust.parcellate(selected_label, summary='absmax',
                                weight_by_degree=False)
#con_parc.save('Con_stat_cl-5_maxspread-.01_{}_prunned_to_contr-parcelled-1sign'.format(freqs))

##### VISUAL
# CIRCLE
# fig, axes = plt.subplots(1, 3, figsize=(8, 4), facecolor='black',
#                           subplot_kw=dict(polar=True))
# # con_parc.plot(n_lines=100, vmin=0, vmax=1, node_colors=label_colors, fontsize_names=10, fontsize_colorbar = 10, fig=fig)
# # plt.title('Strongest parcel-to-parcel connection', color='white')

# axes[1].plot(con_data = con_parc, n_lines=100, vmin=0, vmax=1, node_colors=label_colors, fontsize_names=10, fig=fig, fontsize_colorbar = 10, title='Strongest parcel-to-parcel connection')
# # axes[1].title('Strongest parcel-to-parcel connection', color='white')

# # no_names = [''] * len(label_colors)
# # for ax, method in zip(axes, con_parc):
# #     con_parc.plot(n_lines=100, vmin=0, vmax=1, node_colors=label_colors, fontsize_names=10, fontsize_colorbar = 10, fig=ax)
    
con_parc.plot(n_lines=15, vmin=0, vmax=0.5, node_colors=label_colors, fontsize_names=10, fontsize_colorbar = 10)
# plt.title('Strongest parcel-to-parcel connection', color='white')
# BRAIN
mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)   
                  
            
all_to_all = con_clust.make_stc('sum')                                    #'degree', weight_by_degree=True
brain = all_to_all.plot('fsaverage', subjects_dir=data_path, hemi='split',
                        figure=6, size=400, views=['medial', 'lateral','ventral'])
# mlab.title('All-to-all coherence (contrast)', height=0.9)
brain.add_annotation('aparc', borders=1)



##### SELECT PARTICULAR HEMI                                                                    
all_to_all = con_clust.make_stc('sum')
# labeled_stc = all_to_all.in_label(labels)
?brain = all_to_all.plot('fsaverage', subjects_dir=data_path, hemi='split',
                        figure=6, size=400, views=['dorsal', 'lateral','ventral'])

summary = np.sum(all_to_all.data)
brain.add_annotation('aparc', borders=1)