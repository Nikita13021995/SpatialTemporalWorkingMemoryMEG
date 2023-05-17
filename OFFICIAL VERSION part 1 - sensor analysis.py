  # -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:29:37 2022

@author: Nikita O
Project: Spatial and Temporal WM
Sensor analysis
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
import conpy
import sklearn

from   mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                                 corrmap)
from   mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch, csd_morlet
from   numpy.random import randn

from   mne.epochs import equalize_epoch_counts
from   mne.stats import (spatio_temporal_cluster_1samp_test,
                         summarize_clusters_stc, 
                         spatio_temporal_cluster_test,
                         permutation_cluster_test)
from   mne.minimum_norm import apply_inverse, read_inverse_operator
from   mne.datasets import sample
from   mne import read_source_estimate
from   mne.minimum_norm import read_inverse_operator
from   mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from   mne.cov import compute_covariance
from   mne.beamformer import (make_dics, apply_dics_csd, make_lcmv,
                            apply_lcmv_cov)
from   mne.minimum_norm import (make_inverse_operator, apply_inverse_cov)


from   fooof import FOOOF
from   fooof.sim.gen import gen_aperiodic
from   fooof.plts.spectra import plot_spectrum, plot_spectra
from   fooof.plts.annotate import plot_annotated_peak_search

from   scipy import linalg
from   scipy import stats as st
from   scipy import stats as stats

from   mne.io.constants import FIFF
from   mne.coreg import Coregistration
from   mne.io import read_info
from   mne.datasets import fetch_fsaverage

%matplotlib qt

# %% MERGING OF RECORDINGS

index = 24
### FILE 1 - restriction to gradiometers
SDF        = 'L:\SATURN\S24_Dmitry Cherneev\Process' 
SDRF       = os.path.join(SDF, 'S{}_Test1_tsss_mc_trans.fif'.format(index)) 
raw_data   = mne.io.read_raw_fif(SDRF, allow_maxshield=False, 
                                 preload=False, on_split_missing='raise', 
                                 verbose=None) 
info       = raw_data.info
raw_data.plot()
# picks      = mne.pick_types(info, meg = 'grad', eog=True, ecg=True, 
#                             stim=True, exclude=[]) 
# raw_data.save('S{}_Test1_tsss_mc_trans_grad_sti.fif'.format(index), 
#               picks = picks, overwrite=True)

### FILE 2 - restriction to gradiometers
SDF        = 'L:\SATURN\S24_Dmitry Cherneev\Process' 
SDRF       = os.path.join(SDF, 'S{}_Test2_tsss_mc_trans.fif'.format(index)) 
raw_data_2 = mne.io.read_raw_fif(SDRF, allow_maxshield=False, 
                                 preload=False, on_split_missing='raise', 
                                 verbose=None) 
info_2     = raw_data_2.info
raw_data_2.plot()
# picks_2    = mne.pick_types(info_2, meg = 'grad', eog=True, ecg=True, 
#                             stim=True, exclude=[]) 
# raw_data_2.save('S{}_Test2_tsss_mc_trans_grad_sti.fif'.format(index), 
#                 picks = picks_2, overwrite=True)

### FINDING THE FINAL SAMPLE OF 1ST 
raw_data.first_samp
# raw_data.plot()
# raw_data.info['sfreq']
# raw_data_time_of_first_sample   = raw_data.first_samp / raw_data.info['sfreq']
raw_data_2.first_samp
# raw_data_2.info['sfreq']
# raw_data_time_of_first_sample_2 = raw_data_2.first_samp / raw_data_2.info['sfreq'] 

events    = mne.find_events(raw_data, shortest_event=1, stim_channel='STI101') 
events_2  = mne.find_events(raw_data_2, shortest_event=1, stim_channel='STI101') 

raw_data.plot(events=events    )
raw_data_2.plot(events=events_2    )

# ### ALIGNING THE 
# events_3            = events.copy()
# y = 0
# while y < len(events_3[:,0]):
#     events_3[y, 0]  =  events_3[y, 0]  - raw_data.first_samp
#     y += 1

# raw_data.plot(events=events_3) 
# raw_data.info

# events_4 = events_2.copy()
# y = 0
# while y < len(events_4[:,0]):
#     events_4[y, 0]  =  events_4[y, 0]  - raw_data_2.first_samp
#     y += 1

# ### MERGER
# raw_data_full       = raw_data.copy()
# raw_data_full.append([raw_data_2])
# raw_data_full.first_samp
# events_full         = mne.find_events(raw_data_full, 
#                                      shortest_event=1, stim_channel='STI101') 

# events_f            = events_full.copy()
# y = 0
# while y             < len(events_f[:,0]):
#     events_f[y, 0]  =  events_f[y, 0]  - raw_data_full.first_samp
#     y              += 1
    
# np.save('S{}_full-events.npy'.format(index), events_f)

raw_data_full       = raw_data.copy()
raw_data_full.append([raw_data_2])
raw_data_full.first_samp
events_full         = mne.find_events(raw_data_full, 
                                      shortest_event=1, 
                                      stim_channel='STI101')
### EVENTS OF MERGED FILE
events = events_full.copy()
print(events[:,2])

events_f           = events.copy()
y = 0
while y            < len(events_f[:,0]):
    events_f[y, 0] =  events_f[y, 0]  - raw_data_full.first_samp
    y             += 1
np.save('S{}_full-events.npy'.format(index), events_full)

raw_data_full.plot(events = events_full)

# %% FILTERING 

raw = raw_data_full.copy()
raw.load_data() 

fig = raw.plot_psd(tmax=np.inf, fmax=80, average=True)

list = [    'MEG0113', 'MEG0112', 'MEG0122', 'MEG0123', 'MEG0132', 'MEG0133', 'MEG0143',
 'MEG0142', 'MEG0213', 'MEG0212', 'MEG0222', 'MEG0223', 'MEG0232', 'MEG0233', 'MEG0243',
 'MEG0242', 'MEG0313', 'MEG0312', 'MEG0322', 'MEG0323', 'MEG0333', 'MEG0332', 'MEG0343',
 'MEG0342', 'MEG0413', 'MEG0412', 'MEG0422', 'MEG0423', 'MEG0432', 'MEG0433', 'MEG0443',
 'MEG0442', 'MEG0513', 'MEG0512', 'MEG0523', 'MEG0522', 'MEG0532', 'MEG0533', 'MEG0542',
 'MEG0543', 'MEG0613', 'MEG0612', 'MEG0622', 'MEG0623', 'MEG0633', 'MEG0632', 'MEG0642',
 'MEG0643', 'MEG0713', 'MEG0712', 'MEG0723', 'MEG0722', 'MEG0733', 'MEG0732', 'MEG0743',
 'MEG0742', 'MEG0813', 'MEG0812', 'MEG0822', 'MEG0823', 'MEG0913', 'MEG0912', 'MEG0923',
 'MEG0922', 'MEG0932', 'MEG0933', 'MEG0942', 'MEG0943', 'MEG1013', 'MEG1012', 'MEG1023',
 'MEG1022', 'MEG1032', 'MEG1033', 'MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123',
 'MEG1122', 'MEG1133', 'MEG1132', 'MEG1142', 'MEG1143', 'MEG1213', 'MEG1212', 'MEG1223',
 'MEG1222', 'MEG1232', 'MEG1233', 'MEG1243', 'MEG1242', 'MEG1312', 'MEG1313', 'MEG1323',
 'MEG1322', 'MEG1333', 'MEG1332', 'MEG1342', 'MEG1343', 'MEG1412', 'MEG1413', 'MEG1423',
 'MEG1422', 'MEG1433', 'MEG1432', 'MEG1442', 'MEG1443', 'MEG1512', 'MEG1513',
 'MEG1522', 'MEG1523', 'MEG1533', 'MEG1532', 'MEG1543', 'MEG1542', 'MEG1613', 'MEG1612',
 'MEG1622', 'MEG1623', 'MEG1632', 'MEG1633', 'MEG1643', 'MEG1642', 'MEG1713', 'MEG1712',
 'MEG1722', 'MEG1723', 'MEG1732', 'MEG1733', 'MEG1743', 'MEG1742', 'MEG1813', 'MEG1812',
 'MEG1822', 'MEG1823', 'MEG1832', 'MEG1833', 'MEG1843', 'MEG1842', 'MEG1912', 'MEG1913',
 'MEG1923', 'MEG1922', 'MEG1932', 'MEG1933', 'MEG1943', 'MEG1942', 'MEG2013', 'MEG2012',
 'MEG2023', 'MEG2022', 'MEG2032', 'MEG2033', 'MEG2042', 'MEG2043', 'MEG2113', 'MEG2112',
 'MEG2122', 'MEG2123', 'MEG2133', 'MEG2132', 'MEG2143', 'MEG2142', 'MEG2212',
 'MEG2213', 'MEG2223', 'MEG2222', 'MEG2233', 'MEG2232', 'MEG2242', 'MEG2243', 'MEG2312',
 'MEG2313', 'MEG2323', 'MEG2322', 'MEG2332', 'MEG2333', 'MEG2343', 'MEG2342', 'MEG2412',
 'MEG2413', 'MEG2423', 'MEG2422', 'MEG2433', 'MEG2432', 'MEG2442', 'MEG2443', 'MEG2512',
 'MEG2513', 'MEG2522', 'MEG2523', 'MEG2533', 'MEG2532', 'MEG2543', 'MEG2542', 'MEG2612',
 'MEG2613', 'MEG2623', 'MEG2622', 'MEG2633', 'MEG2632', 'MEG2642', 'MEG2643']

### NOTCH AND BANDPASS FILTER
filt_raw         = raw.copy().filter(l_freq=1, h_freq=80) 
freq = [50]
raw_notch_fit    = filt_raw.notch_filter(
    freqs=freq, picks=list, method='spectrum_fit', filter_length='10s')
fig              = raw_notch_fit.plot_psd(fmax=80, average=True)
raw_notch_fit.plot()

raw_2            = raw_notch_fit.copy()
raw_2.plot(events=events)
raw_2.annotations.save('S{}_full-annotations.csv'.format(index), overwrite = False)


### PREPERATION TO ICA
# raw = raw_2.copy()
# os.chdir('L:/САТУРН/ABRAR WM/mne working memory study _ working files/ANNOTATIONS')
# annot_full = mne.read_annotations('S{}_full-annotations.csv'.format(index))
# raw.set_annotations(annot_full)
# raw.plot(events=events_full)

### ICA 1
ica              = mne.preprocessing.ICA(n_components=40, random_state=97, max_iter=800)
ica_filt_raw     = ica.fit(raw_2)  
ica.plot_sources(raw, show_scrollbars=False)
ica.plot_components(sensors = True, colorbar = True, outlines = 'skirt')
raw_4            = raw

### ICA 2 with EOG and ECG
ica.exclude             = []
eog_indices, eog_scores = ica.find_bads_eog(raw_4)
ica.exclude             = eog_indices
# ica.plot_scores(eog_scores, exclude=eog_indices)
# ica.plot_properties(raw_4, picks=eog_indices)
# ica.plot_sources(eog_evoked)


ecg_indices, ecg_scores = ica.find_bads_ecg(raw_4, 
                                            method='correlation', 
                                            threshold='auto')
# ica.plot_scores(ecg_scores, exclude=ecg_indices)
# ica.plot_properties(raw_4, picks=ecg_indices)
# ica.plot_sources(ecg_evoked)

#MERGING ECG and EOG
ica.plot_sources(raw_4, show_scrollbars=False)
ica.exclude= ica.exclude + ecg_indices

ica.exclude             = [0, 5, 18, 20]
ica.save('S{}_full-ica.fif'.format(index))

ica                     = mne.preprocessing.read_ica('S{}_full-ica.fif'.format(index), 
                                                     verbose=None)
raw_5 = raw_4.copy()
ica.apply(raw_5)
raw_4.plot()
raw_5.plot()

### FILTERING OUTCOME SAVER
os.chdir('L:/САТУРН/ABRAR WM/mne working memory study _ working files/FULL DATA')
info = raw_5.info
picks      = mne.pick_types(info, meg = 'grad', eog=True, ecg=True, 
                            stim=True, exclude=[]) 

raw_5.save('S{}_full_reconst.fif'.format(index), picks = picks, overwrite=False)

### CLEANING RAM
del a, ecg_evoked, eog_evoked, fig, filt_raw, freq, i, 
del ica, raw, raw_2, raw_3, raw_4, raw_data, 
del raw_data_2, raw_data_full, raw_notch_fit, y

# %% UPLOADER

index = 15
SDF                = 'L:/SATURN/ABRAR WM/mne working memory study _ working files/FULL DATA' #Firstly we put in a DIRECT way a directory with the file
SDF                = 'L:/SATURN/ABRAR WM/JAN'
SDRF               = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index)) #Then we show the file in this directory
raw_5              = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None) #Then we upload THE FILE from the directory


# %% VISUAL REJECTION and EPOCHing

events             = mne.find_events(raw_5, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?
events_total       = events.copy()

### EVENT ALIGHNER - not necessary
# len(events_total)
# y = 0
# while y < len(events_total[:,0]):
#     events_total[y, 0]        =  events_total[y, 0]  - raw_5.first_samp
#     y += 1
    
### EVENT NAME CHANGER    
a = 0
i = 0

while i < len(events_total[:,2]):     
    if events_total[i,2]      == 155:
        a = i    
        if  events_total[a-1, 2] > 180 and events_total[a+1, 2] > 180:
            events_total[a,2] = 255
        else: 
            events_total[a,2] = 155
    i += 1 

### VISUAL REJECTION
# In order to be rejected in EPOCHS creation, the annotation group name 
# should start from BAD_

raw_5.plot(events = events_total, event_color={100: 'g', 200: 'g', 
                        101: 'g', 102: 'g',                                    128: 'r', 208: 'r', 152: 'r',  
                        155: 'g', 110: 'g', 120: 'm', 104: 'g',
                        201: 'g', 202: 'g', 203: 'g', 204: 'g',
                        210: 'g', 220: 'm', 255: 'g', 103: 'g',
                        105: 'g', 205: 'g'})

os.chdir('L:\SATURN\ABRAR WM\JAN') 
raw_5.save('S{}_full_reconst_filt.fif'.format(index), overwrite=True)
raw_5.annotations.save('S{}_fullreconst_filt-annotations.csv'.format(index), overwrite = True)

event_dict = {'Instruction/Spatial': 100, ' Instruction/Temporal': 200, 
              'Stimulus1/Spatial':   101, 'Stimulus2/Spatial':     102, 
              'Stimulus3/Spatial':   103, 'Stimulus4/Spatial':     104, 
              'Delay/Spatial':       155, 'Probe/Spatial':         105, 
              'CorResp/Spatial':     110, 'InCorResp/Spatial':     120, 
              'Stimulus1/Temporal':  201, 'Stimulus2/Temporal':    202, 
              'Stimulus3/Temporal':  203, 'Stimulus4/Temporal':    204, 
              'Probe/Temporal':      205, 'CorResp/Temporal':      210, 
              'InCorResp/Temporal':  220, 'Delay/Temporal':        255}
# raw_5.plot(events=events_total, color='gray',
#            event_color={100: 'r', 200: 'r', 101: 'r', 102: 'r', 
#                         155: 'r', 110: 'g', 120: 'm', 104: 'r',
#                         201: 'r', 202: 'r', 203: 'r', 204: 'r',
#                         210: 'g', 220: 'm', 255: 'r', 103: 'r',
#                         105: 'r', 205: 'r'})

### REJECT CRITERIA
reject_criteria    = dict(grad=3000e-13)    # 3000 fT/cm
flat_criteria      = dict(grad=1e-13)         # 1 fT/cm
# raw_5.plot()
# interactive_annot = raw_5.annotations

### CREATING EPOCHS
epochs_s           = mne.Epochs(raw_5, events_total, event_id=155,             #155 - Delay Spatial
                                tmin=-8, tmax=8, reject = reject_criteria, 
                                flat=flat_criteria, preload=True)
epochs_t           = mne.Epochs(raw_5, events_total, event_id=255,             #255 - Delay Temporal
                                tmin=-8, tmax=8, reject = reject_criteria, 
                                flat=flat_criteria, preload=True)


### SAVER
epochs_s.save( 'S{}_CorAns_full_S_epochs-epo.fif'.format(index),  overwrite=True)
epochs_t.save( 'S{}_CorAns_full_T_epochs-epo.fif'.format(index),  overwrite=True)

epochs_t.plot()

### CLEANING RAM
del raw_5
del epochs_s
del epochs_t

# %% POWER 

index_array_2  = [3,6,7,9,10, 11, 14,15, 18,19,20,21,22,23,24,25,26,27,28] 
epochs_s_list  = [3,6,7,9,10, 11, 14,15, 18,19,20,21,22,23,24,25,26,27,28]
epochs_t_list  = [3,6,7,9,10, 11, 14,15, 18,19,20,21,22,23,24,25,26,27,28]

### UPLOADER
os.chdir('L:\SATURN\ABRAR WM\JAN') 

# i = 0
# for i in index_array_2:  
#     epochs_s_list[i] = mne.read_epochs('S{}_CorAns_full_S_epochs-epo.fif'.format(i), preload=False)
#     epochs_t_list[i] = mne.read_epochs('S{}_CorAns_full_T_epochs-epo.fif'.format(i), preload=False)


epochs_s_28 = mne.read_epochs('S28_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_28 = mne.read_epochs('S28_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_27 = mne.read_epochs('S27_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_27 = mne.read_epochs('S27_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_26 = mne.read_epochs('S26_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_26 = mne.read_epochs('S26_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_25 = mne.read_epochs('S25_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_25 = mne.read_epochs('S25_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_24 = mne.read_epochs('S24_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_24 = mne.read_epochs('S24_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_23 = mne.read_epochs('S23_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_23 = mne.read_epochs('S23_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_22 = mne.read_epochs('S22_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_22 = mne.read_epochs('S22_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_21 = mne.read_epochs('S21_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_21 = mne.read_epochs('S21_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_20 = mne.read_epochs('S20_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_20 = mne.read_epochs('S20_CorAns_full_T_epochs-epo.fif', preload=False)

epochs_s_19 = mne.read_epochs('S19_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_19 = mne.read_epochs('S19_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_18 = mne.read_epochs('S18_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_18 = mne.read_epochs('S18_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_15 = mne.read_epochs('S15_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_15 = mne.read_epochs('S15_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_14 = mne.read_epochs('S14_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_14 = mne.read_epochs('S14_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_11 = mne.read_epochs('S11_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_11 = mne.read_epochs('S11_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_10 = mne.read_epochs('S10_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_10 = mne.read_epochs('S10_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_9  = mne.read_epochs('S9_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_9  = mne.read_epochs('S9_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_7  = mne.read_epochs('S7_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_7  = mne.read_epochs('S7_CorAns_full_T_epochs-epo.fif', preload=False)
epochs_s_6  = mne.read_epochs('S6_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_6  = mne.read_epochs('S6_CorAns_full_T_epochs-epo.fif', preload=False)
#epochs_s_4 = mne.read_epochs('S4_full_S_epochs-epo.fif', preload=False)
#epochs_t_4 = mne.read_epochs('S4_full_T_epochs-epo.fif', preload=False)
epochs_s_3  = mne.read_epochs('S3_CorAns_full_S_epochs-epo.fif', preload=False)
epochs_t_3  = mne.read_epochs('S3_CorAns_full_T_epochs-epo.fif', preload=False)

### MERGER
epoch_s_list = [epochs_s_3,  epochs_s_6,  epochs_s_7,  epochs_s_9,  epochs_s_10,
                epochs_s_11, epochs_s_14, epochs_s_15, epochs_s_18, epochs_s_19,
                epochs_s_20, epochs_s_21, epochs_s_22, epochs_s_23, epochs_s_24,
                epochs_s_25, epochs_s_26, epochs_s_27, epochs_s_28]
epoch_t_list = [epochs_t_3,  epochs_t_6,  epochs_t_7,  epochs_t_9,  epochs_t_10,
                epochs_t_11, epochs_t_14, epochs_t_15, epochs_t_18, epochs_t_19,
                epochs_t_20, epochs_t_21, epochs_t_22, epochs_t_23, epochs_t_24,
                epochs_t_25, epochs_t_26, epochs_t_27, epochs_t_28]
### EVOKED
i = 0
for i in range(19): 
    epochs_s = epoch_s_list[i]
    epochs_t = epoch_t_list[i]

    spat_evoked = epochs_s.average()
    temp_evoked = epochs_t.average()

    fig_s = spat_evoked.plot(titles = 'Evoked spatial  data of sub{}'.format(i))
    fig_t = temp_evoked.plot(titles = 'Evoked temporal data of sub{}'.format(i))
    
    fig_s.savefig('Evoked spatial  data of sub{}'.format(i)) 
    fig_t.savefig('Evoked temporal data of sub{}'.format(i))

### POWER SPECTRAL DENSITY
#Two plots in one figure
i = 0
for i in range(19): 
    epochs_s = epoch_s_list[i]
    epochs_t = epoch_t_list[i]

    fig, ax = plt.subplots(2)
    epochs_s.plot_psd(tmin=0, tmax=4, fmin = 1, fmax=80, average=True, ax=ax[0], normalization = 'length') #full / spatial_color = True
    ax[0].set_title('PSD for spatial information of participant  {}'.format(i))
    ax[0].set_ylabel('(fT/cm)^2/Hz (dB)')
    epochs_t.plot_psd(tmin=0, tmax=4, fmin = 1, fmax=80, average=True, ax=ax[1], normalization = 'length') #full / spatial_color = True
    ax[1].set_title('PSD for temporal information of participant  {}'.format(i))
    ax[1].set_ylabel('(fT/cm)^2/Hz (dB)')
    ax[1].set_xlabel('Frequency (Hz)')
    fig.set_tight_layout(True)
    fig.savefig('PSD T&S for sub{}'.format(i)) 
    #fig.set_figwidth(40)
    #fig.set_figheight(40)
    plt.show()

    #Two plots merged
    gs = gridspec.GridSpec(2,1)
    plt.figure()
    ax = plt.axes()
    epochs_s.plot_psd(ax=ax, tmin=0, tmax=4, fmin = 1, fmax=80,  average=True, color= 'r')
    epochs_t.plot_psd(ax=ax, tmin=0, tmax=4, fmin = 1, fmax=80,  average=True, color= 'b')
    plt.show()
    plt.title('PSD general sub{}'.format(i))

    

### TIME- FREQUENCY ANALYSIS
a = np.log10(4)
b = np.log10(80)
frequencies = np.logspace(a,b, num=30)

i = 0
for i in range(19): 
    epochs_s = epoch_s_list[i]
    epochs_t = epoch_t_list[i]

    power_s, itc_s = mne.time_frequency.tfr_morlet(epochs_s, n_cycles=5, return_itc=True,
                                              freqs=frequencies, decim=20, n_jobs=-1)
    power_t, itc_t = mne.time_frequency.tfr_morlet(epochs_t, n_cycles=5, return_itc=True,
                                              freqs=frequencies, decim=20, n_jobs=-1)
    
    power_s.save('S{}_nov_power_s-tfr.h5'.format(i))
    power_t.save('S{}_nov_power_t-tfr.h5'.format(i))
    itc_s.save('S{}_nov_itc_s-tfr.h5'.format(i))
    itc_t.save('S{}_nov_itc_t-tfr.h5'.format(i))
    spat_evoked.save('S{}_nov_evoked_s-ave.fif'.format(i))
    temp_evoked.save('S{}_nov_evoked_t-ave.fif'.format(i))

# %% VISUAL CHECK

power_s_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
power_t_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

os.chdir('L:\SATURN\ABRAR WM\JAN') 

###### UPLOADER
i = 0
for i in range(19): 
    power_s = mne.time_frequency.read_tfrs('S{}_nov_power_s-tfr.h5'.format(i))
    power_s_list[i] = power_s[0]
    power_t = mne.time_frequency.read_tfrs('S{}_nov_power_t-tfr.h5'.format(i))
    power_t_list[i] = power_t[0]
    
###### PLOTING
#ORIGINAL
i = 0 
for i in range(len(power_s_list)):
    power_s_list[i].plot(combine='mean',title = 'subject S {}'.format(i))
    power_t_list[i].plot(combine='mean',title = 'subject T {}'.format(i))
    

    
# WITH TOPOPLOT
i = 0 
for i in range(len(power_s_list)):
    power_s_list[i].plot_joint(title = 'subject joined_plot S {}'.format(i))
    power_t_list[i].plot_joint(title = 'subject joined_plot T {}'.format(i))

# TOPOPLOT 
i = 0 
for i in range(len(power_s_list)):
    power_s_list[i].plot_topo(baseline=(None,None), mode = 'logratio', title ='topo_plot average power spatial Sub{}'.format(i))
    power_t_list[i].plot_topo(baseline=(None,None), mode = 'logratio', title ='topo_plot average power temporal Sub{}'.format(i))
    
#### GRAND AVERAGE TOPOPLOT
grand_average_s = mne.grand_average(power_s_list,interpolate_bads=True, drop_bads=True)
grand_average_s.plot_topomap(baseline=(None,None), mode = 'logratio', title ='Grang average SPATIAL')
grand_average_t = mne.grand_average(power_t_list,interpolate_bads=True, drop_bads=True)
grand_average_t.plot_topomap(baseline=(None,None), mode = 'logratio', title ='Grang average TEMPORAL')





# %% FOOOF separation of signal components

power_s_data = power_s_list
power_t_data = power_t_list

###### CROPPING
i = 0 
for i in range(len(power_t_list)):
    power_t_list[i].crop(0,4)
    power_s_list[i].crop(0,4)


###### PREPARATION
subj = 0
ch = 0

fm = FOOOF()
list_s_ped_ordered =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
len(list_s_ped_ordered)
list_s_aper_ordered =  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
len(list_s_aper_ordered)
list_t_ped_ordered =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
len(list_t_ped_ordered)
list_t_aper_ordered =  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
len(list_t_aper_ordered)

###### FOOOF splitter
########## SPATIAL CONDITION

for subj in range(len(power_s_data)):    
    
    #PREPARATORY PHASE
    spectrum_peak = np.array([])                                                #Here we store everything for 204 channel in 1 subject
    spectrum_aper = np.array([])
    power_s = power_s_list[subj]
    c = power_s.freqs
    a = c 
  
    #AVERAGING ACROSS TIME DIM
    b = power_s.data
    a.shape
    b.shape
    f = np.mean(b, axis=2)                                                      ### This is exactly the place when we lose time dimension
    type(f)
    f.shape
    spectrum = f
    np.shape(spectrum)
       
    for ch in np.arange(204):       
        spec = spectrum[ch, :]
        spec.shape
        fm.fit(a, spec)
        fm.save('FOOOF_sub{}_nov_сropped_results_S_{}'.format(subj, i), save_results=True, save_settings=True, save_data=True)
        init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
        type(init_ap_fit)
        np.shape(init_ap_fit)
        init_flat_spec = fm.power_spectrum - init_ap_fit
        type(init_flat_spec)
        np.shape(init_flat_spec)

        spectrum_peak = np.append(spectrum_peak, init_flat_spec.T)
        spectrum_aper = np.append(spectrum_aper, init_ap_fit.T)
        spectrum_peak.size
        spectrum_aper.size
        spectrum_peak.shape
    
        spectrum_peak = np.reshape(spectrum_peak,[ch+1,len(freqs)]) #( 204, 30) for one subject
        spectrum_aper = np.reshape(spectrum_aper,[ch+1,len(freqs)]) #( 204, 30) for one subject
        spectrum_peak.shape
        spectrum_aper.shape

        ch += 1
    list_s_ped_ordered[subj] = spectrum_peak
    list_s_aper_ordered[subj] = spectrum_aper
    subj += 1

########## TEMPORAL CONDITION
for subj in range(len(power_t_data)):    
    spectrum_peak_t = np.array([])    #Here we story everything for 204 channel in 1 subject
    spectrum_aper_t = np.array([])
    power_t = power_t_list[subj]
    c = power_t.freqs
    a = c
    b = power_t.data
    a.shape
    b.shape
    f = np.mean(b, axis=2)
    type(f)
    f.shape
    freqs = a 
    freqs.shape  
    spectrum = f          
    
    for ch in np.arange(204):       
        spec = spectrum[ch, :]
        spec.shape
        fm.fit(freqs, spec)
        fm.save('FOOOF_sub{}_nov__сropped_results_T_{}'.format(subj, i), save_results=True, save_settings=True, save_data=True)
        init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
        type(init_ap_fit)
        np.shape(init_ap_fit)
        init_flat_spec = fm.power_spectrum - init_ap_fit
        type(init_flat_spec)
        np.shape(init_flat_spec)

        spectrum_peak_t = np.append(spectrum_peak_t, init_flat_spec.T)
        spectrum_aper_t = np.append(spectrum_aper_t, init_ap_fit.T)
        spectrum_peak_t.size
        spectrum_aper_t.size
        spectrum_peak_t.shape
    
        spectrum_peak_t = np.reshape(spectrum_peak_t,[ch+1,len(freqs)]) #( 204, 30) for one subject
        spectrum_aper_t = np.reshape(spectrum_aper_t,[ch+1,len(freqs)]) #( 204, 30) for one subject
        spectrum_peak_t.shape
        spectrum_aper_t.shape

        ch += 1
    list_t_ped_ordered[subj] = spectrum_peak_t
    list_t_aper_ordered[subj] = spectrum_aper_t
    subj += 1

###### SAVER
a = np.array(list_s_ped_ordered)
type(a)
a.shape
b = np.array(list_s_aper_ordered)
type(b)
b.shape
c = np.array(list_t_ped_ordered)
type(c)
c.shape
d = np.array(list_t_aper_ordered)
type(d)
d.shape

np.save(file='list_nov_s_ped_crop.npy', arr=a)
np.save(file='list_nov_s_aper_crop.npy', arr=b)
np.save(file='list_nov_t_ped_crop.npy', arr=c)
np.save(file='list_nov_t_aper_crop.npy', arr=d)

# %% FOOOF VISUAL CHECKER

#### PREPARATION
freq_range    = [4, 80] #!!!! AGAIN 4 instead of 1
plt_log       = False

list_s_ped    = list_s_ped_ordered
list_s_aper   = list_s_aper_ordered
list_t_ped    = list_t_ped_ordered
list_t_aper   = list_t_aper_ordered

#### INDIVIDUAL PLOT
subj          = 1
power_t       = power_t_list[subj]
a             = list_t_ped[1]
freqs         = power_t.freqs
plot_spectrum(fm.freqs, a.T, plt_log)

#### ALL SUBJECTS ALL CHANNEL PLOT
for subj in range(len(power_t_data)):
    power_t   = power_t_list[subj]
    freqs     = power_t.freqs
    a         = list_t_ped[subj]
    plot_spectrum(fm.freqs, a.T, plt_log)
    subj     +=1

### PARTICULAR SUBJECT EACH CHANNEL 
subj          = 1
n_channels    = 15                                                              # Change here the number of channels
power_t       = power_t_list[subj]
freqs         = power_t.freqs

b = list_t_ped[subj]
for ch in np.arange(n_channels):  
    a         = b[ch,:]
    plot_spectrum(fm.freqs, a.T, plt_log)
    ch       +=1

#### PARTICULAR SUBJECT PARTICULAR CHANNEL 
len(power_t_data)

for subj in range(2):
    power_t   = power_t_list[subj]
    freqs     = power_t.freqs
    b         = list_t_ped[subj]

    for ch in np.arange(10):  
        a     = b[ch,:]
        plot_spectrum(fm.freqs, a.T, plt_log)
        ch   +=1

    subj     +=1
    
#### COMBINED POWER SPECTRA FOR BOTH CONDITIONS
subj          = 1
for subj in range(len(power_t_data)):
    
    power_s   = power_s_list[subj]
    power_t   = power_t_list[subj]
    freqs     = power_t.freqs
    freqs.shape

    a         = list_t_ped[subj]
    b         = list_s_ped[subj]
    a.shape
    b.shape
    p_s       = np.mean(a, axis=0)
    p_t       = np.mean(b, axis=0)
    p_s.shape
    p_t.shape

    labels = ['p_s', 'p_t']
    plot_spectra(freqs, [p_t, p_s], log_powers=False,  labels=labels)

# %% STATISTICS

#### UPLOADER
os.chdir('L:\SATURN\ABRAR WM\JAN') 
list_s_ped_np                          = np.load(file='list_nov_s_ped_crop.npy')  
list_s_aper_np                         = np.load(file='list_nov_s_aper_crop.npy')
list_t_ped_np                          = np.load(file='list_nov_t_ped_crop.npy')
list_t_aper_np                         = np.load(file='list_nov_t_aper_crop.npy')

#### ADJACENCY MATRIX
os.chdir('L:\SATURN\ABRAR WM\JAN') 
epochs_s_28                            = mne.read_epochs('S28_CorAns_full_S_epochs-epo.fif', preload=False)
epochs                                 = epochs_s_28
info                                   = epochs.info
adj, ch_names                          = mne.channels.find_ch_adjacency(info, 
                                                                        ch_type= 'grad')

#### DATA PREPARATION
psa                                    = list_s_ped_np #NOW THE DATA AFTER NORMALIZATION TAKES SHAPE - SUB, CH, n_freqs, n_times 
pta                                    = list_t_ped_np # 19 for (204, 30)

psa.shape
pta.shape
a_s                                    = np.transpose(psa, (0,2,1))
a_t                                    = np.transpose(pta, (0,2,1))
a_s.shape
a_t.shape

#### STATISTICS
t_thresh = scipy.stats.t.ppf(1 - 0.05 / 2, df=18)
threshold                              = 3.0
n_permutations                         = 5000
obj                                    = a_t - a_s
T_obs, clusters, cluster_p_values, H0  = mne.stats.spatio_temporal_cluster_1samp_test(obj, #Non-parametric cluster-level paired t-test for spatio-temporal data.
                                                 out_type='mask', adjacency=adj, 
                                                 n_permutations=n_permutations,
                                                 threshold=threshold, tail=0)

#### STAT OUTCOMES FEATURES
type(T_obs) 
type(clusters)
type(cluster_p_values) 
type(H0)
np.shape(T_obs) 
np.shape(clusters)
np.shape(cluster_p_values) 
np.shape(H0)

#### VISUALIZATION
plt.imshow(T_obs, aspect='auto', origin='lower', cmap='cividis', vmin=np.min(T_obs), vmax=np.max(T_obs)) #, extent=[1,204,4,80]) #freq not just linear, but as index
x_label_list = ['4', '6.05', '10.14', '16.99', '28.47', '47.73', '80']
plt.yticks([0,5,10,15,20, 25, 29], x_label_list )
plt.colorbar()

####### PLOTING MULTI CHANNELS
obj.shape
plt.plot(a_t[:,:,1].T) #Channel representation over frequency 
plt.plot(obj[:,:,1].T) #Channel representation over frequency 
# freqs = power_ls[1].freqs
# plt.plot([a_s[:,:,1].T, a_t[:,:,1].T], freqs)  #TWO channel representations over frequency 
plt.show()


####### CHECKING WITH SIGNIFICANT LEVEL
p_accept                              = 0.05
good_cluster_inds                     = np.where(cluster_p_values < p_accept)[0]
print(good_cluster_inds)
len(good_cluster_inds)
print(cluster_p_values)

####### PLOTING SIGNIFICANT CLUSTERS
T_obs_plot                            = 0  * np.ones_like(T_obs)
for c, p_val                          in zip(clusters, cluster_p_values):
    if p_val                          <= p_accept:
         T_obs_plot[c]                = T_obs[c]
         
plt.imshow(T_obs, aspect='auto', origin='lower', cmap='cividis', vmin=-5, vmax=5) #, extent=[1,204,4,80]) #freq not just linear, but as index
plt.imshow(T_obs_plot, aspect='auto', origin='lower', cmap='cool', vmin=-5, vmax=5) #, extent=[1,204,4,80]) #freq not just linear, but as index
# plt.colorbar(label = 'T values of condition comparison')
x_label_list = ['4', '6.05', '10.14', '16.99', '28.47', '47.73', '80']
plt.yticks([0,5,10,15,20, 25, 29], x_label_list )
plt.colorbar(label = 'T values in significant clusters (p<0.05)')
plt.show()

plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Induced power')

# %% INVESTIGATION THE DIFFERENCE BETWEEN CONDITIONS

#### DATA PREPARATION
a_t                    = list_t_ped_np
a_s                    = list_s_ped_np
a_t.shape 
a_s.shape 
a_s                    = np.transpose(a_s, (0,2,1))
a_t                    = np.transpose(a_t, (0,2,1))
obj                    = a_t - a_s

########################## DATA PREPARATION
ch1                    = 110                                                     # here we select particular channels
ch2                    = 125
ch3                    = 135
ch4                    = 160

#SPATIAL BLOCK
interest_ch_freq_s_1   = a_s[:,:,ch1:ch2]
interest_ch_freq_s_2   = a_s[:,:,ch3:ch4]
interest_ch_freq_s_1.shape
interest_ch_freq_s_2.shape
interest_ch_freq_s_1   = np.append(interest_ch_freq_s_1, interest_ch_freq_s_2, axis = 2)
interest_ch_freq_s_1.shape
interest_ch_freq_s     = interest_ch_freq_s_1 

#TEMPORAL BLOCK                                                                 
interest_ch_freq_t_1   = a_t[:,:,ch1:ch2]
interest_ch_freq_t_2   = a_t[:,:,ch3:ch4]
interest_ch_freq_t_1.shape
interest_ch_freq_t_2.shape
interest_ch_freq_t_1   = np.append(interest_ch_freq_t_1, interest_ch_freq_t_2, axis = 2)
interest_ch_freq_t_1.shape
interest_ch_freq_t     = interest_ch_freq_t_1 

#AVERAGE BLOCK
interest_ch_freq_obj_1 = obj[:,:,ch1:ch2]
interest_ch_freq_obj_2 = obj[:,:,ch3:ch4]
interest_ch_freq_obj_1.shape
interest_ch_freq_obj_2.shape
interest_ch_freq_obj_1 = np.append(interest_ch_freq_obj_1, interest_ch_freq_obj_2, axis = 2)
interest_ch_freq_obj_1.shape
interest_ch_freq_obj   = interest_ch_freq_obj_1 

### PLOTTING - power per channel
index = 0
for index in range(19):
    gs = gridspec.GridSpec(3,1)
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('channels')
    ax.set_ylabel('Power')
    ax.set_title('Power across frequency for channels of interest in subject {}'.format(index))
    plt.plot(interest_ch_interestfreq_s[index,  :,1], color='r', label='Spatial')
    plt.plot(interest_ch_interestfreq_t[index,  :,1], color='b', label='Temporal')
    plt.plot(interest_ch_interestfreq_obj[index,:,1], color='g', label='Difference')
    ax.legend()
    plt.show()
    index+=1  

#### FREQUENCY OF INTEREST
interest_ch_freq_s.shape
interest_ch_freq_s.shape     == interest_ch_freq_t.shape
interest_ch_freq_s.shape     == interest_ch_freq_obj.shape
interest_ch_interestfreq_s   = interest_ch_freq_s[:,9:15,:]
interest_ch_interestfreq_t   = interest_ch_freq_t[:,9:15,:]
interest_ch_interestfreq_obj = interest_ch_freq_obj[:,9:15,:]
interest_ch_interestfreq_s.shape

#### PLOTTING  - channel of interest over frequency of interest
fig, ax                      = plt.subplots(3,1)
ax[2].set_xlabel('channel index')
ax[0].set_title('temporal')
ax[1].set_title('spatial')
ax[2].set_title('diff')
ax[0].set_ylabel('Power')
ax[1].set_ylabel('Power')
ax[2].set_ylabel('Power')
ax[0].plot(interest_ch_interestfreq_t[1,:,1]) 
ax[1].plot(interest_ch_interestfreq_s[1,:,1]) 
ax[2].plot(interest_ch_interestfreq_obj[1,:,1]) 

#### PLOTTING - differences for conditions for subjects
index = 0
for index in range(19):
    gs = gridspec.GridSpec(3,1)
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('freq ind')                                                   # WE CAN CHANGE: channel or frequency
    ax.set_ylabel('Power')
    ax.set_title('Power across frequency in averaged channels of interest in subject {}'.format(index))
    
    interest_s = np.mean(interest_ch_interestfreq_s, axis = 2)                  # for channel - 1; for freq - 2
    interest_t = np.mean(interest_ch_interestfreq_t, axis = 2)
    # interest_obj = np.mean(interest_ch_interestfreq_obj, axis = 1)
    
    interest_s = np.mean(interest_ch_interestfreq_s, axis = 1)                  # for channel - 1; for freq - 2
    interest_t = np.mean(interest_ch_interestfreq_t, axis = 1)
    
    plt.plot(interest_s[index,:], color='r', label='Spatial')                   #You need to change here the place of index across axis
    plt.plot(interest_t[index,:], color='b', label='Temporal')
    # plt.plot(interest_ch_interestfreq_obj[index,:,1], color='g', label='Difference')
    ax.legend()
    plt.show()

#### PLOTTING - averaged across subjects
interest_s      = np.mean(interest_ch_interestfreq_s, axis = 0)                  # for channel - 1; for freq - 2
interest_t      = np.mean(interest_ch_interestfreq_t, axis = 0)
interest_obj    = np.mean(interest_ch_interestfreq_obj, axis = 0)

index          = 0
interest_ch_interestfreq_s.shape
freq           = 6 
chan           = 40
for index in range(freq):
    plt.figure()
    ax = plt.axes()
    ax.set_xlabel('freq ind')                                                   # WE CAN CHANGE: channel or frequency
    ax.set_ylabel('Power')
    ax.set_title('Power across frequency in averaged channels of interest in subject {}'.format(index))
    # plt.plot(interest_s[index,:], color='r', label='Spatial')
    # plt.plot(interest_t[index,:], color='b', label='Temporal')
    # plt.plot(interest_ch_interestfreq_obj[index,:], color='g', label='Difference')
    plt.plot(interest_s[:,index], color='r', label='Spatial')
    plt.plot(interest_t[:,index], color='b', label='Temporal')
    plt.plot(interest_ch_interestfreq_obj[:,index], color='g', label='Difference')
    ax.legend()
    plt.show()



# %% GLM MODEL - CORRELATION WITH ACCURACY

#### RESULTS 
df             = pd.read_excel('L:\SATURN\ABRAR WM\mne working memory study _ working files\Results_4.xlsx')
print(df)
type(df)
results        = df.to_numpy()
type(results)

Accuracy_T     = results[:,2]
Accuracy_S     = results[:,1]
Accuracy_full  = np.append(Accuracy_T,Accuracy_S)

os.chdir('L:\SATURN\ABRAR WM\JAN') 
np.save(file   ='Accuracy_T.npy', arr=Accuracy_T)
np.save(file   ='Accuracy_S.npy', arr=Accuracy_S)
np.save(file   ='Accuracy_FULL.npy', arr=Accuracy_full)

#### LOADER
os.chdir('L:\SATURN\ABRAR WM\JAN') 
Accuracy_S     = np.load(file='Accuracy_S.npy', allow_pickle=True)
Accuracy_T     = np.load(file='Accuracy_T.npy', allow_pickle=True)
Accuracy_Full  = np.load(file='Accuracy_FULL.npy', allow_pickle=True)
list_s         = np.load(file='list_nov_s_ped.npy')
list_t         = np.load(file='list_nov_t_ped.npy')
list_full      = np.append(list_t, list_s, axis = 0)

#### CREATING GLM
list_full.shape #TF(ch,freq, subj)
Accuracy_Full.shape  #Acc(subj)
chan           = 204
frequens       = 30

w, h          = 204, 30
S_coef        = [[0 for x in range(w)] for y in range(h)] 
S_pav         = [[0 for x in range(w)] for y in range(h)] 
T_coef        = [[0 for x in range(w)] for y in range(h)] 
T_pav         = [[0 for x in range(w)] for y in range(h)] 

#### SPATIAL CONDITION
ch            = 1
freq          = 1
for freq in range(frequens):   
    for ch in range(chan):           
        S_coef[freq][ch], S_pav[freq][ch] = st.spearmanr(a = list_s[:, ch, freq], b = Accuracy_S[:], axis=0, nan_policy='propagate', alternative='two-sided')        
        ch+=1
    freq+=1

#### TEMPORAL CONDITION
ch            = 1
freq          = 1
for freq in range(frequens):   
    for ch in range(chan):           
        T_coef[freq][ch],  T_pav[freq][ch] = st.spearmanr(a = list_t[:, ch, freq], b = Accuracy_T[:], axis=0, nan_policy='propagate', alternative='two-sided')        
        ch+=1
    freq+=1
    
########## PLOTING
#### ORIGINAL PICTURE
R             = S_coef
P             = S_pav
p_accept      = 0.05

fig, ax       = plt.subplots(2,1)
pic1          = ax[0].imshow(R, cmap=plt.cm.RdBu_r)
cbar          = fig.colorbar(pic1, aspect=5)
pic2          = ax[1].imshow(P)
cbar          = fig.colorbar(pic2, aspect=5)
x_label_list  = [' 6.05', '10.14', '16.99', '28.47', '47.73', '80']
ax[0].set_yticks([5,10,15,20, 25, 30])
ax[0].set_yticklabels(x_label_list)
y_label_list  = [' 6.05', '10.14', '16.99', '28.47', '47.73', '80']
ax[1].set_yticks([5,10,15,20, 25, 30])
ax[1].set_yticks([5,10,15,20, 25, 30])
ax[1].set_yticklabels(y_label_list)
ax[0].set_title( 'Correlation coefficient')
ax[1].set_title( 'P-value')
ax[1].set_xlabel('Channels')
ax[1].set_ylabel('frequency')
ax[0].set_xlabel('Channels')
ax[0].set_ylabel('frequency')


#### MASKING T_values with P_values 
type(T_coef)
type(T_pav)
np.shape(T_coef)
np.shape(T_pav)
T_coef_np                   = np.array(T_coef)
T_pav_np                    = np.array(T_pav)
S_coef_np                   = np.array(S_coef)
S_pav_np                    = np.array(S_pav)
type(T_coef_np)
type(T_pav_np)
T_pav_np.shape
T_coef_np.shape
T_sig_coef                  = np.nan * np.ones_like(T_coef_np)

for i in range(30):
    for y in range(204): 
        if T_pav_np[i,y]    < p_accept: 
            T_sig_coef[i,y] = T_coef_np[i,y]
        else: 
            T_sig_coef[i,y] = 0
        y                   +=1
    i                       +=1 
T_sig_coef.shape

S_sig_coef                  = np.nan * np.ones_like(S_coef_np)
for i in range(30):
    for y in range(204): 
        if S_pav_np[i,y]    < p_accept: 
            S_sig_coef[i,y] = S_coef_np[i,y]
        else: 
            S_sig_coef[i,y] = 0
        y                   +=1
    i                       +=1
S_sig_coef.shape

R = T_sig_coef                                                                  #T_sig_coef
plt.imshow(R)
plt.colorbar()
x_label_list = ['4', '6.05', '10.14', '16.99', '28.47', '47.73', '80']
plt.yticks([1,5,10,15,20, 25, 30], x_label_list )


# %% SENSOR LEVEL TOPOPLOT

#### SPLITTER FOR FREQUENCY
R               = T_obs_plot                                                       #T_obs_plot; T_sig_coef; S_sig_coef
epochs          = epochs_s_28
np.shape(R)
theta           = R[0:6][:]
alpha           = R[7:11][:]
beta            = R[12:19][:]
gamma           = R[20:29][:]
freq_int        = R[10:16][:]


what_to_analyze = freq_int
np.shape(what_to_analyze )
what_to_analyze = np.mean(what_to_analyze, axis = 0)

# im,cm           = mne.viz.plot_topomap(what_to_analyze, pos=epochs.info)
#                      # vlim=(-2, 2))
# clb             = im.colorbar(im, orientation = 'horizontal',extend='both')
                     
#                      , 
#                      sensors=True, names=None, mask=None, mask_params=None, 
#                      contours=6, outlines='head', sphere=None, image_interp='cubic', 
#                      extrapolate='auto', border='mean', res=64, size=1, cmap=None, 
#                      vlim=(None, None), cnorm=None, axes=None, show=True, onselect=None)




#### DATA CHECKER
# what_to_analyze = np.transpose(what_to_analyze, [1,0])
negsum = np.count_nonzero(what_to_analyze[what_to_analyze < 0.0])
possum = np.count_nonzero(what_to_analyze[what_to_analyze > 0.0])
negavg = np.mean(what_to_analyze[what_to_analyze < 0.0])
posavg = np.mean(what_to_analyze[what_to_analyze > 0.0])

#### TOPOPLOT
vmax            = np.max(what_to_analyze)
vmin            = np.min(what_to_analyze)
coef            = np.mean(what_to_analyze , axis = 0)
# negsum = np.count_nonzero(coef[coef < 0.0])
# possum = np.count_nonzero(coef[coef > 0.0])
# negavg = np.mean(coef[coef < 0.0])
# posavg = np.mean(coef[coef > 0.0])

vmax            = max(coef)
vmin            = min(coef)
# negcoef = coef * (-1)
vmax            = 10.0
vmin            = -3.5

fig,ax          = plt.subplots(1,1)
im,cm           = mne.viz.plot_topomap(what_to_analyze, pos = epochs.info, ch_type='grad', image_interp='cubic',  cmap = 'Reds', vlim=(-3,0))        #Blues
clb             = fig.colorbar(im, orientation = 'horizontal',)


# %% CSD CALCULATION

index_array_2 = [3,4,6,7,9,10, 11,14, 18, 19, 20, 21,23,25,26,27,28] 
index_array   = [3,4,6,7,9,10, 11,14, 18, 19, 20, 21,23,25,26,27,28] 

i = 3 
af = np.log10(4)
bf = np.log10(80)
frequencies = np.logspace(af,bf, num=30)

i = 3
for i in index_array_2:    
    os.chdir('L:\SATURN\ABRAR WM\JAN') 
    epochs_s = mne.read_epochs('S{}_CorAns_full_S_epochs-epo.fif'.format(i), preload=True)
    epochs_t = mne.read_epochs('S{}_CorAns_full_S_epochs-epo.fif'.format(i), preload=True)
    #Around 2x4min per subject (2 because of modalities)
    csd_s = mne.time_frequency.csd_morlet(epochs_s, frequencies, tmin=0, tmax=4, n_cycles=5, decim=20, n_jobs=-1)    #!!!! CHECK
    csd_s.save('S{}_nov_full_S_csd.h5'.format(i), overwrite=True, verbose=None)
    csd_t = mne.time_frequency.csd_morlet(epochs_t, frequencies, tmin=0, tmax=4, n_cycles=5, decim=20, n_jobs=-1)
    csd_t.save('S{}_nov_full_T_csd.h5'.format(i), overwrite=True, verbose=None)

# BASELINE CSD CALCULATION average
i = 3
for i in index_array_2:    
    os.chdir('L:\SATURN\ABRAR WM\JAN') 
    epochs_average = mne.read_epochs('S{}_ave_epochs-epo.fif'.format(i), preload=True)
    csd_av = mne.time_frequency.csd_morlet(epochs_average, frequencies, tmin=0, tmax=1, n_cycles=5, decim=20, n_jobs=4)
    csd_av.save('S{}_nov_average_base_csd.h5'.format(i), overwrite=True, verbose=None)   
    #concatenate function can be also used



# %% FIGURES for publication

plt.rcParams['figure.dpi']=300
plt.rcParams['axes.labelsize']=4
plt.rcParams['axes.titlesize']=4
plt.rcParams['font.size']=4
plt.rcParams['font.family'] = 'sans-serif'

grand_average_s = mne.grand_average(power_s_list,interpolate_bads=True, drop_bads=True)
grand_average_s.apply_baseline(baseline=(-8,-7), mode='logratio', verbose=None)
grand_average_s.plot(combine = 'mean')
grand_average_t = mne.grand_average(power_t_list,interpolate_bads=True, drop_bads=True)
grand_average_t.apply_baseline(baseline=(-8,-7), mode='logratio', verbose=None)

grand_average   = grand_average_t - grand_average_s 

channels = [137,148,150,151,155,164,165,167,169,170,181]
channels = [ 'MEG1942', 'MEG2012', 'MEG2023', 'MEG1832', 'MEG2042', 'MEG2142',
            'MEG2212', 'MEG2223', 'MEG2233', 'MEG2232', 'MEG2412']

#Automatic channel identification
#sdf = T_obs_plot[:, T_obs_plot.any(0)]

##### FIGURE - 1
# vmin = -1e-21 #Usual
# vmax =  1e-21 #Usual
vmin = -0.2   #Logratio
vmax =  0.2   #Logratio


  
fig, ax =plt.subplots(3, 2, sharex='col' )#, layout="constrained")

#SPATIAL TFP
grand_average_s.plot(
                     #baseline=(-8,-7), mode='logratio', 
                     tmin=0, tmax=4, fmin=4, fmax=80,axes=ax[0, 0],
                     colorbar=True, 
                     combine = 'mean', picks = channels,vmin=vmin, vmax=vmax,
                     yscale='linear') 
ax[1,0].set_title('Time-frequency plot for spatial condition (averaged across channels)')
ax[1,0].set_xlabel('')


#TEMPORAL TFP
grand_average_t.plot(
                     #baseline=(-8,-7), mode='logratio', 
                     tmin=0, tmax=4, fmin=4, fmax=80,axes=ax[1, 0],
                     colorbar=True,
                     combine = 'mean', picks = channels,vmin=vmin, vmax=vmax,
                     yscale='linear')
ax[0,0].set_title('Time-frequency plot for temporal condition (averaged across channels)')
ax[0,0].set_xlabel('')

#CONTRAST TFP
grand_average.plot(
                   #mode='logratio', 
                   tmin=0, tmax=4, fmin=4, fmax=80,axes=ax[2, 0],
                   colorbar=True, 
                   combine = 'mean', picks = channels,vmin=vmin, vmax=vmax,
                   yscale='linear')  


# clb = fig.colorbar(im, cax=cbar_ax)
# clb.ax.set_title('power',fontsize=3) # title on top of colorbar

ax[2,0].set_title('Time-frequency plot for contast (temporal - spatial)')
# ax[2,0].text(3, 8, 'boxed italics text in data coords', style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

#SPATIO-FREQ PLOT
ax4 = plt.subplot2grid((3, 2), (2, 1), rowspan=1)
plt.imshow(T_obs, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-5, vmax=5) 
plt.colorbar(label = 'T values')
plt.contour(T_obs_plot, levels = 1, colors='k', alpha = 0.5, linewidths=[0.5], linestyles='solid', 
           origin = None)
plt.title('Frequency-spatial plot of T values for contrasting condition (temporal - spatial)')

x_label_list = ['4', '6', '10', '17', '28', '48', '80']
plt.yticks([0,5,10,15,20, 25, 29], x_label_list )

ax4.set_xlabel('Channels')

#TOPOPLOT
ax5   = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
im,cm = mne.viz.plot_topomap(what_to_analyze, pos = epochs.info, ch_type='grad', 
                             image_interp='cubic',  cmap = 'RdBu_r', #"Reds"
                             axes=ax5,  show=True,
                             vlim=(-3,3))        #Blues
clb   = fig.colorbar(im)

# ax_x_start = 0.92
# ax_x_width = 0.007
# ax_y_start = 0.7
# ax_y_height = 0.23
# cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
# fig.colorbar(cm, orientation = 'vertical', cax=cbar_ax, 
#              label = 'T values for significant clusters of activation', 
#              boundaries = [0,3])

# im, cm = grand_average.plot_topomap(axes=ax5)        #Blues
ax5.set_title('Topomap of T-values for significant frequencies')         

#PSD
ax6 = plt.subplot2grid((3, 2), (1, 1), rowspan=1)
plt.title('Power Spectral Density for temporal and spatial conditions')
s_spectrum.plot(average=True, 
                # dB=True, amplitude='auto', 
                xscale='linear', 
                ci='sd', 
                ci_alpha=0.5, 
                color='blue', 
                alpha=None, 
                spatial_colors=True, 
                sphere=None, exclude='bads', 
                axes=ax6, show=True)

t_spectrum.plot(average=True, 
                # dB=True, amplitude='auto', 
                xscale='linear', 
                ci='sd', 
                ci_alpha=0.5, 
                color='red', 
                alpha=None, 
                spatial_colors=True, 
                sphere=None, exclude='bads', 
                axes=ax6, show=True)

# plt.tight_layout(h_pad = 0.1, w_pad = 0.05, pad=10,  rect=(0.1,0.3,0.1,0.1))
# plt.tight_layout(pad=600)
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.1,
#                     hspace=0.5)
plt.tight_layout(pad=3.08, h_pad=None, w_pad=None, rect=None)

# s_spectrum.plot(average=True, 
#                 # dB=True, amplitude='auto', 
#                 xscale='linear', 
#                 ci='sd', 
#                 ci_alpha=0.5, 
#                 color='blue', 
#                 alpha=None, 
#                 spatial_colors=True, 
#                 sphere=None, exclude='bads', 
#                 show=True)

#ADDITIONAL
plt.rcParams['figure.dpi']=100
fig, ax =plt.subplots(1, 3)
grand_average.plot_topomap(axes=ax[0])        #Blues
grand_average_s.plot_topomap(axes=ax[1])  
grand_average_t.plot_topomap(axes=ax[2])  

grand_average.plot_topomap()        #Blues
