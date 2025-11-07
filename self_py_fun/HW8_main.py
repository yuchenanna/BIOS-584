import os
import numpy as np
import scipy.io as sio

from HW8Fun import (
    produce_trun_mean_cov,
    plot_trunc_mean,
    plot_trunc_cov,
)


bp_low = 0.5
bp_upp = 6
E_val = 16
electrode_name_ls = [
    "F3", "Fz", "F4",
    "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8",
]

PARENT_DIR = "/Users/yuchen/Documents/GitHub/BIOS-584"
DATA_DIR = os.path.join(PARENT_DIR, "data")
MAT_FILE = "K114_001_BCI_TRN_Truncated_Data_0.5_6.mat"


mat_path = os.path.join(DATA_DIR, MAT_FILE)
eeg_trunc_obj = sio.loadmat(mat_path)

eeg_trunc_signal = eeg_trunc_obj["Signal"]
eeg_trunc_type = np.squeeze(eeg_trunc_obj["Type"])


(
    signal_tar_mean,
    signal_ntar_mean,
    signal_tar_cov,
    signal_ntar_cov,
    signal_all_cov,
) = produce_trun_mean_cov(eeg_trunc_signal, eeg_trunc_type, E_val)

time_index = np.linspace(-200, 800, signal_tar_mean.shape[1])


subject_name = "K114"


plot_trunc_mean(
    signal_tar_mean,
    signal_ntar_mean,
    subject_name=subject_name,
    time_index=time_index,
    E_val=E_val,
    electrode_name_ls=electrode_name_ls,
)

plot_trunc_cov(
    signal_tar_cov,
    cov_type="Target",
    time_index=time_index,
    subject_name=subject_name,
    E_val=E_val,
    electrode_name_ls=electrode_name_ls,
)

plot_trunc_cov(
    signal_ntar_cov,
    cov_type="Non-Target",
    time_index=time_index,
    subject_name=subject_name,
    E_val=E_val,
    electrode_name_ls=electrode_name_ls,
)

plot_trunc_cov(
    signal_all_cov,
    cov_type="All",
    time_index=time_index,
    subject_name=subject_name,
    E_val=E_val,
    electrode_name_ls=electrode_name_ls,
)

print("HW8 figures generated and saved in ./K114")
