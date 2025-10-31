import numpy as np
import matplotlib.pyplot as plt
import math

def produce_trun_mean_cov(input_signal, input_type, E_val):
    r"""
    args:
    -----
        input_signal: 2d-array, (sample_size_len, feature_len)
        input_type: 1d-array, (sample_size_len,)

    return:
    -----
        A list of 5 arrays including
            signal_tar_mean, (E_val, length_per_electrode)
            signal_ntar_mean, (E_val, length_per_electrode)
            signals_tar_cov, (E_val, length_per_electrode, length_per_electrode)
            signals_ntar_cov, (E_val, length_per_electrode, length_per_electrode)
            signals_cov, (E_val, length_per_electrode, length_per_electrode)

    note:
    -----
        descriptive mean and sample covariance statistics from real data
    """
    input_signal = np.asarray(input_signal)
    input_type = np.asarray(input_type)

    if input_signal.ndim != 2:
        raise ValueError("input_signal must be 2D: (samples, features)")
    if input_type.ndim != 1 or input_type.shape[0] != input_signal.shape[0]:
        raise ValueError("input_type must be 1D and match number of samples in input_signal")
    if E_val <= 0:
        raise ValueError("E_val must be positive")

    N, F = input_signal.shape
    if F % E_val != 0:
        raise ValueError(f"Feature length {F} is not divisible by E_val {E_val}.")
    L = F // E_val

    input_signal = np.reshape(input_signal, [N, E_val, L])
    input_signal_tar = input_signal[input_type == 1, ...]
    input_signal_ntar = input_signal[input_type == -1, ...]

    if not input_signal_tar.any():
        raise ValueError("No target samples found (input_type==1).")
    if not input_signal_ntar.any():
        raise ValueError("No non-target samples found (input_type==0).")


    signal_tar_mean = np.mean(input_signal_tar, axis=0)
    signal_ntar_mean = np.mean(input_signal_ntar, axis=0)

    # Examine sample covariance matrix
    def _stack_cov(x):
        return np.stack(
            [np.cov(x[:, e, :], rowvar=False, bias=False) for e in range(E_val)],
            axis=0
        )
    signal_tar_cov = np.stack([np.cov(input_signal_tar[:, e_iter, :], rowvar=False)
                                   for e_iter in range(E_val)], axis=0)
    signal_ntar_cov = np.stack([np.cov(input_signal_ntar[:, e_iter, :], rowvar=False)
                                    for e_iter in range(E_val)], axis=0)
    signal_all_cov = np.stack([np.cov(input_signal[:, e_iter, :], rowvar=False)
                                 for e_iter in range(E_val)], axis=0)
    return [signal_tar_mean, signal_ntar_mean,
            signal_tar_cov, signal_ntar_cov, signal_all_cov]


def plot_trunc_mean(
        eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
        y_limit=np.array([-5, 8]), fig_size=(12, 12)
):
    r"""
    :param eeg_tar_mean:
    :param eeg_ntar_mean:
    :param subject_name:
    :param time_index:
    :param E_val:
    :param electrode_name_ls:
    :param y_limit: optional parameter, a list or an array of two numbers
    :param fig_size: optional parameter, a tuple of two numbers
    :return:
    """

    eeg_tar_mean = np.asarray(eeg_tar_mean)
    eeg_ntar_mean = np.asarray(eeg_ntar_mean)
    time_index = np.asarray(time_index)

    if eeg_tar_mean.shape != eeg_ntar_mean.shape:
        raise ValueError("eeg_tar_mean and eeg_ntar_mean must have the same shape")
    if eeg_tar_mean.shape[0] != E_val:
        raise ValueError("First dim of mean arrays must equal E_val")
    if len(electrode_name_ls) != E_val:
        raise ValueError("electrode_name_ls length must equal E_val")
    if eeg_tar_mean.shape[1] != time_index.shape[0]:
        raise ValueError("Time length mismatch between mean arrays and time_index")

    cols = math.ceil(math.sqrt(E_val))
    rows = math.ceil(E_val / cols)

    fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False, sharex=True, sharey=True)
    for e in range(E_val):
        r, c = divmod(e, cols)
        ax = axes[r, c]
        ax.plot(time_index, eeg_tar_mean[e], label='Target')
        ax.plot(time_index, eeg_ntar_mean[e], label='Non-target')
        ax.set_title(electrode_name_ls[e])
        ax.set_ylim(*y_limit)
        for k in range(E_val, rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis('off')
        fig.suptitle(subject_name)
        fig.supxlabel("Time (ms)")
        fig.supylabel("Amplitude (µV)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', frameon=False)
        fig.tight_layout(rect=[0, 0, 0.98, 0.96])
    # plt.savefig("./{}/Mean.png".format(subject_name))
    plt.show()