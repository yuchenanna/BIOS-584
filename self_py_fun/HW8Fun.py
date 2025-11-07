import os
import numpy as np
import matplotlib.pyplot as plt


def produce_trun_mean_cov(input_signal, input_type, E_val):
    r"""
    args:
    -----
        input_signal: 2d-array, (sample_size_len, feature_len)
        input_type: 1d-array, (sample_size_len,)
        E_val: integer, number of electrodes

    return:
    -----
        A list of 5 arrays including
            signal_tar_mean, (E_val, length_per_electrode)
            signal_ntar_mean, (E_val, length_per_electrode)
            signal_tar_cov, (E_val, length_per_electrode, length_per_electrode)
            signal_ntar_cov, (E_val, length_per_electrode, length_per_electrode)
            signal_all_cov, (E_val, length_per_electrode, length_per_electrode)
    """
    sample_size_len, feature_len = input_signal.shape
    length_per_electrode = feature_len // E_val

    # target: 1, non-target: -1 (matches your HW7 code)
    sig_tar = input_signal[input_type == 1, :]
    sig_ntar = input_signal[input_type == -1, :]

    signal_tar_mean = np.zeros((E_val, length_per_electrode))
    signal_ntar_mean = np.zeros((E_val, length_per_electrode))
    signal_tar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_ntar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_all_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))

    for e in range(E_val):
        start_idx = e * length_per_electrode
        end_idx = (e + 1) * length_per_electrode

        tar_elec = sig_tar[:, start_idx:end_idx]
        ntar_elec = sig_ntar[:, start_idx:end_idx]
        all_elec = input_signal[:, start_idx:end_idx]

        signal_tar_mean[e, :] = np.mean(tar_elec, axis=0)
        signal_ntar_mean[e, :] = np.mean(ntar_elec, axis=0)

        if tar_elec.shape[0] > 1:
            signal_tar_cov[e, :, :] = np.cov(tar_elec, rowvar=False)
        if ntar_elec.shape[0] > 1:
            signal_ntar_cov[e, :, :] = np.cov(ntar_elec, rowvar=False)
        if all_elec.shape[0] > 1:
            signal_all_cov[e, :, :] = np.cov(all_elec, rowvar=False)

    return [
        signal_tar_mean,
        signal_ntar_mean,
        signal_tar_cov,
        signal_ntar_cov,
        signal_all_cov,
    ]


def plot_trunc_mean(
    eeg_tar_mean,
    eeg_ntar_mean,
    subject_name,
    time_index,
    E_val,
    electrode_name_ls,
    y_limit=np.array([-5, 8]),
    fig_size=(12, 12),
):
    assert E_val == 16

    fig, axes = plt.subplots(4, 4, figsize=fig_size, sharex=True, sharey=True)
    axes = axes.ravel()

    for e in range(E_val):
        axes[e].plot(time_index, eeg_tar_mean[e, :], label="Target", color="b")
        axes[e].plot(
            time_index,
            eeg_ntar_mean[e, :],
            label="Non-target",
            color="r",
            linestyle="--",
        )
        axes[e].set_title(electrode_name_ls[e])
        axes[e].set_ylim(y_limit)
        axes[e].grid(True, linestyle="--", alpha=0.6)
        if e == 0:
            axes[e].legend(fontsize=8)

    plt.suptitle(
        f"{subject_name} — Target vs Non-Target Mean ERPs",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # Save into K114 under CURRENT WORKING DIRECTORY
    save_dir = os.path.join(os.getcwd(), "K114")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "Mean.png"))
    plt.close()


def plot_trunc_cov(
    eeg_cov,
    cov_type,
    time_index,
    subject_name,
    E_val,
    electrode_name_ls,
    fig_size=(14, 12),
    cmap="viridis",
):
    assert E_val == 16
    X, Y = np.meshgrid(time_index, time_index)

    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    axes = axes.ravel()

    pcm = None
    for e in range(E_val):
        pcm = axes[e].contourf(X, Y, eeg_cov[e, :, :], cmap=cmap)
        axes[e].set_title(electrode_name_ls[e])
        axes[e].set_xlabel("Time (ms)")
        axes[e].set_ylabel("Time (ms)")

    if pcm is not None:
        cbar = fig.colorbar(pcm, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("Covariance (μV²)", rotation=90)

    fig.suptitle(
        f"{subject_name} — {cov_type} Sample Covariance per Electrode",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # Save into K114 under CURRENT WORKING DIRECTORY
    save_dir = os.path.join(os.getcwd(), "K114")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"Covariance_{cov_type}.png"))
    plt.close()
