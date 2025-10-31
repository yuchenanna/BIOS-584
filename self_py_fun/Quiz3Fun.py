import numpy as np


# This function is merely for BIOS 584 debugging purposes
def compute_D_partial(input_signal):
    r"""
    :param input_signal: array-like
    """
    input_signal = np.asarray(input_signal)
    T_len = len(input_signal)
    signal_diff = np.diff(input_signal)
    D_val = np.sum(np.sqrt(1+signal_diff**2)) / (T_len - 1)
    return D_val


def compute_D_correct(input_signal):
    r"""
    :param input_signal:
    """
    input_signal = np.asarray(input_signal)
    signal_diff = np.diff(input_signal) 
    D_val = np.sum(np.sqrt(1 + signal_diff ** 2)) / (len(input_signal) - 1)
    return D_val

if __name__ == "__main__":
    test_signal = np.array([0, 1, 2, 3])
    print("Partial:", compute_D_partial(test_signal))
    print("Correct:", compute_D_correct(test_signal))