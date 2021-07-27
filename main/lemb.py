import scipy.io
import numpy as np
import mne

event_id = {'left/hand': 1,'right/hand': 2, 'passive': 3, 'left/leg': 4,
            'tongue': 5, 'right/leg': 6, 'exp/break': 91, 'exp/end': 92,
            'exp/start': 93}


def load_data(filename, drop_half=False):
    """
    Reads a data file from the "Large EEG Motor-Imagery BCI" datasets.

    The data is returned as a MNE Raw object.
    The dataset can be found in https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698.

    Parameters:
    -----------
    filename : str
        The name of the data file.
    """

    ml_data = scipy.io.loadmat(filename)
    ml_data = ml_data['o'][0, 0]
    # id = ml_data['id'][0]
    # ns = ml_data['nS'][0, 0]
    sfreq = ml_data['sampFreq'][0, 0]
    markers = ml_data['marker']
    data = ml_data['data'] * 10e-6  # From uV to V.
    print(np.shape(data))
    print(np.shape(markers))
    if drop_half != False:
        drop_size = drop_half / 100
        size = int(len(data)*drop_size)
        data = data[:size]
        markers = markers[:size]
        print(np.shape(data))
        print(np.shape(markers))
    channel_names = ml_data['chnames'][:, 0].tolist()
    channel_names = [name[0] for name in channel_names]
    n_channels = len(channel_names)

    # Last channel, X5, is for synchronization.
    channel_types = ['eeg'] * (n_channels - 1) + ['stim']

    data = np.hstack((data, markers))
    channel_names.append('STI101')
    channel_types.append('stim')

    montage = 'standard_1020'

    info = mne.create_info(channel_names, sfreq, channel_types)
    info.set_montage(montage)

    return mne.io.RawArray(data.T, info)
