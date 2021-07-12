import math
import numpy as np
import mne


def crop_epochs(epochs, window_size, step_size):
    """
    Get crops from epochs.

    It takes an MNE Epochs object and gets crops from it. The label of each
     crop must be in the last column.

    Parameters:
    -----------
    epochs : mne.Epochs
        The MNE Epochs object.
    window_size : int
        The crop size, in sample points.
    step_size : int
        The distance between crops, in sample points.

    Return:
    -------
    A 2-tuple with the crops and their labels.
    """
    data = epochs.get_data()
    n_epochs, n_channels, epoch_size = data.shape
    assert window_size < epoch_size, 'Window must be shorter than epoch'

    crops_per_epoch = int(math.floor((epoch_size - window_size) / step_size) + 1)
    crops = np.empty((crops_per_epoch * n_epochs, n_channels - 1, window_size))
    labels = np.empty(crops_per_epoch * n_epochs)

    i_crop = 0
    for epoch in epochs:
        offset = 0
        labels[i_crop:(i_crop + crops_per_epoch)] = epoch[-1, 0]
        while offset + window_size <= epoch_size:
            crops[i_crop] = epoch[:-1, offset:(offset + window_size)]
            i_crop += 1
            offset += step_size
    assert i_crop == crops_per_epoch * n_epochs, "Incorrect number of crops"
    return (crops, labels), crops_per_epoch


def crop_passive(raw, events, activity_duration, window_size, step_size,
                 exclude=None, passive_id=None, ignore_longers=None):
    """
    Get crops from passive states.

    Passive states are found between trials, from activity_duration samples
    after a trial onset to the next trial onset.
    If passive_id is set, trials with that event id are considered passive
    states, so the whole trial is cropped.

    Parameters:
    -----------
    raw : mne.Raw
        A MNE Raw object with the Raw data.
    events : numpy.ndarray
        The events as obtained from mne.find_events.
    activity_duration : int
        The number of active samples (skipped) in active events.
    window_size: int
        The crop size, in sample points.
    step_size: int
        The distance between crops, in sample points.
    exclude : iterable, optional
        Event ids that must be completely ignored.
    passive : int, optional
        The id of the passive event. The begining of these trials won't be
        skipped.
    ignore_longers : int | (int, int), optional
        Ignores epochs longer than ignore longer samples. If two values are
         given, the second value sets the number of samples that should be
         cropped.
    """

    raw.load_data()
    data = raw.get_data()
    crops = []
    if ignore_longers and not type(ignore_longers) is tuple:
        ignore_longers = (ignore_longers, 0)

    for i_event, event in enumerate(events):
        if event[2] in exclude:
            continue
        start = event[0]
        if event[2] != passive_id:
            start += activity_duration

        if i_event < len(events) - 1:
            end = events[i_event + 1, 0]
        else:
            end = data.shape[1]  # N. of samples
        if ignore_longers and end - start > ignore_longers[0]:
            end = start + ignore_longers[1]
        if end - start < window_size:
            continue

        n_crops = math.floor((end - start - window_size) / step_size) + 1
        event_crops = np.empty((n_crops, len(data), window_size))
        for i_crop in range(n_crops):
            event_crops[i_crop] = data[:, start:(start + window_size)]
            start += step_size
        assert start + window_size > end

        crops.append(event_crops)

    return np.concatenate(crops, axis=0)


def standarize(channel):
    assert(channel.ndim == 1)

    mean = np.mean(channel)
    stdev = np.std(channel)

    return (channel - mean) / stdev
