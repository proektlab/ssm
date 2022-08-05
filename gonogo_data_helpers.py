"""Utilities to help with loading and dealing with Geffen lab go-nogo data"""
import os
import numpy as np

DATA_DIR = 'Z:\\eblackwood\\gonogo_stuff\\gain-gonogo\\_data'


def db_to_amp_ratio(db):
    return 10 ** (db / 20)


def amp_ratio_to_db(amp_ratio):
    return 20 * np.log10(amp_ratio)


def convert_target_shifts_to_combined_shifts(target_db_shift):
    """
    Convert the "targetDBShift" value(s) (i.e., dB SNR, loudness of target compared to mean background
    to the dB offset of the *combined* stimulus (background + target) compared to background.
    This allows the "no target" condition to have a value of 0, which is better for fitting a model.
    """
    combined_ratio = 1 + db_to_amp_ratio(target_db_shift)
    return amp_ratio_to_db(combined_ratio)


def get_inputs_from_results_dict(fn, result):
    n_trials = len(result['resp'])
    stim_inds = result['tt'][:n_trials, 0]
    this_inputs = np.zeros((n_trials, 2))
    input_type = {5: 0, 15: 1}[result['params'].sd[1]]  # different input column based on noise SD
    
    # deal with different types of targetDBShift
    combined_shift_range = convert_target_shifts_to_combined_shifts(result['params'].targetDBShift)
    past_range_msg = f'tt entries past the number of target shifts for file {os.path.basename(fn)}'
    
    if np.isscalar(combined_shift_range):
        # One volume for all targets
        assert np.all(stim_inds <= 1), past_range_msg
        this_inputs[stim_inds == 1, input_type] = combined_shift_range
        
    elif combined_shift_range.ndim == 1:
        # Different target volumes (just one range)
        assert np.all(stim_inds <= len(combined_shift_range)), past_range_msg
        b_stim_on = stim_inds > 0
        this_inputs[b_stim_on, input_type] = combined_shift_range[stim_inds[b_stim_on] - 1]
        
    elif combined_shift_range.ndim == 2:
        # 2 different ranges of target volumes
        assert len(combined_shift_range) == 2, 'Only 2 target volume ranges supported'
        assert np.all(stim_inds <= combined_shift_range.shape[1]), past_range_msg
        b_stim_on = stim_inds > 0
        switch_trial = result['params'].switchTrial - 1
        range1_stim_on_inds = np.flatnonzero(b_stim_on & (np.arange(n_trials) < switch_trial))
        range2_stim_on_inds = np.flatnonzero(b_stim_on & (np.arange(n_trials) >= switch_trial))
        this_inputs[range1_stim_on_inds, input_type] = combined_shift_range[0, stim_inds[range1_stim_on_inds] - 1]
        this_inputs[range2_stim_on_inds, input_type] = combined_shift_range[1, stim_inds[range2_stim_on_inds] - 1]
    
    else:
        raise ValueError('Cannot interpret the shape of targetDBShift')

    return this_inputs
