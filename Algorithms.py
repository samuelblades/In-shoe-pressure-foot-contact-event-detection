# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import scipy.signal as signal
import scipy.stats as stats


# FCE1 method_1

def threshold_crossing_fce(input_signal, threshold, wait=20):

    idx = np.ones(len(input_signal))
    idx[input_signal < threshold] = 0

    fc = (np.array(np.where(np.diff(idx) == 1))).flatten()
    fo = (np.array(np.where(np.diff(idx) == -1)) + 1).flatten()
    
    fc, fo = peaks_wait_time(fc, fo, wait)

    fc, fo = fc_fo_delete(fc, fo)
    
    return fc, fo

# FCE2 method_2

def threshold_crossing_two_fce(input_signal, threshold_on, threshold_off, wait=20):

    idx_on = np.ones(len(input_signal))
    idx_off = np.ones(len(input_signal))
    idx_on[input_signal < threshold_on] = 0
    idx_off[input_signal < threshold_off] = 0

    fc = (np.array(np.where(np.diff(idx_on) == 1))).flatten()
    fo = (np.array(np.where(np.diff(idx_off) == -1)) + 1).flatten()

    fc, fo = peaks_wait_time(fc, fo, wait)

    fc, fo = fc_fo_delete(fc, fo)
    
    return fc, fo

# FCE3 method_3

def first_derivative_fce(input_signal, fs, wait=20):

    # Find the slope and max positive/negative locations
    signal_derivative = np.diff(input_signal) #takes first derivative by subtraction between elements
    threshold = np.mean(abs(signal_derivative))
    b, a = signal.butter(4, 12, btype='lowpass', fs=fs)
    signal_derivative = signal.filtfilt(b, a, signal_derivative) #filtfilt is forward and backward filter

    # need find peaks again
    fc, fc_height = signal.find_peaks(signal_derivative,height = threshold, prominence = threshold) #scipy find_peaks
    fo, fo_height = signal.find_peaks(-signal_derivative,height = threshold, prominence = threshold) 

    fc, fo = peaks_wait_time(fc, fo, wait)

    fc, fo = fc_fo_delete(fc, fo)
            
    return fc, fo

# FCE4 method_4 - Slope Extension

def slope_extension_fce(input_signal, fs=200, wait=20):

    # Find the slope and max positive/negative locations
    signal_derivative = np.diff(input_signal) # slope
    threshold = np.mean(abs(signal_derivative))
    b, a = signal.butter(4, 12, btype='lowpass', fs=fs)
    signal_derivative = signal.filtfilt(b, a, signal_derivative)

    # need find peaks again
    peak_max, height_max = signal.find_peaks(signal_derivative,height = threshold) 
    peak_min, height_min = signal.find_peaks(-signal_derivative,height = threshold, prominence = threshold)

    peak_heights_max = height_max['peak_heights']
    peak_heights_min = height_min['peak_heights']

    fc = []
    # finds the parameters of the linear line and finds the x2 location (fc) based off the linear eq and y2 location (0)
    for i in range(len(peak_max)):
        x1 = peak_max[i]
        m = peak_heights_max[i]
        y1 = input_signal[peak_max[i]]
        b = y1 - (m*x1)
        x2 = (-b)/m
        fc.append(x2)

    fo = []
    # similar to above
    for i in range(len(peak_min)):
        x1 = peak_min[i]
        m = -peak_heights_min[i]
        y1 = input_signal[peak_min[i]]
        b = y1 - (m*x1)
        x2 = (-b)/m
        fo.append(x2)
        
    fc = np.array(fc)
    fo = np.array(fo)
    
    fc, fo = peaks_wait_time(fc, fo, wait)
    
    fc, fo = fc_fo_delete(fc, fo)
    
    fc = np.round(fc[:],0)
    fo = np.round(fo[:])
        
    return fc, fo

# FCE5 method_5 - Filtered Unity
'''
Unity component is based on:

Zhou, P., & Zhang, X. (2013). 
A novel technique for muscle onset detection using surface EMG signals without removal of ECG artifacts. 
Physiological measurement, 35(1), 45.
'''

def filtered_signal_unity_fce(input_signal, filt_frequency=2, fs=200, wait=20):
    
    b, a = signal.butter(4, filt_frequency, btype='lowpass', fs=fs) 
    input_signal_filt = signal.filtfilt(b, a, input_signal)
    threshold = np.mean(input_signal)

    peaks,peak_height = signal.find_peaks(input_signal_filt, height=threshold, prominence=threshold)
    peak_height = peak_height['peak_heights']
    
    #flip signal and find "peaks" (actually valleys)
    valleys,valley_height = signal.find_peaks(-input_signal_filt,height=-threshold, prominence=-threshold)
    valley_height = -valley_height['peak_heights']
    
    # if starts with a peak, remove the first peak so it starts on a fc
    if peaks[0] < valleys[0]:
        peaks = peaks[1:]
        peak_height = peak_height[1:]

    # if the session doesn't end with a valley then cut off the last peak so it ends with a fo
    if peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
        peak_height = peak_height[:-1]
                
    # FOOT CONTACTS
    fc = []
    for i in range(len(valleys)-2): 
        x1 = int(valleys[i])
        y1 = int(valley_height[i])
        x2 = int(peaks[i])
        y2 = int(peak_height[i])

        # find linear line equation based off the above x and y values
        m = (y2-y1)/(x2-x1)
        b = y1 - (m*x1)
        x_signal = np.arange(x1, x2+1, 1)

        # calculate the points for each sample along the line to create a unity line array
        y_unity = m*(x_signal[:]) + b

        # subtract original signal and y_unity to find max difference and append the fc
        fc_diff = y_unity - input_signal[x1:x2+1]
        fc.append(x_signal[np.argmax(fc_diff)])

    fc = np.array(fc)

    # FOOT OFFS
    # similar to above
    fo = []
    for i in range(len(valleys)-5):
        x1 = int(peaks[i])
        y1 = int(peak_height[i])
        x2 = int(valleys[i+1])
        y2 = int(valley_height[i+1])

        m = (y2-y1)/(x2-x1)
        b = y1 - (m*x1)
        x_signal = np.arange(x1, x2+1, 1)

        # calculate the points for each sample along the line to create a unity line array
        y_unity = m*(x_signal[:]) + b

        # subtract integral and y_unity to find max difference
        fo_diff = y_unity - input_signal[x1:x2+1]
        fo.append(x_signal[np.argmax(fo_diff)])

    fo = np.array(fo)

    fc, fo = peaks_wait_time(fc, fo, wait)
    
    fc, fo = fc_fo_delete(fc, fo)

    return fc, fo

# FCE6 method_6 - Harle

def harle_fce_method(input_signal, wait=20):
    # Harle method uses second derivative however, it was too error prone, so we opted to use only first derivative
    
    idx = np.ones(len(input_signal))
    idx[input_signal < 50] = 0 # 50% threshold (it was normalized to 100)

    fc_rough = (np.array(np.where(np.diff(idx) == 1))).flatten()
    fo_rough = (np.array(np.where(np.diff(idx) == -1)) + 1).flatten()

    # first derivative of signal
    signal_derivative = np.diff(input_signal)

    # look for biggest peak in 10 samples before the 50 threshold fc
    # takes the fc rough estimates, creates a window starting 10 frames back from the estimate
    # searches for the peak in the 10 frame window
    # finds a block up until the peak location, and looks for where in that block the values were lower than 0.3* the peak value
    # appends the last value where it was lower to fc (the one closest to the peak on the x axis)
    fc = []
    for i in range(len(fc_rough)): 
        peak = np.max(signal_derivative[fc_rough[i]-10:fc_rough[i]])
        peak_loc = (fc_rough[i]-10)+np.argmax(signal_derivative[fc_rough[i]-10:fc_rough[i]])
        signal_derivative_block = signal_derivative[0:peak_loc]
        peak_zeros = np.where(signal_derivative_block < 0.3*peak)
        fc.append(peak_zeros[0][-1])

    fo = []
    # similar to above but finds the first time it goes under 0.3*peak value after the peak for fo
    for i in range(len(fo_rough)-1):
        peak = np.max(-signal_derivative[fo_rough[i]:fo_rough[i]+10])  
        peak_loc = (fo_rough[i])+np.argmax(-signal_derivative[fo_rough[i]:fo_rough[i]+10])
        signal_derivative_block = -signal_derivative[peak_loc:]
        peak_zeros = peak_loc + np.where(signal_derivative_block < 0.3*peak)
        fo.append(peak_zeros[0][0])

    fc, fo = peaks_wait_time(fc, fo, wait)
  
    fc, fo = fc_fo_delete(fc, fo)
    
    return fc, fo

# FCE7 method_7 - Mann_Hausdorff 

def mann_fce_method(input_signal, threshold=1, wait=20): #threshold defined as 1 here
    
    threshold_coarse = np.mean(input_signal)
    fc_coarse, fo_coarse = threshold_crossing_fce(input_signal, threshold_coarse, wait)

    signal_derivative = np.diff(input_signal)
    b, a = signal.butter(4, 12, btype='lowpass', fs=200) 
    signal_derivative_filt = signal.filtfilt(b, a, signal_derivative)

    fc = []
    # creates a window of 30 before the coarse fc estimate, looks for where the signal passes the threshold on 1 in that window
    # appends the location of the first time it passes the threshold in that window to fc
    for i in range(len(fc_coarse)):
        array_subset = signal_derivative_filt[(fc_coarse[i]-30):fc_coarse[i]]
        idx_fc = np.ones(30)
        idx_fc[array_subset < threshold] = 0
        fc_int = (np.array(np.where(np.diff(idx_fc) == 1))).flatten()
        if fc_int.size == 0: #this just says if it didn't pass 1 during that window, skip to the next window
            continue
        fc.append(fc_int[0]+(fc_coarse[i]-30))

    fc = np.array(fc)

    fo = []
    # similar to above
    for i in range(len(fo_coarse)-1):
        array_subset = signal_derivative_filt[fo_coarse[i]:fo_coarse[i]+30]
        idx_fo = np.ones(30)
        idx_fo[array_subset < -threshold] = 0
        fo_int = (np.array(np.where(np.diff(idx_fo) == 1))).flatten()
        if fo_int.size == 0: #this just says if it didn't pass 1 during that window, skip to the next window
            continue
        fo.append(fo_int[0]+fo_coarse[i])

    fo = np.array(fo)

    fc, fo = peaks_wait_time(fc, fo, wait)

    fc, fo = fc_fo_delete(fc, fo)

    return fc, fo

