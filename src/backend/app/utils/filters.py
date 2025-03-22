from scipy.signal import butter, lfilter, filtfilt, cheby1
import numpy as np

def bandpass_filter(data, lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data
