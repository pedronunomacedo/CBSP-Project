from scipy.signal import butter, lfilter, filtfilt, cheby1
import numpy as np

def bandpass_filter(data, lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def bandpass_filterV2(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = max(lowcut, 1) / nyquist
    high = min(highcut, nyquist - 1) / nyquist
    try:
        b, a = butter(order, [low, high], btype='band', analog=False)
        filtered_data = lfilter(b, a, data)
        if not np.isfinite(filtered_data).all():
            print("Non-finite values detected after filtering.")
            raise ValueError("Filtered data contains NaN or infinite values after filtering.")
    except ValueError as e:
        print("Filter design error:", e)
        raise
    return filtered_data

def bandpass_filter_chebyshev(data, lowcut, highcut, sr, order=5, rp=1):
    """
    Apply a Chebyshev type I bandpass filter.
    """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, rp, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data