import librosa
import numpy as np
import scipy.signal

def detect_spectral_peaks(y, sr, frame_length=2048, hop_length=512, threshold=0.8):
    # Step 1: Perform STFT to get time-frequency representation
    D = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

    # Step 2: Convert amplitude to dB scale for better analysis
    S_db = librosa.amplitude_to_db(D, ref=np.max)

    # Step 3: Peak picking - detect peaks in each frequency bin
    peaks = []
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)

    for freq_bin in range(S_db.shape[0]):
        # Get spectrum slice for the current frequency bin across time
        spectrum_slice = S_db[freq_bin]
        
        # Detect peaks in this spectrum slice
        peak_indices = scipy.signal.find_peaks(spectrum_slice, height=threshold * np.max(spectrum_slice))[0]
        peak_times = times[peak_indices]
        
        # Store each peak as a (time, frequency) tuple
        for pt in peak_times:
            frequency = librosa.fft_frequencies(sr=sr, n_fft=frame_length)[freq_bin]
            peaks.append((pt, frequency))

    return peaks
