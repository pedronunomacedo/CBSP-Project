import numpy as np
from scipy.signal import find_peaks
import hashlib

def findAudioHighPeaks(samples, sample_rate, min_freq=1000):
    # Frame settings
    frame_length = sample_rate  # 1 second of samples
    peak_times = []

    # Process each second of audio
    for start in range(0, len(samples), frame_length):
        end = start + frame_length
        frame = samples[start:end]

        # Ensure frame is exactly 1 second
        if len(frame) < frame_length:
            continue

        # Perform FFT to get the frequency domain
        freqs = np.fft.fftfreq(len(frame), 1/sample_rate)
        fft_values = np.fft.fft(frame)

        # Only keep positive frequencies
        positive_freqs = freqs[freqs > 0]
        positive_fft_values = np.abs(fft_values[freqs > 0])

        # Filter to keep only high-frequency components
        high_freqs = positive_freqs[positive_freqs > min_freq]
        high_freq_values = positive_fft_values[positive_freqs > min_freq]

        # Smooth the frequency values (optional)
        smoothed_high_freq_values = np.convolve(high_freq_values, np.ones(5)/5, mode='same')

        # Add peak distance (minimum distance between peaks)
        distance = int(sample_rate / min_freq)

        # Find peaks in the high-frequency range
        peaks, _ = find_peaks(smoothed_high_freq_values, height=np.max(smoothed_high_freq_values), distance=distance)  # Adjust threshold

        for peak in peaks:
            frequency = high_freqs[peak]
            time_offset = peak / sample_rate  # Convert frequency index to time (in seconds)
            peak_times.append((round(time_offset, 3), frequency))  # Append both time and frequency as a tuple

        # Removed duplicated detected peaks
        tolerance = 0.001  # Adjust as needed
        peak_times = sorted(set(peak_times), key=lambda x: peak_times.index(x))
        peak_times = sorted([peak for i, peak in enumerate(peak_times) if i == 0 or abs(peak[0] - peak_times[i-1][0]) > tolerance])

    return peak_times


def get_hash_peak(peak):
    peak_data = f"{peak[1]}_{peak[0]}"  # Format as "frequency_time"
    hash = hashlib.sha256(peak_data.encode()).hexdigest()

    return hash