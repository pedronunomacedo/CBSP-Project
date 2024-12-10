import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class HybridMultibandApproach:
    def compute_phase_differences(self, y, sr, hop_length=256, frame_length=2048):
        stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
        phase = np.angle(stft)
        unwrapped_phase = np.unwrap(phase, axis=0)
        phase_diff = np.diff(unwrapped_phase, axis=1)
        phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrapping to [-pi, pi]
        return phase_diff
    
    def detect_onsets(self, y, sample_rate, hop_length=512):
        S = np.abs(librosa.stft(y))

        # Detecting onsets
        onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sample_rate)
        dynamic_delta = 0.3

        print("[DEBUG] dynamic_delta: ", dynamic_delta)

        # Calculate number of frames
        total_samples = len(y)
        number_of_frames = (total_samples + hop_length - 1) // hop_length  # Ceiling division

        print("[DEBUG] Total samples:", total_samples)
        print("[DEBUG] Number of frames:", number_of_frames)


        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
            wait=10,
            delta=dynamic_delta,
        )

        return onset_env, onsets
    
    def convert_frames_to_times(self, onsets, sr, hop_length=512):
        return librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    

    def detect_transients(self, frequencies, sr, threshold_db=3):
        energy = np.sum(np.abs(frequencies)**2, axis=0)  # Energy per frame

        print("[DEBUG] energy shape:", energy.shape)

        energy_diff = np.diff(energy)
        # Convert difference to dB
        energy_diff_db = librosa.power_to_db(energy_diff, ref=np.max)

        # Detect transients where energy increase exceeds the threshold
        transients = np.where(energy_diff_db > threshold_db)[0] + 1  # +1 to correct index after diff

        # Convert transient frame indices to time
        transient_times = librosa.frames_to_time(transients, sr=sr)


        return transients, transient_times
