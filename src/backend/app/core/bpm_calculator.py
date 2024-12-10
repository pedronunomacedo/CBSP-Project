import io
import numpy as np
import librosa
from pydub import AudioSegment

def calculate_bpm_from_peaks(peaks):
    peaks = sorted(peaks)
    print("peaks: ", peaks)
    intervals = [peaks[i+1][0] - peaks[i][0] for i in range(len(peaks) - 1)]
    bpm_values = [60 / interval for interval in intervals if interval > 0]
    # avg_bpm = np.mean(bpm_values) if bpm_values else 0
    return bpm_values

def calculate_bpm(wav_io):
    # Convert BytesIO object to bytes and reload with librosa
    wav_io.seek(0)  # Ensure we're reading from the start of the BytesIO object
    y, sr = librosa.load(io.BytesIO(wav_io.read()), sr=None)

    # Calculate onset envelope and BPM
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    return float(tempo[0])  # Return the first tempo value

