from fastapi import APIRouter, UploadFile, File, HTTPException
import io
import time
import numpy as np
from pydub import AudioSegment
import librosa
from scipy.signal import find_peaks, butter, lfilter, correlate, resample
import SongNameSplit
from scipy.ndimage import uniform_filter1d
from app.utils.file_conversion import applyDBSCANalgorithm, categorizeMusicTempo
from app.utils.song_peaks import findAudioHighPeaks, get_hash_peak
from app.utils.filters import bandpass_filter

router = APIRouter()

@router.post("/bpm_per_second")
async def get_bpm_per_second(file: UploadFile = File(...)):
    try:
        # Read the file content
        audio_data = await file.read()
        
        result = SongNameSplit.namesplit(file.filename)
        song_name = result['songname']
        song_artist = result['artist']

        # Convert mp3 to wav in memory
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Load audio data with librosa
        y, sr = librosa.load(wav_io, sr=None)

        # Segment length for 5-second segments
        segment_length = sr * 5  # 5 seconds worth of samples
        total_length = len(y)
        bpm_per_second = []

        # Loop through each second of the audio
        for start in range(0, total_length, segment_length):
            end = start + segment_length
            segment = y[start:end]

            # Skip if the segment is shorter than 1 second (e.g., at the end of the song)
            if len(segment) < segment_length:
                continue

            # Estimate BPM for the 1-second segment
            onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            
            # Append the calculated BPM (tempo) for this second
            bpm_per_second.append(round(float(tempo[0]), 2))

        song_bpm = applyDBSCANalgorithm(bpm_per_second)

        song_tempo = categorizeMusicTempo(song_bpm)

        return {
            "bpm_per_second": bpm_per_second,
            "song_bpm": round(song_bpm, 2),
            "song_tempo": song_tempo,
            "song_name": song_name,
            "song_artist": song_artist
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    


@router.post("/mine_bpm_per_second")
async def get_mine_bpm_per_second(file: UploadFile = File(...)):
    try:
        # Read the file content
        audio_data = await file.read()

        # Convert mp3 to wav in memory
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sample_rate = librosa.load(wav_io, sr=None)

        # Apply band-pass filter
        filtered_y = bandpass_filter(y, 100, 5000, sample_rate)

        # # List to store BPM for each second
        window_size = sample_rate * 5  # 5-second window
        hop_length = sample_rate       # 1-second hop
        total_length = len(filtered_y)
        bpm_list = []

        # Process each 5-second segment
        for start in range(0, total_length - window_size + 1, hop_length):
            end = start + window_size
            segment = filtered_y[start:end]

            # Step 1: Compute the energy envelope
            energy = np.abs(segment) ** 2 # Square of the amplitude ( energy = amplitude^2 )
            window_size_samples = int(0.1 * sample_rate)  # 100ms window for smoothing
            smoothed_energy = uniform_filter1d(energy, size=window_size_samples)  # Smooth the energy in each sample value by looking to its two direct neighbours [1, 4, 6] -> [((1+1+4)/3), ((1+4+6)/3), ((4+6+6)/3)] -> [2, 3.67, 5.33]

            # Step 2: Detect peaks in the energy envelope
            mean_height = np.mean(smoothed_energy) * 0.5  # Adaptive height threshold
            min_distance = int(sample_rate * 60 / 200)  # Adjust for up to 200 BPM
            peaks, _ = find_peaks(smoothed_energy, height=mean_height, distance=min_distance)

            # Calculate BPM based on the intervals between peaks
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / sample_rate  # Convert intervals from samples to seconds
                avg_interval = np.mean(peak_intervals)  # Average interval between peaks
                bpm = 60 / avg_interval  # Convert seconds per beat to BPM
            else:
                bpm = 0 

            bpm_list.append(round(bpm, 2))

        final_bpm_per_second_list = []
        for idx, bpm in enumerate(bpm_list):
            if bpm == 0:
                if idx == 0:
                    final_bpm_per_second_list.append(bpm_list[idx + 1])
                elif idx == len(bpm_list) - 1:
                    final_bpm_per_second_list.append(bpm_list[idx - 1])
                else:
                    final_bpm_per_second_list.append((bpm_list[idx - 1] + bpm_list[idx + 1]) / 2)
            else:
                final_bpm_per_second_list.append(bpm)

        # Optionally, calculate an average BPM for the entire song
        if final_bpm_per_second_list:
            song_bpm = applyDBSCANalgorithm(final_bpm_per_second_list)
            song_bpm = round(song_bpm, 2)
            song_tempo = categorizeMusicTempo(song_bpm)
        else:
            song_bpm = 0
            song_tempo = "Unknown"

        return {
            "bpm_per_second": final_bpm_per_second_list,
            "song_bpm": song_bpm,
            "song_tempo": song_tempo
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    

# Lowpass filter for low frequencies
def butter_lowpass(cutoff, sr, order=4):
    nyquist = 0.5 * sr
    if cutoff <= 0:
        raise ValueError("Lowpass cutoff frequency must be greater than 0")
    normal_cutoff = max(1e-6, cutoff / nyquist)  # Ensure valid normalized frequency > 0
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def butter_lowpass_filter(data, cutoff, sr, order=4):
    b, a = butter_lowpass(cutoff, sr, order)
    return lfilter(b, a, data)

# Bandpass filter for middle and high frequencies
def butter_bandpass(lowcut, highcut, sr, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, sr, order=4):
    b, a = butter_bandpass(lowcut, highcut, sr, order)
    return lfilter(b, a, data)

# Utility: Transient Detection (Counts energy spikes)
def transient_detection(y, threshold=0.01):
    diff = np.abs(np.diff(y))
    return (diff > threshold).astype(float)

# Highpass filter for high frequencies
def butter_highpass(cutoff, sr, order=4):
    nyquist = 0.5 * sr
    if cutoff <= 0:
        raise ValueError("Highpass cutoff frequency must be greater than 0")
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

def butter_highpass_filter(data, cutoff, sr, order=4):
    b, a = butter_highpass(cutoff, sr, order)
    return lfilter(b, a, data)

# Periodicity Detection Function (Autocorrelation)
def compute_PeDF(odf):
    autocorr = correlate(odf, odf, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags only
    return autocorr

# Combine PeDFs from bands
def combine_PeDFs(PeDF_low, PeDF_mid, PeDF_high, weights):
    min_length = min(len(PeDF_low), len(PeDF_mid), len(PeDF_high))
    return (weights[0] * PeDF_low[:min_length] +
            weights[1] * PeDF_mid[:min_length] +
            weights[2] * PeDF_high[:min_length])


@router.post("/hybrid_bpm_per_second")
async def get_hybrid_bpm_per_second(file: UploadFile = File(...)):
    try:
        # Step 1: Read and convert audio to WAV
        start_time = time.time()
        audio_data = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sr = librosa.load(wav_io, sr=None)
        end_time = time.time()
        print(f"[{file.filename} - File conversion] Elapsed time: {(end_time - start_time):.2f} seconds")

        # Step 2: Bandpass Filtering
        start_time = time.time()
        low_band = butter_lowpass_filter(y, 200, sr)           # Low Frequency Band
        mid_band = butter_bandpass_filter(y, 200, 5000, sr)    # Middle Frequency Band
        high_band = butter_highpass_filter(y, 5000, sr)        # High Frequency Band
        end_time = time.time()
        print(f"[{file.filename} - Bandpass filtering] Elapsed time: {(end_time - start_time):.2f} seconds")

        # Step 3: Onset and Transient Detection
        start_time = time.time()
        onset_low = librosa.onset.onset_strength(y=low_band, sr=sr)   # Spectral Onset for Low
        transient_mid = transient_detection(mid_band)                 # Transient Detection for Mid
        transient_high = transient_detection(high_band)               # Transient Detection for High
        end_time = time.time()
        print(f"[{file.filename} - Onset/transient detection] Elapsed time: {(end_time - start_time):.2f} seconds")

        # Resample transient_mid and transient_high to match onset_low
        transient_mid_resampled = resample(transient_mid, len(onset_low))
        transient_high_resampled = resample(transient_high, len(onset_low))

        # Step 4: Periodicity Detection (PeDF) for each band
        start_time = time.time()
        PeDF_low = compute_PeDF(onset_low)
        PeDF_mid = compute_PeDF(transient_mid_resampled)
        PeDF_high = compute_PeDF(transient_high_resampled)
        end_time = time.time()
        print(f"[{file.filename} - PeDFs] Elapsed time: {(end_time - start_time):.2f} seconds")

        # Step 5: Combine PeDFs with weights
        weights = [0.3, 0.4, 0.3]  # Weights for Low, Mid, High bands
        combined_PeDF = combine_PeDFs(PeDF_low, PeDF_mid, PeDF_high, weights)

        # Step 6: Peak Detection in Combined PeDF
        start_time = time.time()
        peaks, _ = find_peaks(combined_PeDF, height=np.mean(combined_PeDF), distance=30)

        # Step 7: BPM Calculation
        if len(peaks) < 2:
            raise ValueError("Not enough periodic peaks detected to calculate BPM.")

        peak_intervals = np.diff(peaks) * (512 / sr)  # Convert lags to time intervals (hop_size=512)
        average_interval = np.mean(peak_intervals)
        song_bpm = 60 / average_interval

        # Step 8: Categorize music tempo
        song_tempo = categorizeMusicTempo(song_bpm)
        end_time = time.time()
        print(f"[{file.filename} - Peak detection and BPM calculation] Elapsed time: {(end_time - start_time):.2f} seconds")


        # Additional Step: Get the BPM in each 5 second window
        window_size = sr * 5  # 5-second window
        hop_length = sr       # 1-second hop
        total_length = len(y)
        bpm_list = []

        for start in range(0, total_length - window_size + 1, hop_length):
            end = start + window_size
            segment = y[start:end]

            # Hybrid BPM Calculation for the Segment
            low_band = butter_lowpass_filter(segment, 200, sr)
            mid_band = butter_bandpass_filter(segment, 200, 5000, sr)
            high_band = butter_highpass_filter(segment, 5000, sr)

            onset_low = librosa.onset.onset_strength(y=low_band, sr=sr)   # Spectral Onset for Low
            transient_mid = transient_detection(mid_band)                 # Transient Detection for Mid
            transient_high = transient_detection(high_band)               # Transient Detection for High

            transient_mid_resampled = resample(transient_mid, len(onset_low))
            transient_high_resampled = resample(transient_high, len(onset_low))

            PeDF_low = compute_PeDF(onset_low)
            PeDF_mid = compute_PeDF(transient_mid_resampled)
            PeDF_high = compute_PeDF(transient_high_resampled)

            combined_PeDF = combine_PeDFs(PeDF_low, PeDF_mid, PeDF_high, weights)

            peaks, _ = find_peaks(combined_PeDF, height=np.mean(combined_PeDF), distance=30)
            if len(peaks) < 2:
                bpm_list.append(bpm_list[-1] if bpm_list else 0)
                continue

            peak_intervals = np.diff(peaks) * (512 / sr)
            average_interval = np.mean(peak_intervals)
            bpm = 60 / average_interval

            bpm_list.append(round(bpm, 2))

        print("--------------------------------------------------")

        return {
            "bpm_per_second": bpm_list,
            "song_bpm": round(song_bpm, 2),
            "song_tempo": song_tempo
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")