import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.core.bpm_calculator import calculate_bpm, calculate_bpm_from_peaks
from app.core.detect_peaks import detect_spectral_peaks
from app.utils.file_conversion import applyDBSCANalgorithm, categorizeMusicTempo
from app.utils.song_peaks import findAudioHighPeaks, get_hash_peak
from app.utils.filters import bandpass_filter, bandpass_filter_chebyshev
import io
import numpy as np
from pydub import AudioSegment
import librosa
from scipy.signal import find_peaks
import hashlib
import SongNameSplit
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from ..utils.hybridApproach import HybridMultibandApproach


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

        # Segment length for 1-second segments
        segment_length = sr  # Number of samples in one second
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
        segment_length = 1 * sample_rate # Number of samples in one second
        bpm_per_second = []

        # Process each 1-second segment
        for start in range(0, len(filtered_y), segment_length):
            end = start + segment_length
            segment = filtered_y[start:end]

            # Ensure segment is exactly 1 second
            if len(segment) < segment_length:
                continue

            # # Detect peaks in the segment
            # distance = int(sample_rate * 60 / 200)  # Adjust for up to 200 BPM
            # peaks, _ = find_peaks(segment, distance=distance, height=np.mean(segment) * 0.5)

            # # Calculate BPM based on peak intervals
            # if len(peaks) > 1:
            #     peak_intervals = np.diff(peaks) / sample_rate  # Convert intervals to seconds
            #     avg_interval = np.mean(peak_intervals)
            #     bpm = 60 / avg_interval
            # elif bpm_per_second:
            #     # Interpolate using the last valid BPM if available
            #     bpm = bpm_per_second[-1]
            # else:
            #     bpm = 0  # Use 0 if there's no previous BPM to use

            # Step 1: Compute the energy envelope
            energy = np.abs(segment) ** 2 # Square of the amplitude ( energy = amplitude^2 )
            window_size = int(0.1 * sample_rate)  # 100ms window for smoothing
            smoothed_energy = uniform_filter1d(energy, size=window_size)  # Smooth the energy in each sample value by looking to its two direct neighbours [1, 4, 6] -> [((1+1+4)/3), ((1+4+6)/3), ((4+6+6)/3)]

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
                bpm = 0  # If no peaks, set BPM to 0

            bpm_per_second.append(round(bpm, 2))

        # Optionally, calculate an average BPM for the entire song
        if bpm_per_second:
            song_bpm = applyDBSCANalgorithm(bpm_per_second)
            song_bpm = round(song_bpm, 2)
            song_tempo = categorizeMusicTempo(song_bpm)
        else:
            song_bpm = 0
            song_tempo = "Unknown"

        return {
            "bpm_per_second": bpm_per_second,
            "song_bpm": song_bpm,
            "song_tempo": song_tempo
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    

























@router.post("/bpm")
async def get_bpm(file: UploadFile = File(...)):
    try:
        # Read the file content
        audio_data = await file.read()

        # Convert mp3 to wav in memory
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Load audio data into librosa format
        y, sr = librosa.load(wav_io, sr=None)

        # Step 1: Detect peaks in the audio spectrum
        peaks = detect_spectral_peaks(y, sr)

        # Calculate the BPM by passing the audio data to calculate_bpm
        bpm = calculate_bpm_from_peaks(peaks)

        return {"bpm": peaks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    



@router.post("/detect_music")
async def detect_music(file: UploadFile = File(...)):
    try:
        # Read the file content
        audio_data = await file.read()

        # Convert mp3 to wav in memory
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Convert audio to numpy array
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        samples = samples / np.max(np.abs(samples))  # Normalize audio

        high_peaks = findAudioHighPeaks(samples=samples, sample_rate=sample_rate)

        # Generate hash for each peak
        peak_hashes = []
        for peak in high_peaks:
            peak_hashes.append(get_hash_peak(peak=peak))

        # Combine all peak hashes to generate a final hash for the song
        combined_hash_input = ''.join(peak_hashes)
        final_hash_object = hashlib.sha256(combined_hash_input.encode())
        final_hash = final_hash_object.hexdigest()

        return {
            "peaks": final_hash
        }

    except Exception as e:
        return {"error": str(e)}
    






# def clean_data(data):
#     if not np.isfinite(data).all():
#         cleaned_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
#         return cleaned_data
#     return data

# def check_data(data, step_description):
#     if not np.isfinite(data).all():
#         print(f"Data check failed at {step_description}. Data contains NaN or infinite values.")
#         raise ValueError(f"Data at {step_description} contains NaN or infinite values.")
    
# def inspect_audio_properties(audio):
#     print("Mean amplitude:", np.mean(audio))
#     print("Max amplitude:", np.max(audio))
#     print("Min amplitude:", np.min(audio))
#     print("DC offset (mean amplitude):", np.mean(audio))

# def normalize_data(data):
#     max_val = np.max(data)
#     if max_val > 0:  # Avoid division by zero
#         return data / max_val
#     return data

# async def load_audio(file):
#     try:
#         print("DEBUG: Loading audio file!")
#         # Read the uploaded file into a BytesIO buffer
#         contents = await file.read()
#         buffer = io.BytesIO(contents)
#         audio, sr = librosa.load(buffer, sr=None)  # librosa handles the buffer directly
#         print("DEBUG: Before cleaning the audio file data!")
#         audio = clean_data(audio) 
#         print("DEBUG: After cleaning the audio file data!")

#         # Check for NaN or infinite values in the audio data
#         print("DEBUG: Before checking if the audio file as NaN or infinite values!")
#         if not np.isfinite(audio).all():
#             raise ValueError("Audio data contains NaN or infinite values.")
#         print("DEBUG: Before checking if the audio file as NaN or infinite values!")
        
#         return audio, sr
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
# def split_into_bands(audio, sr):
#     print("Checking data before filtering...")
#     if not np.isfinite(audio).all():
#         print("Data contains non-finite values before filtering.")
#         audio = clean_data(audio)
        
#     lfb = bandpass_filterV2(audio, 1, 200, sr)     # Low Frequency Band
#     check_data(lfb, "Low Frequency Band")
#     mfb = bandpass_filterV2(audio, 200, 5000, sr)  # Middle Frequency Band
#     check_data(mfb, "Middle Frequency Band")
#     hfb = bandpass_filterV2(audio, 5000, sr * 0.49, sr) # High Frequency Band
#     check_data(hfb, "High Frequency Band")

#     return lfb, mfb, hfb

# # Low Frequency ODF
# def get_odf_low(audio, sr):
#     onset_env = librosa.onset.onset_strength(y=audio, sr=sr, aggregate=np.mean)
#     normalized_onset_env = normalize_data(onset_env)
#     check_data(normalized_onset_env, "ODF Middle")
#     return normalized_onset_env

# # Middle Frequency ODF
# def get_odf_middle(audio, sr):
#     # Spectral flux is the default for onset_strength with no aggregation
#     onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
#     normalized_onset_env = normalize_data(onset_env)
#     check_data(normalized_onset_env, "ODF Middle")
#     return normalized_onset_env

# # High Frequency ODF
# def get_odf_high(audio, sr):
#     S = np.abs(librosa.stft(audio))
#     check_data(S, "STFT High before DB conversion")
#     db_S = librosa.amplitude_to_db(S)
#     check_data(db_S, "STFT High after DB conversion")
#     onset_env = librosa.onset.onset_strength(S=db_S, sr=sr, feature=librosa.feature.spectral_contrast, fmin=2000.0)
#     normalized_onset_env = normalize_data(onset_env)
#     check_data(normalized_onset_env, "ODF High")
#     return normalized_onset_env


# # Helper function to save plot to a file
# def save_plot_to_file(data, filename, title):
#     directory = "images"
#     if not os.path.exists(directory):
#         os.makedirs(directory)  # Create the directory if it does not exist
    
#     plt.figure(figsize=(10, 3))
#     plt.plot(data)
#     plt.title(title)
#     file_path = f"{directory}/{filename}.png"
#     plt.savefig(file_path)
#     plt.close()

#     return file_path

# def plot_periodicity(times, autocorr, peaks, title='Periodicity Detection', filename='periodicity'):
#     directory = "images"
#     if not os.path.exists(directory):
#         os.makedirs(directory)  # Create the directory if it does not exist

#     plt.figure(figsize=(10, 4))
#     plt.plot(autocorr, label='Autocorrelation')
#     plt.vlines(times, 0, autocorr[peaks], color='r', alpha=0.5, label='Peaks')
#     plt.title(title)
#     plt.xlabel('Lag')
#     plt.ylabel('Autocorrelation value')
#     plt.legend()
#     file_path = f"{directory}/{filename}.png"
#     plt.savefig(file_path)
#     plt.close()

#     return file_path


# def calculate_periodicity(odf, sr):
#     # Autocorrelation of the ODF
#     autocorr = np.correlate(odf, odf, mode='full')
#     autocorr = autocorr[autocorr.size // 2:]

#     # Detect peaks in the autocorrelation to find intervals
#     peaks = librosa.util.peak_pick(autocorr, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.1, wait=1)

#     # Convert peak locations to time intervals and return
#     times = peaks / float(sr)
#     return times, autocorr, peaks

# def weighting_function(period):
#     # Apply the weighting function to each period (time interval)
#     bpm = 60 / period  # Convert period (in seconds) to BPM
#     if 100 <= bpm <= 120:
#         return 1.5  # Prioritize common dance/pop music tempos
#     elif 60 <= bpm <= 180:
#         return 1.0  # Normal weight for reasonable music tempos
#     else:
#         return 0.5  # Deprioritize very slow or very fast tempos
    
# def apply_weights_to_periods(times, autocorr):
#     # Calculate periods (time differences between peaks)
#     periods = np.diff(times)  # Calculate intervals between peaks
    
#     # Apply weights to each period
#     weighted_autocorr = np.zeros_like(autocorr)
#     for i, period in enumerate(periods):
#         weight = weighting_function(period)  # Apply the weighting function
#         weighted_autocorr[i] = weight * autocorr[i]  # Apply weight to the corresponding autocorrelation value
    
#     return weighted_autocorr

# def combine_weighted_periodicities(times_low, times_mid, times_high, autocorr_low, autocorr_mid, autocorr_high):
#     # Apply weights to each frequency band's autocorrelation
#     weighted_ac_low = apply_weights_to_periods(times_low, autocorr_low)
#     weighted_ac_mid = apply_weights_to_periods(times_mid, autocorr_mid)
#     weighted_ac_high = apply_weights_to_periods(times_high, autocorr_high)
    
#     # Combine the weighted periodicities (PeDFs)
#     combined_pe_df = weighted_ac_low + weighted_ac_mid + weighted_ac_high
    
#     return combined_pe_df

# def estimate_combined_bpm(peaks_combined, sr):
#     # Calculate intervals between peaks (in samples)
#     intervals = np.diff(peaks_combined)
    
#     # Convert intervals to BPM (beats per minute)
#     bpm_values = (sr / intervals) * 60  # Convert intervals to BPM
#     return np.mean(bpm_values)  # Return average BPM


# @router.post("/improved_bpm_detector")
# async def get_mine_bpm_per_second(file: UploadFile = File(...)):
#     try:
#         # Get the audio data and sample rate
#         audio, sr = await load_audio(file)
#         print("DEBUG: After loading audio file")

#         inspect_audio_properties(audio)

#         print("DEBUG: Before splitting audio wave into frequency bands")
#         # Divide the audio into low, middle, and high frequency bands
#         lfb, mfb, hfb = split_into_bands(audio, sr)
#         print("DEBUG: After splitting audio wave into frequency bands")

#         print("DEBUG: Before getting the ODF for each frequency band")
#         # Detect onsets for each frequency band
#         onset_low = get_odf_low(lfb, sr)
#         print("- DEBUG: After getting the ODF for low frequency band")
#         onset_middle = get_odf_middle(mfb, sr)
#         print("- DEBUG: After getting the ODF for middle frequency band")
#         onset_high = get_odf_high(hfb, sr)
#         print("- DEBUG: After getting the ODF for high frequency band")
#         print("DEBUG: After getting the ODF for each frequency band")

#         print("DEBUG: Before saving the ODF plot for each frequency band")
#         # Save plots to files
#         low_path = save_plot_to_file(onset_low, "low_freq_band", "Low Frequency Band ODF")
#         mid_path = save_plot_to_file(onset_middle, "middle_freq_band", "Middle Frequency Band ODF")
#         high_path = save_plot_to_file(onset_high, "high_freq_band", "High Frequency Band ODF")
#         print("DEBUG: After saving the ODF plot for each frequency band")

#         # Assuming odf_low, odf_mid, and odf_high are the ODFs obtained from previous onset/transient detection steps
#         times_low, autocorr_low, peaks_low = calculate_periodicity(onset_low, sr)
#         times_mid, autocorr_mid, peaks_mid = calculate_periodicity(onset_middle, sr)
#         times_high, autocorr_high, peaks_high = calculate_periodicity(onset_high, sr)

#         low_periodicity_path = plot_periodicity(times_low, autocorr_low, peaks_low, 'Low Frequency Band Periodicity', 'low_frequency_periodicity')
#         middle_periodicity_path = plot_periodicity(times_mid, autocorr_mid, peaks_mid, 'Middle Frequency Band Periodicity', 'middle_frequency_periodicity')
#         high_periodicity_path = plot_periodicity(times_high, autocorr_high, peaks_high, 'High Frequency Band Periodicity', 'high_frequency_periodicity')

#         # Apply weighting and combine the PeDFs
#         combined_pe_df = combine_weighted_periodicities(times_low, times_mid, times_high, autocorr_low, autocorr_mid, autocorr_high)

#         # Detect peaks in the combined PeDF
#         combined_peaks = librosa.util.peak_pick(combined_pe_df, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.1, wait=1)

#         # Calculate the BPM from the combined peaks
#         combined_bpm = estimate_combined_bpm(combined_peaks, sr)

        
        
#         return {
#             "low_freq_band": low_path,
#             "mid_freq_band": mid_path,
#             "high_freq_band": high_path,
#             "low_periodicity_path": low_periodicity_path,
#             "middle_periodicity_path": middle_periodicity_path,
#             "high_periodicity_path": high_periodicity_path,
#             "combined_bpm": combined_bpm,
#             "musicTempo": categorizeMusicTempo(combined_bpm)
#         }
#     except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")
    

@router.get("/download/{filename}")
async def download_image(filename: str):
    file_path = f"images/{filename}.png"
    try:
        if os.path.exists(file_path):
            return FileResponse(path=file_path, filename=f"{filename}.png", media_type='image/png')
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing file: {str(e)}")
    




@router.get("/hybrid_approach_bpm")
async def get_hybrid_approach_bpm(file: UploadFile = File(...)):
    try:
        # Read the file content
        audio_data = await file.read()

        # Convert mp3 to wav in memory
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sample_rate = librosa.load(wav_io, sr=None)

        lfb, mfb, hfb = split_into_bands(y, sample_rate)

        # Check if y is a numpy array right after loading
        if not isinstance(y, np.ndarray):
            raise TypeError("Loaded audio data is not a numpy array.")

        print("[DEBUG] y shape: ", y.shape)
        print("[DEBUG] lbf shape: ", lfb.shape)
        print("[DEBUG] mfb shape: ", mfb.shape)
        print("[DEBUG] hfb shape: ", hfb.shape)

        plot_frequency_spectrum(y, sample_rate, title="Full Frequency Spectrum")
        plot_frequency_spectrum(lfb, sample_rate, title="Low Frequency Band")
        plot_frequency_spectrum(mfb, sample_rate, title="Middle Frequency Band")
        plot_frequency_spectrum(hfb, sample_rate, title="High Frequency Band")

        # Create a Hybrid Approach BPM Calculator object
        hybrid_calculator = HybridMultibandApproach()

        onset_env, onsets = hybrid_calculator.detect_onsets(lfb, sample_rate)

        onset_times = hybrid_calculator.convert_frames_to_times(onsets, sample_rate)

        transients, transients_times = hybrid_calculator.detect_transients(hfb, sample_rate)

        print("[DEBUG] onsets: ", onsets)
        print("[DEBUG] onset_times: ", onset_times)
        
        print("[DEBUG] transients: ", transients)
        print("[DEBUG] transients_times: ", transients_times)


        # Plotting
        plt.figure(figsize=(14, 5))
        plt.plot(librosa.times_like(onset_env, sr=sample_rate), onset_env, label='Onset Strength')
        plt.vlines(librosa.frames_to_time(onsets, sr=sample_rate), ymin=0, ymax=max(onset_env), color='r', linestyle='--', label='Detected Onsets')
        plt.title('Onset Strength Envelope with Detected Onsets')
        plt.xlabel('Time (s)')
        plt.ylabel('Strength')
        plt.legend()
        plt.savefig('onset_times.png')

        plt.subplot(2, 1, 2)
        plt.plot(librosa.times_like(transients_times, sr=sample_rate), transients_times, label='Transient Strength (HFB)')
        plt.vlines(transients_times, ymin=0, ymax=max(transients_times), color='r', linestyle='--', label='Detected Transients')
        plt.title('High Frequency Band Transient Strength and Detected Transients')
        plt.xlabel('Time (s)')
        plt.ylabel('Strength')
        plt.legend()
        plt.savefig('transient_times.png')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    return {
        "onsets": onset_times,
        "transients": transients_times
    }



def split_into_bands(audio, sr):
    print("Checking data before filtering...")
    if not np.isfinite(audio).all():
        print("Data contains non-finite values before filtering.")
        
    lfb = bandpass_filter_chebyshev(audio, 1, 200, sr)     # Low Frequency Band
    mfb = bandpass_filter_chebyshev(audio, 200, 5000, sr)  # Middle Frequency Band
    hfb = bandpass_filter_chebyshev(audio, 5000, sr * 0.49, sr) # High Frequency Band

    return lfb, mfb, hfb

def plot_frequency_spectrum(signal, sr, title="Frequency Spectrum"):
    # Compute the Fast Fourier Transform (FFT) of the signal
    fft = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft), 1/sr)
    
    # Only take the magnitude of the first half of the spectrum
    magnitude = np.abs(fft)[:len(fft)//2]
    freq = frequencies[:len(fft)//2]

    plt.figure(figsize=(10, 4))
    plt.plot(freq, magnitude)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, sr/2)  # Limit x-axis to Nyquist frequency
    plt.savefig(f"{title}.png")
