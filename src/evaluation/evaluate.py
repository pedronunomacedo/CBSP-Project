import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, accuracy_score

def evaluate_bpm_and_tempo(file_path):
    # Load JSON data
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}.")
        return
    
    # Initialize variables for metrics
    bpm_librosa = []
    bpm_mine = []
    bpm_hybrid = []
    
    tempo_librosa = []
    tempo_mine = []
    tempo_hybrid = []
    
    # Matching and not-matching counters
    matching_mine = 0
    not_matching_mine = 0
    matching_hybrid = 0
    not_matching_hybrid = 0
    
    # Populate data, skipping entries with missing or invalid BPM
    for entry in data:
        if isinstance(entry.get("librosaBpm"), (int, float)) and isinstance(entry.get("mineBmp"), (int, float)) and isinstance(entry.get("hybridBpm"), (int, float)):
            bpm_librosa.append(entry["librosaBpm"])
            bpm_mine.append(entry["mineBmp"])
            bpm_hybrid.append(entry["hybridBpm"])
        if entry.get("librosaTempo") and entry.get("mineTempo") and entry.get("hybridTempo"):
            tempo_librosa.append(entry["librosaTempo"])
            tempo_mine.append(entry["mineTempo"])
            tempo_hybrid.append(entry["hybridTempo"])
            # Check for matching
            if entry["librosaTempo"] == entry["mineTempo"]:
                matching_mine += 1
            else:
                not_matching_mine += 1
            if entry["librosaTempo"] == entry["hybridTempo"]:
                matching_hybrid += 1
            else:
                not_matching_hybrid += 1
    
    # Convert to numpy arrays for numerical calculations
    bpm_librosa = np.array(bpm_librosa)
    bpm_mine = np.array(bpm_mine)
    bpm_hybrid = np.array(bpm_hybrid)
    
    # BPM Metrics
    if len(bpm_librosa) > 0:
        mae_mine = mean_absolute_error(bpm_librosa, bpm_mine)
        mae_hybrid = mean_absolute_error(bpm_librosa, bpm_hybrid)
        rmse_mine = np.sqrt(mean_squared_error(bpm_librosa, bpm_mine))
        rmse_hybrid = np.sqrt(mean_squared_error(bpm_librosa, bpm_hybrid))
    else:
        mae_mine = mae_hybrid = rmse_mine = rmse_hybrid = None
    
    # Tempo Classification Metrics
    if len(tempo_librosa) > 0:
        accuracy_mine = accuracy_score(tempo_librosa, tempo_mine)
        accuracy_hybrid = accuracy_score(tempo_librosa, tempo_hybrid)
        
        report_mine = classification_report(tempo_librosa, tempo_mine, zero_division=0, output_dict=True)
        report_hybrid = classification_report(tempo_librosa, tempo_hybrid, zero_division=0, output_dict=True)
        
        # Extract Recall and F1-Score from the report
        recall_mine = report_mine["macro avg"]["recall"]
        recall_hybrid = report_hybrid["macro avg"]["recall"]
        f1_mine = report_mine["macro avg"]["f1-score"]
        f1_hybrid = report_hybrid["macro avg"]["f1-score"]
    else:
        accuracy_mine = accuracy_hybrid = recall_mine = recall_hybrid = f1_mine = f1_hybrid = None
    
    # Display Results
    print("BPM Evaluation:")
    if mae_mine is not None:
        print(f"Mean Absolute Error (Mine vs. Librosa): {mae_mine:.2f}")
        print(f"Mean Absolute Error (Hybrid vs. Librosa): {mae_hybrid:.2f}")
        print(f"Root Mean Square Error (Mine vs. Librosa): {rmse_mine:.2f}")
        print(f"Root Mean Square Error (Hybrid vs. Librosa): {rmse_hybrid:.2f}")
    else:
        print("No valid BPM data available for evaluation.")
    
    print("\nTempo Evaluation:")
    if accuracy_mine is not None:
        print(f"Accuracy (Mine vs. Librosa): {accuracy_mine:.2%}")
        print(f"Accuracy (Hybrid vs. Librosa): {accuracy_hybrid:.2%}")
        print(f"Recall (Mine vs. Librosa): {recall_mine:.2%}")
        print(f"Recall (Hybrid vs. Librosa): {recall_hybrid:.2%}")
        print(f"F1-Score (Mine vs. Librosa): {f1_mine:.2%}")
        print(f"F1-Score (Hybrid vs. Librosa): {f1_hybrid:.2%}")
        print(f"Matching Tempo (Mine vs. Librosa): {matching_mine}")
        print(f"Not-Matching Tempo (Mine vs. Librosa): {not_matching_mine}")
        print(f"Matching Tempo (Hybrid vs. Librosa): {matching_hybrid}")
        print(f"Not-Matching Tempo (Hybrid vs. Librosa): {not_matching_hybrid}")
    else:
        print("No valid tempo data available for evaluation.")

# Call the function with the JSON file
evaluate_bpm_and_tempo("results.json")
