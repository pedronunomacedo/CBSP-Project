import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from app.models.tempoRanges import tempo_ranges

def applyDBSCANalgorithm(bpm_samples):
    bpm_samples = np.array(bpm_samples).reshape(-1, 1)  # Reshape to 2D array

    # Step 2b: Use DBSCAN Clustering to Find Dominant BPM
    dbscan = DBSCAN(eps=2, min_samples=5)  # Adjust `eps` and `min_samples` as needed
    dbscan_labels = dbscan.fit_predict(bpm_samples)

    # Determine the largest cluster as the dominant BPM
    dbscan_cluster_counts = Counter(dbscan_labels)
    if -1 in dbscan_cluster_counts:  # Ignore noise points
        dbscan_cluster_counts.pop(-1)
    dominant_dbscan_cluster = dbscan_cluster_counts.most_common(1)[0][0]
    dominant_bpm_dbscan = bpm_samples[dbscan_labels == dominant_dbscan_cluster].mean()

    return dominant_bpm_dbscan

def categorizeMusicTempo(song_bpm):
    for tempo_name, min_bpm, max_bpm in tempo_ranges:
        if min_bpm <= song_bpm <= max_bpm:
            return tempo_name
    return "Unknown"  # If no range matches, return "Unknown"