import numpy as np
import hashlib
from sklearn.neighbors import NearestNeighbors

# Function to generate audio fingerprint from peaks (frequency, time)
def generate_fingerprint(peaks):
    hashes = []
    for peak in peaks:
        freq, time = peak
        hash_value = hashlib.sha256(f"{freq},{time}".encode()).hexdigest()
        hashes.append(hash_value)
    return hashes

# Function to convert hashes to a vector (for LSH)
def hash_to_vector(hashes):
    return np.array([int(h, 16) for h in hashes])  # Convert hex to integers

# Example: fingerprints of database songs (as hash vectors)
database_fingerprints = [
    hash_to_vector(generate_fingerprint([(300, 1.0), (800, 1.2), (1500, 1.5)])),
    hash_to_vector(generate_fingerprint([(100, 0.5), (600, 1.1), (1300, 1.4)])),
    # Add more fingerprints for other songs
]

# Query song fingerprint
query_fingerprint = hash_to_vector(generate_fingerprint([(200, 1.0), (800, 1.2), (1500, 1.5)]))

# LSH using Nearest Neighbors
nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='hamming')  # 'hamming' for hash-based comparison
nn.fit(database_fingerprints)

# Find the nearest song match
distances, indices = nn.kneighbors([query_fingerprint])

print("Closest match song index:", indices[0])
print("Distance:", distances[0])