import json
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Compute shift vectors for triplets and save to a new JSON file.')
parser.add_argument('--data_name', type=str, default='fb15k-237', help='Name of the dataset (e.g., fb15k-237)')

args = parser.parse_args()

# Use the command-line argument
data_name = args.data_name

# Load shifted triplet set from the provided file
with open(f'../exp/{data_name}/explanations/shifted_triplet_set_{data_name}.json', 'r') as file:
    shifted_triplet_set = json.load(file)

# Process each triplet element to compute the shift vectors
for key, value in shifted_triplet_set.items():
    # Extract the original and shifted embeddings
    h_emb = np.array(value["original"]["h_emb"])
    r_emb = np.array(value["original"]["r_emb"])
    t_emb = np.array(value["original"]["t_emb"])
    hsft_emb = np.array(value["shifted"]["hsft_emb"])
    rsft_emb = np.array(value["shifted"]["rsft_emb"])
    tsft_emb = np.array(value["shifted"]["tsft_emb"])

    # Compute the shift vectors
    h_shift_vector = (hsft_emb - h_emb).tolist()
    r_shift_vector = (rsft_emb - r_emb).tolist()
    t_shift_vector = (tsft_emb - t_emb).tolist()

    # Add shift vectors to the current triplet element
    value["h_shift_vector"] = h_shift_vector
    value["r_shift_vector"] = r_shift_vector
    value["t_shift_vector"] = t_shift_vector

# Save the modified data back to a new JSON file
with open(f'../exp/{data_name}/explanations/shifted_triplet_set_{data_name}_with_shifts.json', 'w') as file:
    json.dump(shifted_triplet_set, file)

print("Shift vectors computed and added successfully. The updated data has been saved to a new JSON file.")
