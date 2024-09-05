import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process shifted triplet data for normalization and analysis.')
parser.add_argument('--data_name', type=str, default='fb15k-237', help='Name of the dataset (e.g., fb15k-237)')
parser.add_argument('--visible_range', type=int, default=3, help='Number of visible triplets to display')
# parser.add_argument('--number_of_candidates', type=int, default=0, help='Number of candidates (default is 0)')
parser.add_argument('--dim', type=int, default=50,help='The dimension of embedding.')
args = parser.parse_args()

# Use command-line arguments
data_name = args.data_name
visible_range = args.visible_range
dim = args.dim
# number_of_candidates = args.number_of_candidates
# number_of_candidates = 0
# Set the file path based on the dataset name
filename = f'../exp/{data_name}/explanations/shifted_triplet_set_{data_name}_with_shifts.json'

# Read JSON file
with open(filename, 'r') as file:
    data = json.load(file)

# Initialize the total values for the vector dimensions
h_shift_total = np.zeros(dim)
r_shift_total = np.zeros(dim)
t_shift_total = np.zeros(dim)

# Record the count of h_shift_vector
h_shift_vector_count = 0

# Store the sum of hsft_emb, rsft_emb, tsft_emb for each triplet
hsft_emb_sums = []
rsft_emb_sums = []
tsft_emb_sums = []

# Traverse the data and calculate the total values
for key, value in data.items():
    if 'shifted' in value:
        h_shift_total += np.array(value['h_shift_vector'])
        r_shift_total += np.array(value['r_shift_vector'])
        t_shift_total += np.array(value['t_shift_vector'])
        h_shift_vector_count += 1

    # Calculate the sum for each triplet
    hsft_emb_sum = sum(value['shifted']['hsft_emb'])
    rsft_emb_sum = sum(value['shifted']['rsft_emb'])
    tsft_emb_sum = sum(value['shifted']['tsft_emb'])

    hsft_emb_sums.append(hsft_emb_sum)
    rsft_emb_sums.append(rsft_emb_sum)
    tsft_emb_sums.append(tsft_emb_sum)

# Normalize the total values
scaler = MinMaxScaler()

h_shift_total_normalized = scaler.fit_transform(h_shift_total.reshape(-1, 1)).flatten()
r_shift_total_normalized = scaler.fit_transform(r_shift_total.reshape(-1, 1)).flatten()
t_shift_total_normalized = scaler.fit_transform(t_shift_total.reshape(-1, 1)).flatten()

# Normalize hsft_emb_sums, rsft_emb_sums, tsft_emb_sums
hsft_emb_sums_normalized = scaler.fit_transform(np.array(hsft_emb_sums).reshape(-1, 1)).flatten()
rsft_emb_sums_normalized = scaler.fit_transform(np.array(rsft_emb_sums).reshape(-1, 1)).flatten()
tsft_emb_sums_normalized = scaler.fit_transform(np.array(tsft_emb_sums).reshape(-1, 1)).flatten()

# Round results to three decimal places
h_shift_total_normalized = np.round(h_shift_total_normalized, 3)
r_shift_total_normalized = np.round(r_shift_total_normalized, 3)
t_shift_total_normalized = np.round(t_shift_total_normalized, 3)
hsft_emb_sums_normalized = np.round(hsft_emb_sums_normalized, 3)
rsft_emb_sums_normalized = np.round(rsft_emb_sums_normalized, 3)
tsft_emb_sums_normalized = np.round(tsft_emb_sums_normalized, 3)

# Find maximum values and their indices
h_list = h_shift_total_normalized.tolist()
h_max = max(h_list)
h_max_index = h_list.index(h_max) + 1

r_list = r_shift_total_normalized.tolist()
r_max = max(r_list)
r_max_index = r_list.index(r_max) + 1

t_list = t_shift_total_normalized.tolist()
t_max = max(t_list)
t_max_index = t_list.index(t_max) + 1

print(f"h_max: {h_max}, index: {h_max_index}")
print(f"r_max: {r_max}, index: {r_max_index}")
print(f"t_max: {t_max}, index: {t_max_index}")

# Print normalized results
print("h_shift_vector normalized result: ", h_shift_total_normalized.tolist())
print("r_shift_vector normalized result: ", r_shift_total_normalized.tolist())
print("t_shift_vector normalized result: ", t_shift_total_normalized.tolist())

# Print the first three and the last triplet's normalized results
print("First three triplet hsft_emb normalized result: ", hsft_emb_sums_normalized.tolist()[:visible_range])
print("First three triplet rsft_emb normalized result: ", rsft_emb_sums_normalized.tolist()[:visible_range])
print("First three triplet tsft_emb normalized result: ", tsft_emb_sums_normalized.tolist()[:visible_range])
print("Last triplet hsft_emb normalized result: ", hsft_emb_sums_normalized.tolist()[-1])
print("Last triplet rsft_emb normalized result: ", rsft_emb_sums_normalized.tolist()[-1])
print("Last triplet tsft_emb normalized result: ", tsft_emb_sums_normalized.tolist()[-1])

number_of_candidates = len(hsft_emb_sums)
print("Number of candidate predictions:", number_of_candidates)
