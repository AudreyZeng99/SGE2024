import numpy as np
import codecs
import operator
import json
from trainer import train_loader
from tester import test_loader, distance, load_ids_from_file, check_id_exists
import time
from datetime import datetime
import argparse
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


##############################################################################
#
#   Utility function to check and create directories
#
##############################################################################
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


##############################################################################
#
#   Related to the preprocessing before getting the explanations
#
##############################################################################
def load_entity_embeddings(check_file_path):
    entity_embeddings = {}
    entity_id_list = []
    ensure_directory_exists(os.path.dirname(check_file_path))
    with codecs.open(check_file_path, 'r', 'utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            entity_id = parts[0]
            entity_id = int(entity_id)
            entity_id_list.append(entity_id)
            embedding = list(map(float, parts[1].strip('[]').split(',')))
            entity_embeddings[entity_id] = embedding
    return entity_id_list, entity_embeddings


def check_id_exists(check_id, entity_id_list):
    return check_id in entity_id_list


def preprocess_test_set(test_emb_save_path, data_name, alpha, dim):
    check_file_path = f"{emb_save_path}entity_{dim}dim_{data_name}_batch200"
    start_time = time.time()  # Start time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    _, _, train_triple = train_loader(f"../dataset/{data_name}/")  # Read train_triple for filtering training method

    entity_dict, relation_dict, test_triple = \
        test_loader(f"{emb_save_path}entity_{dim}dim_{data_name}_batch200", f"{emb_save_path}relation_{dim}dim_{data_name}_batch200", f"../dataset/{data_name}/test.txt")

    entity_id_list, _ = load_entity_embeddings(check_file_path)

    testing_set = {}
    test = 0

    for triple in test_triple:
        h = triple[0]
        r = triple[1]
        t = triple[2]
        triple_id = (h, r, t)

        test_t_exists = check_id_exists(check_id=h, entity_id_list=entity_id_list)
        test_h_exists = check_id_exists(check_id=t, entity_id_list=entity_id_list)

        if test_t_exists and test_h_exists:
            h_emb = entity_dict[h]
            r_emb = relation_dict[r]
            t_emb = entity_dict[t]
            triple_emb = [h_emb.tolist(), r_emb.tolist(), t_emb.tolist()]
            triple_id_str = f"{h},{r},{t}"
            testing_set[triple_id_str] = triple_emb
            test += 1
        else:
            continue

    print("The size of the testing set is ", test)
    print("Finished testing set generation.")

    # Ensure directory exists before saving
    ensure_directory_exists(test_emb_save_path)

    # Store testing_set into JSON file
    test_emb_save_file = f"{test_emb_save_path}test_emb_set_{data_name}_set.json"
    with open(test_emb_save_file, 'w') as json_file:
        json.dump(testing_set, json_file, indent=4)

    print(f"Testing set saved to {test_emb_save_file}")

    return test_emb_save_file


def read_and_process_json(file_path):
    # Read the JSON file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    test_head_emb = {}
    test_relation_emb = {}
    test_tail_emb = {}
    test_triple = {}
    test_entity_emb = {}

    for key, value in data.items():
        hid, rid, tid = map(int, key.split(','))
        triple = (hid, rid, tid)
        list1, list2, list3 = value

        test_head_emb[hid] = list1
        test_relation_emb[rid] = list2
        test_tail_emb[tid] = list3
        test_triple[triple] = [list1, list2, list3]

    for key, value in test_head_emb.items():
        test_entity_emb[key] = value
    for key, value in test_tail_emb.items():
        if key not in test_entity_emb:
            test_entity_emb[key] = value

    return test_triple, test_entity_emb, test_head_emb, test_relation_emb, test_tail_emb


def save_to_txt(file_path, data_dict):
    # Ensure directory exists before saving
    ensure_directory_exists(os.path.dirname(file_path))

    with open(file_path, 'w') as f:
        for key, value in data_dict.items():
            f.write(f"{key}\t{value}\n")


class ExplainWithPrototype:
    def __init__(self, entity_dict, relation_dict, train_triple, test_triple, test_entity_emb, check_file_path, dim,
                 alpha, mode, isFit=True):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.test_entity_dict = test_entity_emb
        self.train_triple = train_triple
        print(len(self.test_entity_dict), len(self.relation_dict), len(self.test_triple), len(self.train_triple))
        self.check_file_path = check_file_path
        self.check_entity_emb_dict = {}
        self.explain_triple = test_triple
        self.isFit = isFit
        self.dim = dim
        self.mode = mode

        self.head_emb_shifted = entity_dict
        self.relation_emb_shifted = relation_dict
        self.tail_emb_shifted = entity_dict

        self.alpha = alpha
        self.prototype = ()

        self._hits1 = 0
        self._MRR = 0
        self.hits1 = 0
        self.MRR = 0

        self.skip = 0
        self.skip_t = 0
        self.skip_h = 0
        self.skip_r = 0

        self.prototype_generator()
        self.embedding_shifting()

    def prototype_generator(self):
        print("*************************************")
        print("Prototype generating...")
        counter = 0
        h_p = np.zeros(self.dim)  # Initialize a zero vector with dimension dim
        r_p = np.zeros(self.dim)  # Initialize a zero vector with dimension dim
        t_p = np.zeros(self.dim)  # Initialize a zero vector with dimension dim

        for train_h_id, train_r_id, train_t_id in self.train_triple:
            h_emb = self.entity_dict[train_h_id]
            r_emb = self.relation_dict[train_r_id]
            t_emb = self.entity_dict[train_t_id]

            h_p += h_emb
            r_p += r_emb
            t_p += t_emb

            counter += 1

        h_p /= counter
        r_p /= counter
        t_p /= counter

        self.prototype = (h_p, r_p, t_p)
        print("Successfully generated prototype!")

        # Ensure directory exists before saving
        output_dir = f'../exp/{data_name}/explanations/'
        ensure_directory_exists(output_dir)

        # Save to JSON file
        prototype_dict = {
            "h_p": h_p.tolist(),
            "r_p": r_p.tolist(),
            "t_p": t_p.tolist()
        }

        with open(f'{output_dir}{data_name}_prototypes.json', 'w') as json_file:
            json.dump(prototype_dict, json_file)

        print("Successfully saved prototype to JSON file!")


    def embedding_shifting(self):
        step = 1
        start = time.time()
        print("Phase1: Embedding Shifting")
        # Phase 1: Apply shifting
        counter = 0
        hits = 0
        reciprocal_rank_sum = 0

        # Generate the prototype
        self.prototype_generator()

        h_p = self.prototype[0]
        r_p = self.prototype[1]
        t_p = self.prototype[2]

        print("***")

        shifted_triplet_set = {}
        for target in self.test_triple.keys():
            h_target, r_target, t_target = target

            # Initialize ranking dictionary
            rank_tail_dict = {}
            for t_id in self.test_entity_dict.keys():
                h_emb = np.array(self.test_entity_dict[h_target])
                r_emb = np.array(self.relation_dict[r_target])
                t_emb = np.array(self.test_entity_dict[t_id])
                # Calculate the distance for tail prediction
                rank_tail_dict[t_id] = distance(h_emb, r_emb, t_emb)

            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))
            # Check top ranking and apply shifting
            for i in range(len(rank_tail_sorted)):
                if t_target == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits += 1
                    elif i > 0:
                        h_original_emb = self.explain_triple[target][0]
                        r_original_emb = self.explain_triple[target][1]
                        t_original_emb = self.explain_triple[target][2]
                        # Apply shifting to the current triple embeddings
                        self.explain_triple[target][0] += self.alpha * (h_p - self.explain_triple[target][0])
                        self.explain_triple[target][1] += self.alpha * (r_p - self.explain_triple[target][1])
                        self.explain_triple[target][2] += self.alpha * (t_p - self.explain_triple[target][2])
                        print("successfully shifting triple", target, "!")

                        # Convert the target tuple to a string as the key
                        target_key_str = f"{h_target},{r_target},{t_target}"

                        shifted_triplet_set[target_key_str] = {
                            "original": {
                                "h_emb": h_original_emb,
                                "r_emb": r_original_emb,
                                "t_emb": t_original_emb
                            },
                            "shifted": {
                                "hsft_emb": self.explain_triple[target][0].tolist(),  # Convert numpy array to list
                                "rsft_emb": self.explain_triple[target][1].tolist(),
                                "tsft_emb": self.explain_triple[target][2].tolist()
                            }
                        }
                        print("successfully saved the embeddings.")

                    reciprocal_rank_sum += 1 / (i + 1)
                    break

        output_dir = f"../exp/{data_name}/explanations/"
        os.makedirs(output_dir, exist_ok=True)

        output_file_path = os.path.join(output_dir, f"shifted_triplet_set_{data_name}.json")
        with open(output_file_path, 'w') as f:
            json.dump(shifted_triplet_set, f, indent=4)
        # Output final evaluation results
        self.hits1 = hits / len(self.explain_triple)
        self.MRR = reciprocal_rank_sum / len(self.explain_triple)
        print(f"Test Hits@1: {self.hits1}, Test MRR: {self.MRR}")

        print('There we have', counter, ' that are not hit@1 triple finished the embedding shifting.')
        print("Phase2: Explanation Evaluation")
        # Phase 2: Recalculate distances and evaluate
        hits = 0
        reciprocal_rank_sum = 0


        # Directly test on the shifting set which has already done embedding shifting
        for target in self.explain_triple.keys():
            h_target, r_target, t_target = target
            # Initialize ranking dictionary
            rank_tail_dict = {}
            for t_id in self.test_entity_dict.keys():
                if t_id != t_target:
                    # When it is not the target, t_emb is not shifted
                    h_emb = np.array(self.test_entity_dict[h_target])
                    r_emb = np.array(self.relation_dict[r_target])
                    t_emb = np.array(self.test_entity_dict[t_id])
                else:
                    # When it is the target, t_emb is shifted
                    if mode == 't':
                        h_emb = np.array(self.test_entity_dict[h_target])
                        r_emb = np.array(self.relation_dict[r_target])
                        t_emb = np.array(self.explain_triple[target][2])
                    elif mode == 'h':
                        h_emb = np.array(self.explain_triple[target][0])
                        r_emb = np.array(self.relation_dict[r_target])
                        t_emb = np.array(self.test_entity_dict[t_id])
                    elif mode == 'r':
                        h_emb = np.array(self.explain_triple[target][0])
                        r_emb = np.array(self.explain_triple[target][1])
                        t_emb = np.array(self.test_entity_dict[t_id])
                    elif mode == 'all':
                        h_emb = np.array(self.explain_triple[target][0])
                        r_emb = np.array(self.explain_triple[target][1])
                        t_emb = np.array(self.explain_triple[target][2])
                    else:
                        print("You forgot to set the mode!")
                # Calculate the distance for tail prediction
                # Check if the ranking has improved after shifting
                rank_tail_dict[t_id] = distance(h_emb, r_emb, t_emb)

            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))
            # Check top ranking and apply shifting
            for i in range(len(rank_tail_sorted)):
                if t_target == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits += 1
                    reciprocal_rank_sum += 1 / (i + 1)
                    break

        # Output final evaluation results
        self._hits1 = hits / len(self.explain_triple)
        self._MRR = reciprocal_rank_sum / len(self.explain_triple)
        print(f"Final Hits@1: {self._hits1}, Final MRR: {self._MRR}")
        print(f"Total execution time: {time.time() - start:.2f} seconds")
        print(
            f"Skip number: {self.skip}, Skipped tail number: {self.skip_t}, Skipped Head Number: {self.skip_h}, Skipped relation Number: {self.skip_r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ExplainWithPrototype Parameters")
    parser.add_argument('--data_name', type=str, default='wn18rr', required=False, help='Name of the dataset')
    parser.add_argument('--alpha', type=float, default=0.8, required=False, help='Alpha value for embedding shifting')
    parser.add_argument('--dim', type=int, default=50, required=False, help='The number of embedding dimensions.')
    parser.add_argument('--mode', type=str, default='t', required=False)

    args = parser.parse_args()
    data_name = args.data_name
    alpha = args.alpha
    dim = args.dim
    mode = args.mode

    emb_save_path = f'../exp/{data_name}/TransE/'
    test_res_save_path = f'../exp/{data_name}/results/'
    test_emb_save_path = f'../exp/{data_name}/test/'

    print("Dataset:", data_name, " Alpha =", alpha, " Embedding dimension =", dim)
    check_file_path = f"{emb_save_path}entity_{dim}dim_{data_name}_batch200"

    ensure_directory_exists(emb_save_path)
    ensure_directory_exists(test_res_save_path)
    ensure_directory_exists(test_emb_save_path)

    _, _, train_triple = train_loader(f"../dataset/{data_name}/")

    entity_dict, relation_dict, _ = \
        test_loader(f"{emb_save_path}entity_{dim}dim_{data_name}_batch200", f"{emb_save_path}relation_{dim}dim_{data_name}_batch200",
                    f"../dataset/{data_name}/test.txt")

    test_emb_save_file = preprocess_test_set(test_emb_save_path=test_emb_save_path, data_name=data_name, alpha=alpha, dim=dim)

    test_triple, test_entity_emb, test_head_emb, test_relation_emb, test_tail_emb = read_and_process_json(test_emb_save_file)

    save_to_txt(f'{test_emb_save_path}{data_name}_test_head_emb.txt', test_head_emb)
    save_to_txt(f'{test_emb_save_path}{data_name}_test_relation_emb.txt', test_relation_emb)
    save_to_txt(f'{test_emb_save_path}{data_name}_test_tail_emb.txt', test_tail_emb)

    explain = ExplainWithPrototype(entity_dict, relation_dict, train_triple, test_triple, test_entity_emb,
                                   check_file_path, dim, alpha, mode, isFit=False)

    output_dir = f"../exp/{data_name}/explanations/"
    ensure_directory_exists(output_dir)

    data_to_save = {
        "test_hit": explain.hits1,
        "test_mrr": explain.MRR,
        "shifted_hit": explain._hits1,
        "shifted_mrr": explain._MRR,
        "explain_hit": explain._hits1 - explain.hits1,
        "explain_mrr": explain._MRR - explain.MRR
    }

    with open(os.path.join(output_dir, f"explanation_results_{data_name}.json"), 'w') as f:
        json.dump(data_to_save, f, indent=4)

    print(f"Results saved to {output_dir}explanation_results_{data_name}.json")