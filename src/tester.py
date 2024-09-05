import numpy as np
import codecs
import operator
import json
from trainer import train_loader, entity2id, relation2id
import time
import os
import argparse

def test_loader(entity_file, relation_file, test_file):
    # Load entities, relations, and test triples from files
    entity_dict = {}
    relation_dict = {}
    test_triple = []

    with codecs.open(entity_file, encoding='utf-8') as e_f:
        lines = e_f.readlines()
        for line in lines:
            entity, embedding = line.strip().split('\t')  # Get entity and its vector
            embedding = np.array(json.loads(embedding))
            entity_dict[int(entity)] = embedding  # Map entity and vector to a dictionary

    with codecs.open(relation_file, encoding='utf-8') as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation, embedding = line.strip().split('\t')  # Get relation and its vector
            embedding = np.array(json.loads(embedding))
            relation_dict[int(relation)] = embedding  # Map relation and vector to a dictionary

    with codecs.open(test_file, encoding='utf-8') as t_f:
        lines = t_f.readlines()
        for line in lines:
            triple = line.strip().split('\t')  # Get test set data
            if len(triple) != 3:  # Ensure proper triple format
                continue
            h_ = entity2id[triple[0]]
            r_ = relation2id[triple[1]]
            t_ = entity2id[triple[2]]
            test_triple.append(tuple((h_, r_, t_)))

    return entity_dict, relation_dict, test_triple

def distance(h, r, t):
    return np.linalg.norm(h + r - t)

class Test:
    def __init__(self, entity_dict, relation_dict, test_triple, train_triple, isFit=True):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.train_triple = train_triple
        print(len(self.entity_dict), len(self.relation_dict), len(self.test_triple), len(self.train_triple))
        self.isFit = isFit

        self.hits1 = 0
        self.MRR = 0

    def rank(self):
        hits = 0
        reciprocal_rank_sum = 0
        step = 1
        start = time.time()
        for triple in self.test_triple:
            test_h_id, _, test_t_id = triple
            if test_h_id not in self.entity_dict or test_t_id not in self.entity_dict:
                continue

            rank_head_dict = {}
            rank_tail_dict = {}

            for entity in self.entity_dict.keys():
                if self.isFit:
                    if [entity, triple[1], triple[2]] not in self.train_triple:
                        h_emb = self.entity_dict[entity]
                        r_emb = self.relation_dict[triple[1]]
                        t_emb = self.entity_dict[triple[2]]
                        rank_head_dict[entity] = distance(h_emb, r_emb, t_emb)
                else:
                    h_emb = self.entity_dict[entity]
                    r_emb = self.relation_dict[triple[1]]
                    t_emb = self.entity_dict[triple[2]]
                    rank_head_dict[entity] = distance(h_emb, r_emb, t_emb)

                if self.isFit:
                    if [triple[0], triple[2], entity] not in self.train_triple:
                        h_emb = self.entity_dict[triple[0]]
                        r_emb = self.relation_dict[triple[1]]
                        t_emb = self.entity_dict[entity]
                        rank_tail_dict[entity] = distance(h_emb, r_emb, t_emb)
                else:
                    h_emb = self.entity_dict[triple[0]]
                    r_emb = self.relation_dict[triple[1]]
                    t_emb = self.entity_dict[entity]
                    rank_tail_dict[entity] = distance(h_emb, r_emb, t_emb)

            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1))
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))

            for i in range(len(rank_tail_sorted)):
                if triple[2] == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits += 1
                    reciprocal_rank_sum += 1 / (i + 1)
                    break

            step += 1
            if step % 200 == 0:
                end = time.time()
                print("step: ", step, " ,hit_top1_rate: ", hits / (2 * step), " ,MRR ", reciprocal_rank_sum / (2 * step),
                      'time of testing one triple: %s' % (round((end - start), 3)))
                start = end
        self.hits1 = hits / (2 * len(self.test_triple))
        self.MRR = reciprocal_rank_sum / (2 * len(self.test_triple))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding Evaluation')
    parser.add_argument('--data_name', type=str, default='wn18rr', help='Name of the dataset')
    parser.add_argument('--emb_save_path', type=str, default='../exp/wn18rr/TransE/', help='Path to the saved embeddings')
    parser.add_argument('--is_fit', type=bool, default=True, help='Fit setting for testing')

    args = parser.parse_args()

    data_name = args.data_name
    emb_save_path = args.emb_save_path
    is_fit = args.is_fit

    _, _, train_triple = train_loader(f"../dataset/{data_name}/")

    entity_dict, relation_dict, test_triple = test_loader(
        f"{emb_save_path}entity_50dim_{data_name}_batch200",
        f"{emb_save_path}relation_50dim_{data_name}_batch200",
        f"../dataset/{data_name}/test.txt"
    )

    test = Test(entity_dict, relation_dict, test_triple, train_triple, isFit=is_fit)

    test.rank()
    print("Entity Hits@1: ", test.hits1)
    print("Entity MRR: ", test.MRR)

    exp_dir = f"../exp/{data_name}/results/"
    os.makedirs(exp_dir, exist_ok=True)

    data_to_save = {
        "hits1": test.hits1,
        "mrr": test.MRR
    }

    exp_file_path = os.path.join(exp_dir, f"test_{data_name}.json")

    with open(exp_file_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
