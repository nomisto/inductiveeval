from pykeen.datasets.inductive.ilp_teru import InductiveFB15k237, InductiveWN18RR, InductiveNELL
from pykeen.datasets.inductive import ILPC2022Large, ILPC2022Small
from pykeen.evaluation.rank_based_evaluator import SampledRankBasedEvaluator, RankBasedEvaluator

import logging
logging.getLogger("pykeen.evaluation.rank_based_evaluator").setLevel(logging.ERROR)


from collections import defaultdict
def get_collections(mapped_triples):
    relations = set()
    heads = defaultdict(set)
    tails = defaultdict(set)
    for triple in mapped_triples:
        (h,r,t) = tuple(triple.tolist())
        relations.add(r)
        heads[r].add(h)
        tails[r].add(t)
    return relations, heads, tails

def get_rules(mapped_triples_transductive, mapped_triples_inductive, num_entities, unseen_neg=0, min_prob=0.0, dynamic_prob=True):
    
    relations, heads, tails = get_collections(mapped_triples_transductive)
    relations_ind, heads_ind, tails_ind = get_collections(mapped_triples_inductive)

    rules_heads = defaultdict(list)
    rules_tails = defaultdict(list)
    for r1 in relations:
        for r2 in relations:
            """ if r1 == r2:
                continue """
            for dir1 in ["H", "T"]:
                for dir2 in ["H", "T"]:
                    from_ = heads[r1] if dir1 == "H" else tails[r1]
                    to_ = heads[r2] if dir2 == "H" else tails[r2]
                    from_ind = heads_ind[r1] if dir1 == "H" else tails_ind[r1]
                    to_ind = heads_ind[r2] if dir2 == "H" else tails_ind[r2]
                    intersection = from_.intersection(to_)
                    intersection_ind = from_ind.intersection(to_ind)
                    if len(intersection) > 0:
                        prob = len(intersection) / (len(to_) + unseen_neg)
                    else:
                        continue
                    predictions = to_ind
                    # prob = len(intersection) / (len(to_) + unseen_neg)
                    """ if prob > 0.1:
                        intersection_ind.update(to_ind) """
                    if dir1 == "H":
                        rules_heads[r1].append((prob, dir2, r2, intersection, predictions))
                    else:
                        rules_tails[r1].append((prob, dir2, r2, intersection, predictions))
                        
    for relation in rules_heads.keys():
        scores_baseline = torch.full((num_entities,), -1).float()
        touched_x = set()
        rules_heads[relation] = list(sorted(rules_heads[relation], key=lambda x: x[0], reverse=True))
        for (prob, dir2, r2, intersection, intersection_ind) in rules_heads[relation]:
            if prob < min_prob:
                continue
            for x in intersection_ind:
                touched_x.add(x)
                if scores_baseline[x] < 0.:
                    scores_baseline[x] = (prob if dynamic_prob else 1.)
                else:
                    scores_baseline[x] = max(scores_baseline[x], (prob if dynamic_prob else 1.))
        rules_heads[relation] = scores_baseline
        
    for relation in rules_tails.keys():
        scores_baseline = torch.full((num_entities,), -1).float()
        touched_x = set()
        rules_tails[relation] = list(sorted(rules_tails[relation], key=lambda x: x[0], reverse=True))
        for (prob, dir2, r2, intersection, intersection_ind) in rules_tails[relation]:
            if prob < min_prob:
                continue
            for x in intersection_ind:
                touched_x.add(x)
                if scores_baseline[x] < 0.:
                    scores_baseline[x] = (prob if dynamic_prob else 1.)
                else:
                    scores_baseline[x] = max(scores_baseline[x], (prob if dynamic_prob else 1.))
        rules_tails[relation] = scores_baseline
    
    return rules_heads, rules_tails


from torch import nn
import torch

class Wrapper(nn.Module):
    
    def read_predfile(self,path):
        with open(path, "r") as infile:
            content = infile.readlines()
        content = [x.strip() for x in content]
        
        for i in range(0,len(content),3):
            h,r,t = content[i].split(" ")
            h,r,t = (
                self.dataset.inductive_inference.entity_to_id[h],
                self.dataset.inductive_inference.relation_to_id[r],
                self.dataset.inductive_inference.entity_to_id[t]
            )
            
            scores_h = torch.full((self.dataset.inductive_inference.num_entities,), -1).float()
            scores_t = torch.full((self.dataset.inductive_inference.num_entities,), -1).float()
            
            if self.anyburl:
                heads = content[i+1][len("Heads: "):].split("\t")
                if len(heads) > 1:
                    for i_h in range(0, len(heads), 2):
                        try:
                            entity = self.dataset.inductive_inference.entity_to_id[heads[i_h]]
                            score = float(heads[i_h+1])
                            scores_h[entity] = score
                        except KeyError:
                            pass
                        
                tails = content[i+2][len("Tails: "):].split("\t")
                if len(tails) > 1:
                    for i_t in range(0, len(tails), 2):
                        try:
                            entity = self.dataset.inductive_inference.entity_to_id[tails[i_t]]
                            score = float(tails[i_t+1])
                            scores_t[entity] = score
                        except KeyError:
                            pass
                
            self.scores_heads[(h,r,t)] = scores_h
            self.scores_tails[(h,r,t)] = scores_t

    def __init__(self, path, dataset, anyburl=True, baseline=True, dynamic_prob=False):
        super().__init__()
        self.device = "cpu"
        self.anyburl=anyburl
        self.baseline=baseline
        self.dynamic_prob=dynamic_prob
        self.scores_heads = dict()
        self.scores_tails = dict()
        self.dataset = dataset
        if path is not None:
            self.read_predfile(path)
    
    def score_heads(self, triple):
        triple = tuple(triple.tolist())
        scores = self.scores_heads[triple]
        
        if self.baseline:
            scores_baseline = rules_heads[triple[1]]
            scores = torch.maximum(scores, scores_baseline * 0.0001 if self.anyburl else scores_baseline)                                
        return scores
    
    def score_tails(self, triple):
        triple = tuple(triple.tolist())
        scores = self.scores_tails[triple]
        
        if self.baseline:
            scores_baseline = rules_tails[triple[1]]
            scores = torch.maximum(scores, scores_baseline * 0.0001 if self.anyburl else scores_baseline)       
        return scores
        
    def predict(self, hrt_batch, target, slice_size, mode):
        if target == "head":
            return torch.vstack([self.score_heads(triple) for triple in hrt_batch]).to(self.device)
        elif target == "tail":
            return torch.vstack([self.score_tails(triple) for triple in hrt_batch]).to(self.device)
        else:
            raise
        

def get_positive_tail_mask(mapped_triples):
    mask = torch.zeros((mapped_triples.shape[0], dataset.inductive_inference.num_entities)).long()
    for i in range(mapped_triples.shape[0]):
        for t in all_hr_t[(mapped_triples[i][0].item(), mapped_triples[i][1].item())]:
            mask[i][t] = 1
    return mask.long()

def get_positive_head_mask(mapped_triples):
    mask = torch.zeros((mapped_triples.shape[0], dataset.inductive_inference.num_entities)).long()
    for i in range(mapped_triples.shape[0]):
        for h in all_tr_h[(mapped_triples[i][2].item(), mapped_triples[i][1].item())]:
            mask[i][h] = 1
    return mask.long()
    
def get_triples_str(mapped_triples):
    triples = []
    for i in range(mapped_triples.shape[0]):
        triple = mapped_triples[i]
        triple = (
            dataset.inductive_inference.entity_id_to_label[triple[0].item()],
            dataset.inductive_inference.relation_id_to_label[triple[1].item()],
            dataset.inductive_inference.entity_id_to_label[triple[2].item()]
        )
        triples.append(triple)
    return triples

def filter_positive(triple, candidates, direction, all_triples): # I am sorry
    import copy
    filtered_candidates = set()
    too_many_attempts =0 
    
    for candidate in candidates:
        if direction == "s":
            if candidate not in filtered_candidates and (candidate, triple[1], triple[2]) not in all_triples:
                filtered_candidates.add(candidate)
            else:
                killswitch = 10000
                while True:
                    new_candidate = np.random.randint(dataset.inductive_inference.num_entities, size=1)[0]
                    new_candidate = dataset.inductive_inference.entity_id_to_label[new_candidate]
                    if killswitch > 0:
                        if new_candidate not in filtered_candidates and (new_candidate, triple[1], triple[2]) not in all_triples:
                            filtered_candidates.add(new_candidate)
                            break
                    else: # killswitch triggered, find random positive
                        if new_candidate not in filtered_candidates and new_candidate != triple[0]:
                            filtered_candidates.add(new_candidate) # still append positive for being able to construct a tensor afterwards
                            too_many_attempts += 1
                            break
                    if killswitch < -10000:
                        raise
                    killswitch -= 1
        elif direction == "o":
            if candidate not in filtered_candidates and (triple[0], triple[1], candidate) not in all_triples:
                filtered_candidates.add(candidate)
            else:
                killswitch = 10000
                while True:
                    new_candidate = np.random.randint(dataset.inductive_inference.num_entities, size=1)[0]
                    new_candidate = dataset.inductive_inference.entity_id_to_label[new_candidate]
                    if killswitch > 0:
                        if new_candidate not in filtered_candidates and (triple[0], triple[1], new_candidate) not in all_triples:
                            filtered_candidates.add(new_candidate)
                            break
                    else:
                        if new_candidate not in filtered_candidates and new_candidate != triple[2]:
                            filtered_candidates.add(new_candidate) # still append positive for being able to construct a tensor afterwards
                            too_many_attempts += 1
                            break
                    if killswitch < -10000:
                        raise
                    killswitch -= 1
    if too_many_attempts > 0:
        print("For triple", triple, "there are", too_many_attempts, ("tail" if direction == "o" else "head"), f"candidates of {len(filtered_candidates)} still positive")
    
    return list(filtered_candidates)
        
from collections import defaultdict
from tqdm import tqdm

anyburl = False
baseline = True
path = None
        
for dataset_name in ["WN18RR", "nell", "fb237"]: # "WN18RR", "nell", "fb237"
    for version in ["v1", "v2", "v3", "v4"]: # "v1", "v2", "v3", "v4"
        results = defaultdict(list)
        if dataset_name == "WN18RR":
            dataset = InductiveWN18RR(version=version, create_inverse_triples=False)
        elif dataset_name == "nell":
            dataset = InductiveNELL(version=version, create_inverse_triples=False)
        elif dataset_name == "fb237":
            dataset = InductiveFB15k237(version=version, create_inverse_triples=False)
        elif dataset_name == "ilpc_large":
            dataset = ILPC2022Large(create_inverse_triples=False)
        elif dataset_name == "ilpc_small":
            dataset = ILPC2022Small(create_inverse_triples=False)
        else:
            print(dataset_name)
            raise
        rules_heads, rules_tails = get_rules(
            dataset.transductive_training.mapped_triples, 
            dataset.inductive_inference.mapped_triples, 
            dataset.inductive_inference.num_entities, 
            unseen_neg=0, min_prob=0.0, dynamic_prob=True)
            
        model = Wrapper(path, dataset, anyburl=anyburl, baseline=baseline, dynamic_prob=True)
        model.num_entities = dataset.inductive_inference.num_entities
        
        import numpy as np
        import json
        import pickle
        np.random.seed(42)
        
        print(dataset_name, version, dataset.inductive_testing.mapped_triples.shape[0])
        
        id2entity = np.vectorize(dataset.inductive_inference.entity_id_to_label.__getitem__)
        
        # convert all mapped_triples to list of string triples
        test_triples = get_triples_str(dataset.inductive_testing.mapped_triples)
        valid_triples = get_triples_str(dataset.inductive_validation.mapped_triples)
        train_triples = get_triples_str(dataset.inductive_inference.mapped_triples)
            
        # create a set containing all list of string triples in the KG
        all_triples = set()
        all_triples.update(test_triples)
        all_triples.update(valid_triples)
        all_triples.update(train_triples)
        
        # create a hr_t index (int) mapping (used for filtering positives)
        all_hr_t = defaultdict(set)
        all_tr_h = defaultdict(set)
        for (h,r,t) in all_triples:
            h,r,t = dataset.inductive_inference.entity_to_id[h], dataset.inductive_inference.relation_to_id[r], dataset.inductive_inference.entity_to_id[t]
            all_hr_t[(h,r)].add(t)
            all_tr_h[(t,r)].add(h)
            
        threshold1 = 0.75
        threshold2 = 0.25
        
        t_scores = model.predict(dataset.inductive_testing.mapped_triples, "tail", None, None) # get scores of tails
        t_scores[get_positive_tail_mask(dataset.inductive_testing.mapped_triples)] = float("-inf") # set all positives scores to -inf to be not included in topk (at least if there are enough negatives)
        
        randomize = torch.randn_like(t_scores) * 0.001
        
        if not anyburl:
            mask_gt_threshold = (t_scores >= threshold1)
            print(torch.sum(mask_gt_threshold))
            t_scores[mask_gt_threshold] = 1.0
            t_scores[mask_gt_threshold] = t_scores[mask_gt_threshold] + randomize[mask_gt_threshold]
            
            mask_gt_threshold = ((t_scores < threshold1) & (t_scores >= threshold2))
            print(torch.sum(mask_gt_threshold))
            t_scores[mask_gt_threshold] = 0.75
            t_scores[mask_gt_threshold] = t_scores[mask_gt_threshold] + randomize[mask_gt_threshold]
            
            mask_gt_threshold = ((t_scores < threshold2) & (t_scores >= 0.))
            print(torch.sum(mask_gt_threshold))
            t_scores[mask_gt_threshold] = 0.5
            t_scores[mask_gt_threshold] = t_scores[mask_gt_threshold] + randomize[mask_gt_threshold]
        
        mask_gt_threshold = (t_scores < 0.)
        print(torch.sum(mask_gt_threshold))
        t_scores[mask_gt_threshold] = t_scores[mask_gt_threshold] + randomize[mask_gt_threshold]
        
        t_scores = torch.topk(t_scores, largest=True, k=50)
        print((t_scores[0] < 0.).sum(dim=1).float().mean().item(), (t_scores[0] >= 0.).sum(dim=1).float().mean().item())
        negative_tails = t_scores[1]
        # random_entities = torch.vstack([torch.tensor(np.random.choice(dataset.inductive_inference.num_entities, size=50, replace=False)) for _ in range(t_scores[0].shape[0])]) # random entities without replacement
        # negative_tails = torch.where(t_scores[0] < 0., random_entities, t_scores[1]).numpy() # select random entity where positive or no-score
        
        h_scores = model.predict(dataset.inductive_testing.mapped_triples, "head", None, None) # same for heads
        h_scores[get_positive_head_mask(dataset.inductive_testing.mapped_triples)] = float("-inf") # all positive set to -inf
        
        # unpredicted but not positive are -1
        
        randomize = torch.randn_like(h_scores) * 0.001
        
        if not anyburl:
            mask_gt_threshold = (h_scores >= threshold1)
            print(torch.sum(mask_gt_threshold))
            h_scores[mask_gt_threshold] = 1.0
            h_scores[mask_gt_threshold] = h_scores[mask_gt_threshold] + randomize[mask_gt_threshold]
            
            mask_gt_threshold = ((h_scores < threshold1) & (h_scores >= threshold2))
            print(torch.sum(mask_gt_threshold))
            h_scores[mask_gt_threshold] = 0.75
            h_scores[mask_gt_threshold] = h_scores[mask_gt_threshold] + randomize[mask_gt_threshold]
            
            mask_gt_threshold = ((h_scores < threshold2) & (h_scores >= 0.))
            print(torch.sum(mask_gt_threshold))
            h_scores[mask_gt_threshold] = 0.25
            h_scores[mask_gt_threshold] = h_scores[mask_gt_threshold] + randomize[mask_gt_threshold]
        
        mask_gt_threshold = (h_scores < 0.)
        print(torch.sum(mask_gt_threshold))
        h_scores[mask_gt_threshold] = h_scores[mask_gt_threshold] + randomize[mask_gt_threshold]
        
        h_scores = torch.topk(h_scores, largest=True, k=50)
        print((h_scores[0] < 0.).sum(dim=1).float().mean().item(), (h_scores[0] >= 0.).sum(dim=1).float().mean().item())
        negative_heads = h_scores[1]
        # random_entities = torch.vstack([torch.tensor(np.random.choice(dataset.inductive_inference.num_entities, size=50, replace=False)) for _ in range(h_scores[0].shape[0])])
        # negative_heads = torch.where(h_scores[0] < 0., random_entities, h_scores[1]).numpy()
                
        negative_tails = id2entity(negative_tails)
        negative_heads = id2entity(negative_heads)
                    
        json_out = {}
        for i, triple in enumerate(test_triples):
            json_out[triple] = {
                "negative_heads": filter_positive(triple, negative_heads[i], "s", all_triples), # weird filter positive function is still needed as it is still positive we introduced positives in the random entities
                "negative_tails": filter_positive(triple, negative_tails[i], "o", all_triples)
            }
        
        with open(f"./negatives_pkl/{dataset_name}_{version}_neg{'_hard' if anyburl == True else ''}.pkl", "wb") as outfile:
            pickle.dump(json_out, outfile)