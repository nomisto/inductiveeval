from pykeen.datasets.inductive.ilp_teru import InductiveFB15k237, InductiveWN18RR, InductiveNELL
from pykeen.datasets.inductive import ILPC2022Large, ILPC2022Small
from pykeen.evaluation.rank_based_evaluator import SampledRankBasedEvaluator, RankBasedEvaluator

import logging
logging.getLogger("pykeen.evaluation.rank_based_evaluator").setLevel(logging.ERROR)

logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
logging.getLogger("torch_max_mem.api").setLevel(logging.ERROR)


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

def get_rules(mapped_triples_transductive, mapped_triples_inductive, num_entities):
    
    relations, heads, tails = get_collections(mapped_triples_transductive)
    relations_ind, heads_ind, tails_ind = get_collections(mapped_triples_inductive)

    rules_heads = defaultdict(list)
    rules_tails = defaultdict(list)
    for r1 in relations:
        for r2 in relations:
            for dir1 in ["H", "T"]:
                for dir2 in ["H", "T"]:
                    from_ = heads[r1] if dir1 == "H" else tails[r1]
                    to_ = heads[r2] if dir2 == "H" else tails[r2]
                    from_ind = heads_ind[r1] if dir1 == "H" else tails_ind[r1]
                    to_ind = heads_ind[r2] if dir2 == "H" else tails_ind[r2]
                    intersection = from_.intersection(to_)
                    intersection_ind = from_ind.intersection(to_ind)
                    if len(intersection) > 0:
                        prob = len(intersection) / (len(to_))
                    else:
                        continue
                    predictions = to_ind
                    if dir1 == "H":
                        rules_heads[r1].append((prob, dir2, r2, intersection, predictions))
                    else:
                        rules_tails[r1].append((prob, dir2, r2, intersection, predictions))
                        
    for relation in rules_heads.keys():
        scores_baseline = torch.full((num_entities,), -1).float()
        touched_x = set()
        rules_heads[relation] = list(sorted(rules_heads[relation], key=lambda x: x[0], reverse=True))
        for (prob, dir2, r2, intersection, intersection_ind) in rules_heads[relation]:
            for x in intersection_ind:
                touched_x.add(x)
                if scores_baseline[x] < 0.:
                    scores_baseline[x] = 0.0001 * prob
                else:
                    scores_baseline[x] = max(scores_baseline[x], 0.0001 * prob)
        rules_heads[relation] = scores_baseline
        
    for relation in rules_tails.keys():
        scores_baseline = torch.full((num_entities,), -1).float()
        touched_x = set()
        rules_tails[relation] = list(sorted(rules_tails[relation], key=lambda x: x[0], reverse=True))
        for (prob, dir2, r2, intersection, intersection_ind) in rules_tails[relation]:
            for x in intersection_ind:
                touched_x.add(x)
                if scores_baseline[x] < 0.:
                    scores_baseline[x] = 0.0001 * prob
                else:
                    scores_baseline[x] = max(scores_baseline[x], 0.0001 * prob)
        rules_tails[relation] = scores_baseline
    
    return rules_heads, rules_tails


from torch import nn
import torch

class Wrapper(nn.Module):
    
    def __init__(self, dataset):
        super().__init__()
        self.device = "cpu"
        self.dataset = dataset
    
    def score_heads(self, triple):
        triple = tuple(triple.tolist())                         
        return rules_heads[triple[1]]
    
    def score_tails(self, triple):
        triple = tuple(triple.tolist())
        return rules_tails[triple[1]]
        
    def predict(self, hrt_batch, target, slice_size, mode):
        if target == "head":
            return torch.vstack([self.score_heads(triple) for triple in hrt_batch]).to(self.device)
        elif target == "tail":
            return torch.vstack([self.score_tails(triple) for triple in hrt_batch]).to(self.device)
        else:
            raise
        
from collections import defaultdict
from tqdm import tqdm

for dataset_name in ["fb237", "WN18RR", "nell"]: # "fb237", "WN18RR", "nell"
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
            dataset.inductive_inference.num_entities)
        
        model = Wrapper(dataset)
        model.num_entities = dataset.inductive_inference.num_entities
        
        for i in range(100):
            test_evaluator = SampledRankBasedEvaluator(
                mode="testing",   # necessary to specify for the inductive mode - this will use inference nodes
                evaluation_factory=dataset.inductive_testing,  # test triples to predict
                additional_filter_triples=[dataset.inductive_inference.mapped_triples, dataset.inductive_validation.mapped_triples],  # filter out true inference triples
                num_negatives=50, # clear_on_finalize=False
            )

            result = test_evaluator.evaluate(
                model=model,
                mapped_triples=dataset.inductive_testing.mapped_triples,
                additional_filter_triples=[dataset.inductive_inference.mapped_triples, dataset.inductive_validation.mapped_triples],
                batch_size=1024,
                use_tqdm=False
            )
            results[version].append(result.to_flat_dict()["both.realistic.hits_at_10"])
        print(dataset_name, version, "sampled", "hits@10_avg", str(sum(results[version])/ len(results[version])).replace(".", ","), "hits@10_min", str(min(results[version])).replace(".", ","), "hits@10_max", str(max(results[version])).replace(".", ","))
        
        test_evaluator = RankBasedEvaluator(
            mode="testing",   # necessary to specify for the inductive mode - this will use inference nodes
        )
        model.num_entities = dataset.inductive_inference.num_entities
        result = test_evaluator.evaluate(
            model=model,
            mapped_triples=dataset.inductive_testing.mapped_triples,
            additional_filter_triples=[dataset.inductive_inference.mapped_triples,dataset.inductive_validation.mapped_triples],
            batch_size=1024,
            use_tqdm=False
        )
        print(dataset_name, version, "all", "hits@10", result.to_flat_dict()["both.realistic.hits_at_10"], "mrr", result.to_flat_dict()["both.realistic.inverse_harmonic_mean_rank"])