from pykeen.datasets.inductive.ilp_teru import InductiveFB15k237, InductiveWN18RR, InductiveNELL
from pykeen.datasets.inductive import ILPC2022Large, ILPC2022Small
from pykeen.evaluation.rank_based_evaluator import SampledRankBasedEvaluator, RankBasedEvaluator

from collections import defaultdict
import sys
import argparse
import logging
from torch import nn
import torch
import pickle

logging.getLogger("pykeen.evaluation.rank_based_evaluator").setLevel(logging.ERROR)
logging.getLogger("torch_max_mem.api").setLevel(logging.ERROR)
logging.getLogger("torch_max_mem").setLevel(logging.ERROR)

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
            
            scores_h = torch.full((self.dataset.inductive_inference.num_entities,), float("-inf")).float()
            scores_t = torch.full((self.dataset.inductive_inference.num_entities,), float("-inf")).float()
            
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
            
    
    def __init__(self, path, dataset):
        super().__init__()
        self.device = "cpu"
        self.scores_heads = dict()
        self.scores_tails = dict()
        self.dataset = dataset
        self.heads_not_in_baseline = set()
        self.tails_not_in_baseline = set()
        self.pkl = path.endswith("pkl")
        if not self.pkl:
            self.read_predfile(path)
        else:
            with open(path, "rb") as infile:
                scores_heads, scores_tails = pickle.load(infile)
                scores_heads = dict(scores_heads)
                scores_tails = dict(scores_tails)
                for (r,t) in scores_heads:
                    self.scores_heads[(self.dataset.inductive_inference.relation_to_id[r],self.dataset.inductive_inference.entity_to_id[t])] = torch.full((self.dataset.inductive_inference.num_entities,), float("-inf")).float()
                    for (h,p) in scores_heads[(r,t)]:
                        self.scores_heads[(self.dataset.inductive_inference.relation_to_id[r],self.dataset.inductive_inference.entity_to_id[t])][self.dataset.inductive_inference.entity_to_id[h]] = (p if isinstance(p, float) else p.item())
                for (h,r) in scores_tails:
                    self.scores_tails[(self.dataset.inductive_inference.entity_to_id[h],self.dataset.inductive_inference.relation_to_id[r])] = torch.full((self.dataset.inductive_inference.num_entities,), float("-inf")).float()
                    for (t,p) in scores_tails[(h,r)]:
                        self.scores_tails[(self.dataset.inductive_inference.entity_to_id[h],self.dataset.inductive_inference.relation_to_id[r])][self.dataset.inductive_inference.entity_to_id[t]] = (p if isinstance(p, float) else p.item())
    
            for ix in range(self.dataset.inductive_testing.mapped_triples.shape[0]):
                h,r,t = self.dataset.inductive_testing.mapped_triples[ix].tolist()
                assert len(self.scores_heads[(r,t)]) == dataset.inductive_inference.num_entities
                assert len(self.scores_tails[(h,r)]) == dataset.inductive_inference.num_entities
            
    def score_heads(self, triple):
        triple = tuple(triple.tolist())
        """ print(
            self.dataset.inductive_inference.entity_id_to_label[triple[0]],
            self.dataset.inductive_inference.relation_id_to_label[triple[1]],
            self.dataset.inductive_inference.entity_id_to_label[triple[2]]
        ) """
        if not self.pkl:
            scores = self.scores_heads[triple]
        else:
            scores = self.scores_heads[(triple[1], triple[2])]
        return scores
    
    def score_tails(self, triple):
        triple = tuple(triple.tolist())
        if not self.pkl:
            scores = self.scores_tails[triple] 
        else:
            scores = self.scores_tails[(triple[0], triple[1])]  
        return scores

    def predict(self, hrt_batch, target, slice_size, mode):
        if target == "head":
            return torch.vstack([self.score_heads(triple) for triple in hrt_batch]).to(self.device)
        elif target == "tail":
            return torch.vstack([self.score_tails(triple) for triple in hrt_batch]).to(self.device)
        else:
            raise

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument("--dataset", "-d", type=str, default="sample", choices=["WN18RR", "fb237", "nell", "ilpc_large", "ilpc_small"], help="Dataset name")
    parser.add_argument("--version", "-v", type=str, default="v1", choices=["v1", "v2", "v3", "v4"], help="Dataset version")
    parser.add_argument("--path", "-p", type=str, default="./predictions.txt", help="Path to predictions file (txt or pkl)")
    parser.add_argument('--no_sampled', help='whether to do sampling evalution', action='store_true')
    parser.add_argument('--no_tmn', help='whether to do type matched negative sampling evalution', action='store_true')
    parser.add_argument('--no_all_entities', help='whether to do non-sampling evaluation', action='store_true')
    params = parser.parse_args()

    do_sampled = False if params.no_sampled else True
    do_tmn = False if params.no_sampled else True
    do_all_entities = False if params.no_sampled else True
    dataset_name = params.dataset
    version = params.version
    path = params.path

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
            
        
                
    model = Wrapper(path, dataset)
    model.num_entities = dataset.inductive_inference.num_entities


    def evaluate_sampled(mode, num_runs=100):
        
        assert mode in ["sampled", "tmn"]
        
        head_negatives = None
        tail_negatives = None

        if mode != "sampled":
            num_runs = 1
            path = f"./negatives_pkl/{dataset_name}_{version}_neg.pkl"

            import pickle
            with open(path, "rb") as infile:
                negatives = pickle.load(infile)
            
            head_negatives = []
            tail_negatives = []
            for i in range(dataset.inductive_testing.mapped_triples.shape[0]):
                h_str = dataset.inductive_inference.entity_id_to_label[dataset.inductive_testing.mapped_triples[i][0].item()]
                r_str = dataset.inductive_inference.relation_id_to_label[dataset.inductive_testing.mapped_triples[i][1].item()]
                t_str = dataset.inductive_inference.entity_id_to_label[dataset.inductive_testing.mapped_triples[i][2].item()]
                head_negatives.append([dataset.inductive_inference.entity_to_id[x] for x in negatives[(h_str, r_str, t_str)]["negative_heads"]])
                tail_negatives.append([dataset.inductive_inference.entity_to_id[x] for x in negatives[(h_str, r_str, t_str)]["negative_tails"]])
            head_negatives = torch.tensor(head_negatives)
            tail_negatives = torch.tensor(tail_negatives)
        
        results = defaultdict(list)
        for i in range(num_runs):
            test_evaluator = SampledRankBasedEvaluator(
                mode="testing",   # necessary to specify for the inductive mode - this will use inference nodes
                evaluation_factory=dataset.inductive_testing,  # test triples to predict
                additional_filter_triples=[dataset.inductive_inference.mapped_triples, dataset.inductive_validation.mapped_triples],  # filter out true inference triples
                head_negatives=head_negatives if head_negatives is not None else None,
                tail_negatives=tail_negatives if tail_negatives is not None else None,
            )

            result = test_evaluator.evaluate(
                model=model,
                mapped_triples=dataset.inductive_testing.mapped_triples,
                additional_filter_triples=[dataset.inductive_inference.mapped_triples, dataset.inductive_validation.mapped_triples],
                batch_size=1024,
                use_tqdm=False
            )
            results["hits_at_1"].append(result.to_flat_dict()["both.realistic.hits_at_1"])
            results["hits_at_3"].append(result.to_flat_dict()["both.realistic.hits_at_3"])
            results["hits_at_10"].append(result.to_flat_dict()["both.realistic.hits_at_10"])
            results["inverse_harmonic_mean_rank"].append(result.to_flat_dict()["both.realistic.inverse_harmonic_mean_rank"])
            
        avg_h_10 = sum(results["hits_at_10"]) / len(results["hits_at_10"])
        avg_inverse_harmonic_mean_rank = sum(results["inverse_harmonic_mean_rank"]) / len(results["inverse_harmonic_mean_rank"])
            
        print(dataset_name, version, mode, "h@10", str(avg_h_10).replace(".", ","), "mrr", str(avg_inverse_harmonic_mean_rank).replace(".", ","))
        return avg_h_10, avg_inverse_harmonic_mean_rank
            

    if do_sampled:
        evaluate_sampled("sampled")
    if do_tmn:
        evaluate_sampled("tmn")
    if do_all_entities:
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
        print(dataset_name, version, "all", "h@10", str(result.to_flat_dict()["both.realistic.hits_at_10"]).replace(".", ","), "mrr", str(result.to_flat_dict()["both.realistic.inverse_harmonic_mean_rank"]).replace(".", ","))