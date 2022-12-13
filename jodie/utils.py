import networkx as nx
from itertools import permutations, combinations
import numpy as np
import warnings
from torch import Tensor
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import scipy.stats
from torch.utils.data import Dataset
import torch
from collections import namedtuple
CorrCoeff = namedtuple('corr_coeff', ['correlation', 'pvalue'])


def collect_n_clique_sets(G, clique_size):
    max_cliques = list(nx.algorithms.find_cliques(G))
    clique_sets = set(
        frozenset(clique_p) for clique in max_cliques for clique_p in permutations(clique) if
        len(clique) == clique_size)
    return [list(clique_set) for clique_set in clique_sets]


def determine_triad_class(G, set_indice):
    t = []
    set_indice_length = len(set_indice)
    for (i, j) in combinations(range(set_indice_length), 2):
        t.append(min(G[set_indice[i]][set_indice[j]]['timestamp']))
    times = list(sorted(t))
    times = np.array(times).reshape((1, -1))

    perm = np.array(list(permutations(t)))
    index = np.argmax(np.all(perm == times, axis=1))
    return index


def sort_set_indices_and_assign_labels(G, set_indices):
    labels = []
    for i, set_indice in enumerate(set_indices):
        label = determine_triad_class(G, set_indice)
        labels.append(label)
    labels = np.array(labels)

    return set_indices, labels


def permute(set_indices, labels):
    permutation = np.random.permutation(len(set_indices))
    set_indices = set_indices[permutation]
    labels = labels[permutation]
    return set_indices, labels


def generate_set_indices(G, set_indice_length=3):
    """Generate set indeces, which are used for training/test target sets. But no labels now.
    # only triad prediction, 6 classes.
    """
    print('Generating train/test set indices and labels from graph...')

    clique_samples = collect_n_clique_sets(G, set_indice_length)
    set_indices = np.array(clique_samples)

    permutation = np.random.permutation(len(set_indices))
    set_indices = set_indices[permutation]  # permute
    set_indices, labels = sort_set_indices_and_assign_labels(G, set_indices)

    set_indices, labels = permute(set_indices, labels)

    return G, set_indices, labels


def label_to_order(label, l):
    for i, p in enumerate(permutations(range(l))):
        if i == label:
            return list(p)


def fact_to_num(n):
    return {6: 3, 720: 6}[n]


def bleu_metric(predictions, labels, ngram=3):
    if ngram == 3:
        weights = (1 / 3, 1 / 3, 1 / 3, 0)
    elif ngram == 2:
        weights = (1 / 2, 1 / 2, 0, 0)
    else:
        raise NotImplementedError
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l = fact_to_num(predictions.shape[1])
        predictions = np.argmax(predictions, axis=1).tolist()
        bleu = 0
        for i in range(len(labels)):
            reference = [label_to_order(labels[i], l)]
            candidate = label_to_order(predictions[i], l)
            bleu += sentence_bleu(reference, candidate, weights=weights)
        bleu /= len(labels)
        return bleu


def spearman_metric(predictions, labels):
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    l = fact_to_num(predictions.shape[1])
    predictions = np.argmax(predictions, axis=1)
    tau = 0
    pval = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(labels)):
            metric_val = scipy.stats.spearmanr(label_to_order(predictions[i], l), label_to_order(labels[i], l))
            pred_tau = metric_val.correlation if not np.isnan(metric_val.correlation) else 0.0
            pred_pval = metric_val.pvalue if not np.isnan(metric_val.pvalue) else 0.0
            tau += pred_tau
            pval += pred_pval
        tau /= len(labels)
        pval /= len(labels)
    return CorrCoeff(tau, pval)


def kendall_tau_metric(predictions, labels):
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    l = fact_to_num(predictions.shape[1])
    predictions = np.argmax(predictions, axis=1)
    tau = 0
    pval = 0
    for i in range(len(labels)):
        kendall_tau = scipy.stats.kendalltau(label_to_order(predictions[i], l), label_to_order(labels[i], l))
        pred_tau = kendall_tau.correlation if not np.isnan(kendall_tau.correlation) else 0.0
        pred_pval = kendall_tau.pvalue if not np.isnan(kendall_tau.pvalue) else 0.0
        tau += pred_tau
        pval += pred_pval
    tau /= len(labels)
    pval /= len(labels)
    return CorrCoeff(tau, pval)


def meteor_metric(predictions, labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, Tensor):
            labels = labels.cpu().tolist()
        l = fact_to_num(predictions.shape[1])
        predictions = np.argmax(predictions, axis=1).tolist()
        meteor = 0
        for i in range(len(labels)):
            reference = [[str(s) for s in label_to_order(labels[i], l)]]
            candidate = [str(s) for s in label_to_order(predictions[i], l)]
            meteor += meteor_score(reference, candidate)
        meteor /= len(labels)
        return meteor


# def label_to_order_inc_one(label, l):
#     # Different from remaining label to order functions
#     for i, p in enumerate(permutations(range(1, l+1))):
#         if i == label:
#             return list(p)


class JodieDataset(Dataset):
    def __init__(self, set_indice_length, set_indices, labels, user_embeddings):
        self.set_indice_length = set_indice_length
        self.set_indices = set_indices
        self.labels = labels
        self.user_embeddings = user_embeddings
        self.seq_len = int((set_indice_length * (set_indice_length - 1)) / 2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ue = []
        for k in range(self.set_indice_length):
            ue.append(self.user_embeddings[self.set_indices[idx, k]])
        ue = torch.cat(ue)
        label_seq = label_to_order(self.labels[idx], self.seq_len)
        label_0 = lookup_table_n(self.set_indice_length, return_tensor=True)[label_seq[0]].to(torch.float32)
        label_1 = lookup_table_n(self.set_indice_length, return_tensor=True)[label_seq[1]].to(torch.float32)
        return {'x': ue, 'label': torch.tensor(self.labels[idx]), 'label_seq': torch.tensor(label_seq),
                'label_0': label_0, 'label_1': label_1}


def get_data(edges, ts, set_indice_length):
    G = nx.Graph(edges)

    for epoch, edge in enumerate(edges):
        if G[edge[0]][edge[1]].get('timestamp', None) is None:
            G[edge[0]][edge[1]]['timestamp'] = [ts[epoch]]
        else:
            G[edge[0]][edge[1]]['timestamp'].append(ts[epoch])

    return generate_set_indices(G, set_indice_length)


def lookup_table_n(n=3, return_tensor=False):
    if n == 3:
        lt = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]])

    elif n == 4:
        lt = np.array([[1, 1, 0, 0],
                         [1, 0, 1, 0],
                         [1, 0, 0, 1],
                         [0, 1, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 1],
                         ])
    else:
        raise NotImplementedError

    if return_tensor:
        return torch.tensor(lt)
    return lt


def inverse_lookup(seq, n=3):
    if n == 3:
        if seq == [1, 1, 0]: return 0
        if seq == [1, 0, 1]: return 1
        if seq == [0, 1, 1]: return 2
    if n == 4:
        if seq == [1, 1, 0, 0]: return 0
        if seq == [1, 0, 1, 0]: return 1
        if seq == [1, 0, 0, 1]: return 2
        if seq == [0, 1, 1, 0]: return 3
        if seq == [0, 1, 0, 1]: return 4
        if seq == [0, 0, 1, 1]: return 5
