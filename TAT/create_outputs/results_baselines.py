import os
from functools import partial

import numpy as np
import pandas as pd
import torch

from TAT.train import acc_metric, bleu_metric, \
    kendall_tau_metric, compute_metric, meteor_metric, spearman_metric

metrics = {
    'acc': acc_metric,
    'BLEU-2': partial(bleu_metric, ngram=2),
    'BLEU-3': partial(bleu_metric, ngram=3),
    'meteor': meteor_metric,
    'kendall_tau': kendall_tau_metric,
    'spearman': spearman_metric
}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

all_results = []
for folder in ['SMS-A-3-baseline', 'SMS-A-3-baseline_p0', 'CollegeMsg-3-baseline',
               'CollegeMsg-4-baseline', 'CollegeMsg-4-baseline_p0']:
    path = os.path.join(ROOT_DIR, f'results/{folder}/')
    full_metrics = []
    for epoch in range(50):
        epoch_results = {}
        for split in ['val', 'test']:
            predictions = torch.load(path + f'{split}_predictions_{epoch}.pt')
            labels = torch.load(path + f'{split}_labels_{epoch}.pt')
            results = compute_metric(predictions, labels, metrics)
            for key in ['kendall_tau', 'spearman']:
                if key in metrics:
                    results[f'{key} correlation'] = results[key].correlation
                    results[f'{key} (p-value)'] = results[key].pvalue
                    del results[key]
            results = {split + '_' + k: v for k, v in results.items()}
            epoch_results.update(results)
        full_metrics.append(epoch_results)
    full_metrics = pd.DataFrame(full_metrics)
    acc_idx = np.argpartition(-full_metrics["val_acc"], 1)[:1]
    test_acc = full_metrics["test_acc"][acc_idx]
    bleu_2_idx = np.argpartition(-full_metrics["val_BLEU-2"], 1)[:1]
    test_bleu_2 = full_metrics["test_BLEU-2"][bleu_2_idx]
    bleu_3_idx = np.argpartition(-full_metrics["val_BLEU-3"], 1)[:1]
    test_bleu_3 = full_metrics["test_BLEU-3"][bleu_3_idx]
    meteor_idx = np.argpartition(-full_metrics["val_meteor"], 1)[:1]
    test_meteor = full_metrics["test_meteor"][meteor_idx]
    idx = np.argpartition(-full_metrics["val_kendall_tau correlation"], 1)[:1]
    test_kt_corr = full_metrics["test_kendall_tau correlation"][idx]
    test_kt_pval = full_metrics["test_kendall_tau (p-value)"][idx]
    idx = np.argpartition(-full_metrics["val_spearman correlation"], 1)[:1]
    test_sp_corr = full_metrics["test_spearman correlation"][idx]
    test_sp_pval = full_metrics["test_spearman (p-value)"][idx]
    all_results.append({'Name': folder, 'Accuracy': np.round(np.mean(test_acc), 3),
                        'BLEU-2': np.round(np.mean(test_bleu_2), 3),
                        'BLEU-3': np.round(np.mean(test_bleu_3), 3),
                        'METEOR': np.round(np.mean(test_meteor), 3),
                        'Kendall Tau': (np.round(np.mean(test_kt_corr), 3), np.round(np.mean(test_kt_pval), 3)),
                        'Spearman': (np.round(np.mean(test_sp_corr), 3), np.round(np.mean(test_sp_pval), 3))
                        })

all_results = pd.DataFrame(all_results)
print(all_results)
